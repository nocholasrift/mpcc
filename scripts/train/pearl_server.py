#!/usr/bin/env python3

import os
import json
import time
import torch
import rospy
import numpy as np
import oyster.rlkit.torch.pytorch_util as ptu

from std_msgs.msg import Float32
from mpcc.srv import QuerySAC, QuerySACResponse

from gym import spaces
from oyster.RobotEnv import RobotEnv
from oyster.ParamLoader import ParameterLoader
from oyster.rlkit.torch.sac.policies import TanhGaussianPolicy
from oyster.rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from oyster.launch_training import deep_update_dict
from oyster.rlkit.torch.sac.agent import PEARLAgent
from oyster.configs.default import default_config
from oyster.rlkit.torch.sac.policies import MakeDeterministic
from oyster.rlkit.samplers.util import rollout


class ModelServer:
    def __init__(self, variant):

        self.n_obs = 11
        self.n_actions = 2

        low = np.array(np.zeros(self.n_obs), dtype=np.float64)
        high = np.array(np.ones(self.n_obs), dtype=np.float64)

        observation_space = spaces.Box(low, high, dtype=np.float64)
        action_space = spaces.Box(
            low=np.zeros(self.n_actions),
            high=np.ones(self.n_actions),
            dtype=np.float64,
        )

        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))

        latent_dim = variant["latent_size"]
        context_encoder_output_dim = (
            latent_dim * 2
            if variant["algo_params"]["use_information_bottleneck"]
            else latent_dim
        )
        reward_dim = 1
        net_size = variant["net_size"]
        recurrent = variant["algo_params"]["recurrent"]
        encoder_model = RecurrentEncoder if recurrent else MlpEncoder

        config_path = os.getenv("PEARL_CONFIG_PATH")
        param_fname = str(rospy.get_param("model_server/mpc_type")) + ".yaml"
        param_loader = ParameterLoader([os.path.join(config_path, param_fname)])
        self.params = param_loader[0]

        self.context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=(
                2 * obs_dim + action_dim + reward_dim
                if variant["algo_params"]["use_next_obs_in_context"]
                else obs_dim + action_dim + reward_dim
            ),
            output_size=context_encoder_output_dim,
        )

        self.policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        self.agent = PEARLAgent(
            latent_dim, self.context_encoder, self.policy, **variant["algo_params"]
        )
        self.agent = MakeDeterministic(self.agent)
        self.agent.clear_z()

        path_to_exp = os.getenv("PEARL_MODEL_PATH")
        self.context_encoder.load_state_dict(
            torch.load(os.path.join(path_to_exp, "context_encoder.pth"))
        )
        self.policy.load_state_dict(torch.load(os.path.join(path_to_exp, "policy.pth")))

        self.context_encoder.eval().to(ptu.device)
        self.policy.eval().to(ptu.device)

        self.prev_obs = None
        self.prev_action = None
        self.posterior_counter = 0

        rospy.Service("query_sac", QuerySAC, self.query_sac)

        rospy.logerr("*******************************")
        rospy.logerr("Model Server Initialized")
        rospy.logerr("*******************************")

        # ROS Publishers
        self.alpha_dot_abv_pub = rospy.Publisher("alpha_dot_abv", Float32, queue_size=1)
        self.alpha_dot_blw_pub = rospy.Publisher("alpha_dot_blw", Float32, queue_size=1)

    def spin(self):
        rospy.spin()

    def unnormalize(self, val, min, max):
        return (val + 1.0) * (max - min) / 2.0 + min

    def query_sac(self, req):

        obs = list(req.state.state)
        obs = np.array(obs[: self.n_obs])

        # obs = torch.FloatTensor(obs).to(ptu.device)

        print(obs)
        raw_act, _ = self.agent.get_action(obs)

        action = self.unnormalize(
            raw_act, self.params["MIN_ALPHA_DOT"], self.params["MAX_ALPHA_DOT"]
        )

        alpha_abv = obs[9] + action[0] * self.params["DT"]
        alpha_blw = obs[10] + action[1] * self.params["DT"]

        exceed_count = 0
        if alpha_abv < self.params["MIN_ALPHA"] or alpha_abv > self.params["MAX_ALPHA"]:
            exceed_count += 1

        if alpha_blw < self.params["MIN_ALPHA"] or alpha_blw > self.params["MAX_ALPHA"]:
            exceed_count += 1

        # self.params["CBF_ALPHA_ABV"] = np.clip(
        #     alpha_abv, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        # )
        # self.params["CBF_ALPHA_BLW"] = np.clip(
        #     alpha_blw, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        # )

        if self.prev_obs is not None:
            v_max = self.params["LINVEL"]
            v_prog = req.state.state[self.n_obs]
            v_prog /= np.sqrt(2 * v_max**2)
            r = RobotEnv.get_reward(
                obs,
                req.state.solver_status,
                v_prog,
                action,
                exceed_count,
                False,
                self.params,
            )
            self.agent.update_context(
                [self.prev_obs, self.prev_action, r, obs, False, {}]
            )

            if self.posterior_counter > 50:
                self.agent.infer_posterior(self.agent.context)

        self.prev_obs = obs
        self.prev_action = action

        resp = QuerySACResponse()
        resp.alpha_dot = [raw_act[0], raw_act[1]]
        resp.success = True

        alpha_dot_msg = Float32()
        alpha_dot_msg.data = resp.alpha_dot[0]
        self.alpha_dot_abv_pub.publish(alpha_dot_msg)

        alpha_dot_msg.data = resp.alpha_dot[1]
        self.alpha_dot_blw_pub.publish(alpha_dot_msg)

        return resp


def main():
    rospy.init_node("meta_model_server")
    ptu.set_gpu_mode(True)

    variant = default_config
    config_path = os.getenv("PEARL_CONFIG_PATH")
    with open(os.path.join(config_path, "robo-env.json")) as f:
        exp_params = json.load(f)

    variant = deep_update_dict(exp_params, variant)

    model_server = ModelServer(variant)
    model_server.spin()


if __name__ == "__main__":
    main()

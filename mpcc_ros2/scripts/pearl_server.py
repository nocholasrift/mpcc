#!/usr/bin/env python3

import os
import json
import torch
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from mpcc.srv import QuerySAC

import oyster.rlkit.torch.pytorch_util as ptu
from gym import spaces
from oyster.CBFEnv import CBFEnv, RLObs
from oyster.ParamLoader import ParameterLoader
from oyster.rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from oyster.rlkit.torch.networks import MlpEncoder, RecurrentEncoder
from oyster.launch_training import deep_update_dict
from oyster.rlkit.torch.sac.agent import PEARLAgent
from oyster.configs.default import default_config


class ModelServer(Node):
    def __init__(self, variant):
        super().__init__("meta_model_server")

        # -------------------------
        # Params (ROS2 requires declare)
        # -------------------------
        self.declare_parameter("param_file", "")
        param_file = self.get_parameter("param_file").get_parameter_value().string_value

        # -------------------------
        # Dimensions
        # -------------------------
        self.N_alpha = 2
        self.N_horizon = 3

        self.n_obs = len(RLObs) * self.N_horizon + self.N_alpha + 1
        self.n_actions = self.N_alpha

        self.low = -10 * np.ones(self.n_obs)
        self.high = 10 * np.ones(self.n_obs)

        observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
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

        # -------------------------
        # Load params
        # -------------------------
        config_path = os.getenv("MPCC_CONFIG_PATH")
        if not config_path:
            self.get_logger().error("MPCC_CONFIG_PATH not set")
            raise RuntimeError

        fname = os.path.join(config_path, param_file + ".yaml")
        self.get_logger().info(f"Loading param file: {fname}")

        param_loader = ParameterLoader([fname])
        self.params = param_loader[0]

        # -------------------------
        # Networks
        # -------------------------
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

        # -------------------------
        # Load weights
        # -------------------------
        path_to_exp = os.getenv("PEARL_MODEL_PATH")
        if not path_to_exp:
            self.get_logger().error("PEARL_MODEL_PATH not set")
            raise RuntimeError

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        itr = 74
        self.context_encoder.load_state_dict(
            torch.load(os.path.join(path_to_exp, f"context_encoder_itr_{itr}.pth"), map_location=device)
        )
        self.policy.load_state_dict(
            torch.load(os.path.join(path_to_exp, f"policy_itr_{itr}.pth"), map_location=device)
        )

        self.context_encoder.eval().to(device)
        self.policy.eval().to(device)

        # -------------------------
        # Agent
        # -------------------------
        self.obs_mu, self.obs_std = CBFEnv.get_mu_and_std(
            self.N_horizon, self.N_alpha, self.params
        )

        self.agent = PEARLAgent(
            latent_dim, self.context_encoder, self.policy, **variant["algo_params"]
        )
        self.agent = MakeDeterministic(self.agent)
        self.agent.clear_z()

        # -------------------------
        # State
        # -------------------------
        self.prev_obs = None
        self.prev_action = None

        # -------------------------
        # ROS2 interfaces
        # -------------------------
        self.srv = self.create_service(QuerySAC, "query_sac", self.query_sac)

        self.alpha_dot_abv_pub = self.create_publisher(Float32, "alpha_dot_abv", 10)
        self.alpha_dot_blw_pub = self.create_publisher(Float32, "alpha_dot_blw", 10)

        self.get_logger().info("Model Server Initialized")

    # -------------------------
    # Utils
    # -------------------------
    def normalize_obs(self, obs):
        z = (obs - self.obs_mu) / (2 * self.obs_std)
        return np.clip(z, self.low, self.high)

    def unnormalize(self, val, min_val, max_val):
        return (val + 1.0) * (max_val - min_val) / 2.0 + min_val

    # -------------------------
    # Service callback
    # -------------------------
    def query_sac(self, req, resp):
        # print("HELLO?")
        obs = np.array(req.state.state)
        unnormalized_obs = obs.copy()

        self.get_logger().info(f"obs: {obs}")

        obs = self.normalize_obs(obs)

        self.get_logger().info(f"normalized obs: {obs}")

        raw_act, _ = self.agent.get_action(obs)

        action = self.unnormalize(
            raw_act,
            self.params["MIN_ALPHA_DOT"],
            self.params["MAX_ALPHA_DOT"],
        )

        alpha_abv = unnormalized_obs[-2] + action[0] * self.params["DT"]
        alpha_blw = unnormalized_obs[-1] + action[1] * self.params["DT"]

        # context update
        if self.prev_obs is not None:
            r = CBFEnv.get_reward(
                unnormalized_obs, False, self.params, self.N_horizon, action
            )
            self.agent.update_context(
                [self.prev_obs, self.prev_action, r, obs, False, {}]
            )

        self.prev_obs = obs
        self.prev_action = action

        # response
        resp.alpha_dot = [float(raw_act[0]), float(raw_act[1])]
        resp.success = True

        # publish
        msg = Float32()

        msg.data = resp.alpha_dot[0]
        self.alpha_dot_abv_pub.publish(msg)

        msg.data = resp.alpha_dot[1]
        self.alpha_dot_blw_pub.publish(msg)

        return resp


def main():
    rclpy.init()

    ptu.set_gpu_mode(torch.cuda.is_available())

    variant = default_config
    config_path = os.getenv("PEARL_CONFIG_PATH")

    with open(os.path.join(config_path, "robo-env.json")) as f:
        exp_params = json.load(f)

    variant = deep_update_dict(exp_params, variant)

    node = ModelServer(variant)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


from logging import exception
import os
import sys
import torch
import yaml
import random
import sqlite3
import numpy as np
import gymnasium as gym
import rospkg

import oyster.rlkit.torch.pytorch_util as ptu

from mpcc.msg import RLState

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from oyster.configs.default import default_config
from oyster.rlkit.torch.sac.agent import PEARLAgent
from oyster.rlkit.torch.sac.sac import PEARLSoftActorCritic
from oyster.rlkit.torch.networks import FlattenMlp, MlpEncoder
from oyster.rlkit.torch.sac.policies import TanhGaussianPolicy


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.state_dim = 11
        self.action_dim = 2

    def set_obs_space(self):

        low = np.array(
            np.zeros(self.state_dim),
            dtype=np.float64,
        )
        high = np.array(
            np.ones(self.state_dim),
            dtype=np.float64,
        )

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

        self.state = np.zeros(self.state_dim, dtype=np.float64)

    def set_action_space(self):
        # define action space
        self.action_space = gym.spaces.Box(
            low=np.array(-1.0),
            high=np.array(1.0),
            dtype=np.float64,
        )

        ptu.set_gpu_mode(True)


class TrainManager:
    def __init__(self, buffer_fname, log_fname, batch_size, env):

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_fname)
        self.global_step = 0

        self.variant = default_config

        self.tasks = list(range(self.variant["n_train_tasks"]))
        self.task_db_start_ids = [0] * len(self.tasks)

        rospack = rospkg.RosPack()
        base_path = rospack.get_path("mpcc")
        param_file = os.path.join(base_path, "params", "meta_train.yaml")

        with open(param_file, "r") as f:
            yaml_data = yaml.safe_load(f)

        self.batch_size = batch_size
        self.buffer_fname = buffer_fname

        self.obs_dim = env.state_dim
        self.action_dim = env.action_dim
        self.latent_dim = yaml_data["latent_dims"]
        self.hidden_dim = yaml_data["hidden_dims"]
        self.reward_dim = 1

        self.buffer_size = yaml_data["buffer_size"]

        context_encoder_input_dim = (
            2 * self.obs_dim + self.action_dim + self.reward_dim
            if self.variant["algo_params"]["use_next_obs_in_context"]
            else self.obs_dim + self.action_dim + self.reward_dim
        )
        context_encoder_output_dim = (
            self.latent_dim * 2
            if self.variant["algo_params"]["use_information_bottleneck"]
            else self.latent_dim
        )

        # Define networks
        self.context_encoder = MlpEncoder(
            hidden_sizes=[200, 200],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
        )
        self.qf1 = FlattenMlp(
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.qf2 = FlattenMlp(
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.vf = FlattenMlp(
            input_size=self.obs_dim + self.latent_dim,
            output_size=1,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.policy = TanhGaussianPolicy(
            obs_dim=self.obs_dim + self.latent_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_sizes=[self.hidden_dim, self.hidden_dim],
        ).to(ptu.device)
        self.agent = PEARLAgent(
            self.latent_dim,
            self.context_encoder,
            self.policy,
            **self.variant["algo_params"],
        )

        self.env = env

        # Define SAC trainer
        self.algorithm = PEARLSoftActorCritic(
            env=env,
            train_tasks=list(self.tasks[: self.variant["n_train_tasks"]]),
            eval_tasks=list(self.tasks[-self.variant["n_eval_tasks"] :]),
            nets=[self.agent, self.qf1, self.qf2, self.vf],
            latent_dim=self.latent_dim,
            **self.variant["algo_params"],
        )
        self.algorithm.sample_sac = self.sample_task_buffer
        self.algorithm.sample_context = self.sample_encoder_buffer

        # try to connect to the database
        self.conn = None
        try:
            self.conn = sqlite3.connect(buffer_fname)
        except Exception as e:
            print("Error connecting to database: ", str(e))
            sys.exit(-1)

    def deserialize_state(self, serialized_str, msg_type):
        try:
            serialized_bytes = bytes.fromhex(serialized_str)
        except ValueError as e:
            print("Error deserializing state: ", str(e))
            print("serialized_str: ", serialized_str)
            return None

        new_msg = msg_type()
        new_msg.deserialize(serialized_bytes)

        ret = []
        for elem in new_msg.state:
            ret.append(elem)

        ret.append(1 if new_msg.solver_status else 0)

        ret = np.array(ret, dtype=np.float64)

        return ret

    def sample_task_buffer(self, task_ids):

        all_batches = []

        for task_id in task_ids:
            cur = self.conn.cursor()
            try:

                query_limit_recent = f"""
                SELECY * FROM (
                    SELECT * FROM replay_buffer
                    WHERE task_id = {task_id}
                    ORDER BY global_id DESC
                    LIMIT {self.buffer_size}
                ) ORDER BY RANDOM()
                LIMIT {self.batch_size}
                """

                cur.execute(query_limit_recent)
                recent_entries = cur.fetchall()

            except Exception as e:
                print("Failed to get entries from replay buffer: ", str(e))
                return [], [], [], [], []

            processed = self.process_query(recent_entries, cur.description)
            all_batches.append(processed)

        # need to return in (task, batch, feature)
        return [np.stack(x) for x in zip(*all_batches)]

    def train_sac(self):

        indices = np.random.choice(
            self.tasks, self.variant["algo_params"]["meta_batch"]
        )

        self.algorithm._do_training(indices)

        self.global_step += 1

    def close(self):
        self.writer.close()

    def start_epoch(self, epoch):
        self.algorithm._start_epoch(epoch)

    def training_mode(self, mode):
        self.algorithm.training_mode(mode)

    def process_query(self, rows, descriptions):

        labels = [description[0] for description in descriptions]

        # make mapping of label to index
        label_to_index = {label: i for i, label in enumerate(labels)}

        # return states, actions, rewards, next_states, dones
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for row in rows:
            prev_state = self.deserialize_state(
                row[label_to_index["prev_state"]], RLState
            )
            next_state = self.deserialize_state(
                row[label_to_index["next_state"]], RLState
            )
            action = np.array(
                row[label_to_index["action"]].split(","), dtype=np.float64
            )
            action = np.array([float(a) for a in action], dtype=np.float64)
            reward = np.array(float(row[label_to_index["reward"]]), dtype=np.float64)
            done = np.array(bool(row[label_to_index["is_done"]]), dtype=np.float64)

            states.append(prev_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones

    def clear_encoder_buffer(self, task_id):
        cur = self.conn.cursor()
        try:
            global_id_query = (
                f"""SELECT MAX(id) FROM replay_buffer WHERE task_id = {task_id}"""
            )
            cur.execute(global_id_query)
            max_id = cur.fetchone()[0]
        except Exception as e:
            print("Failed to clear encoder buffer: ", str(e))
            return

        if max_id is not None:
            self.task_db_start_ids[task_id] = max_id
        else:
            self.task_db_start_ids[task_id] = 0

    def sample_encoder_buffer(self, task_ids):

        all_batches = []
        batch_sz = self.variant["algo_params"]["embedding_batch_size"]

        for task_id in task_ids:
            cur = self.conn.cursor()
            try:
                query_limit_recent = f"""
                SELECT * FROM replay_buffer
                WHERE global_id > {self.task_db_start_ids[task_id]} AND task_id = {task_id}
                """

                cur.execute(query_limit_recent)
                recent_entries = cur.fetchall()

                # Randomly sample from the limited data set
                if len(recent_entries) < batch_sz:
                    random_sample = recent_entries
                else:
                    random_sample = random.sample(recent_entries, batch_sz)

            except Exception as e:
                print("Failed to get entries from replay buffer: ", str(e))
                return [], [], [], [], []

            processed = self.process_query(random_sample, cur.description)
            all_batches.append(processed)

        return [np.stack(x) for x in zip(*all_batches)]

    def encoder_buffer_size(self, task_id):
        cur = self.conn.cursor()
        try:
            query_limit_recent = f"""
            SELECT COUNT(*) FROM replay_buffer
            WHERE global_id > {self.task_db_start_ids[task_id]} AND task_id = {task_id}
            """
            cur.execute(query_limit_recent)
            count = cur.fetchone()[0]
        except Exception as e:
            print("Failed to get entries from replay buffer: ", str(e))
            return 0

        return count

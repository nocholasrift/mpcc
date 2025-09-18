import gym
import numpy as np

from gym import spaces
from py_mpcc import MPCCore


class RobotEnv(gym.Env):
    def __init__(self):

        super(RobotEnv, self).__init__()

        self.state_dim = 13
        self.action_dim = 2
        self.high = np.array(
            np.ones(self.state_dim),
            dtype=np.float64,
        )

        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float64,
        )

        # robot state: x, y, vx, vy
        self.robot_state = np.zeros(4, dtype=np.float64)

        params = {
            "DT": 0.1,
            "STEPS": 10,
            "ANGVEL": 3.2,
            "LINVEL": 2.0,
            "MAX_LINACC": 1.5,
            "MAX_ANGA": 6.28,
            "BOUND": 1e3,
            "W_ANGVEL": 0.15,
            "W_DANGVEL": 0.7,
            "W_DA": 0.1,
            "W_LAG": 100,
            "W_CONTOUR": 0.8,
            "W_SPEED": 1.0,
            "REF_LENGTH": 6,
            "REF_SAMPLES": 11,
            "CLF_GAMMA": 0.5,
            "CLF_W_LAG": 1,
            "USE_CBF": False,
            "CBF_ALPHA_ABV": 1.1,
            "CBF_ALPHA_BLW": 1.1,
            "CBF_COLINEAR": 0.1,
            "CBF_PADDING": 0.1,
        }

        self.dt = params["DT"]
        self.v_max = params["LINVEL"]
        self.ref_len = params["REF_LENGTH"]

        self.prev_s = 0.0

        self.mpc = MPCCore("double_integrator")
        self.mpc.load_params(params)

        self.knots = np.linspace(0, 10, 100).reshape(-1, 1)
        self.traj_x = np.linspace(0, 10, 100).reshape(-1, 1)
        self.traj_y = np.linspace(0, 0, 100).reshape(-1, 1)

        self.mpc.set_trajectory(self.traj_x, self.traj_y, 3, self.knots)

        self.reset()

    def step(self, action):
        len_start = self.mpc.get_s_from_pose(self.robot_state[:2])
        s_dot = min(max((len_start - self.prev_s) / self.dt, 0), self.v_max)

        state = np.concatenate((self.robot_state, np.array([0, s_dot])))
        u = self.mpc.solve(state, False)

        # apply dynamics
        self.robot_state[0] += self.robot_state[2] * self.dt
        self.robot_state[1] += self.robot_state[3] * self.dt
        self.robot_state[2] += u[0] * self.dt
        self.robot_state[3] += u[1] * self.dt

    def reset(self):
        pass

    def _get_obs(self):
        pass


if __name__ == "__main__":
    env = RobotEnv()

    env.step([0])

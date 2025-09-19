from logging import exception
import gym
import time
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from py_mpcc import MPCCore
from matplotlib.patches import Circle


def normalize(val, min, max):
    return (val - min) / (max - min)


def unnormalize(val, min, max):
    return val * (max - min) + min


class BezierCurve:
    def __init__(self, p0, p1, p2, p3):
        self.p0 = np.array(p0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.p3 = np.array(p3, dtype=float)

    def pos(self, t):
        """Position on curve at param t ∈ [0,1]"""
        return (
            (1 - t) ** 3 * self.p0
            + 3 * (1 - t) ** 2 * t * self.p1
            + 3 * (1 - t) * t**2 * self.p2
            + t**3 * self.p3
        )

    def vel(self, t):
        """Derivative wrt t"""
        return (
            3 * (1 - t) ** 2 * (self.p1 - self.p0)
            + 6 * (1 - t) * t * (self.p2 - self.p1)
            + 3 * t**2 * (self.p3 - self.p2)
        )

    def compute_arclen(self, t0=0.0, tf=1.0, n=100):
        """Numerical arc length using trapezoidal rule"""
        ts = np.linspace(t0, tf, n + 1)
        vels = np.array([self.vel(t) for t in ts])
        speeds = np.linalg.norm(vels, axis=1)
        return np.trapz(speeds, ts)

    def binary_search(self, target_s, start=0.0, end=1.0, tol=1e-4):
        """Find t such that arclen(0,t) ≈ target_s"""
        t_left, t_right = start, end
        prev_s, s = 0, -1e9

        while abs(prev_s - s) > tol:
            prev_s = s
            t_mid = (t_left + t_right) / 2.0
            s = self.compute_arclen(0, t_mid)
            if s < target_s:
                t_left = t_mid
            else:
                t_right = t_mid

        return 0.5 * (t_left + t_right)

    def reparam_traj(self, M=20):
        """Return M+1 samples evenly spaced in arc length"""
        total_len = self.compute_arclen(0, 1)
        ds = total_len / M

        ss, xs, ys = [], [], []
        prev_t = 0.0
        for i in range(M + 1):
            s = i * ds
            ti = self.binary_search(s, prev_t, 1.0)
            pos = self.pos(ti)
            ss.append(s)
            xs.append(pos[0])
            ys.append(pos[1])
            prev_t = ti

        return np.array(ss), np.array(xs), np.array(ys)


class RobotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):

        super(RobotEnv, self).__init__()

        self.state_dim = 13
        self.action_dim = 2
        self.low = np.array(
            np.zeros(self.state_dim),
            dtype=np.float64,
        )

        self.high = np.array(
            np.ones(self.state_dim),
            dtype=np.float64,
        )

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=np.zeros(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float64,
        )

        # for some reason this is needed for PEARL, but it can be
        # set to anything
        self._goal = None

        self.params = {
            "DT": 0.1,
            "STEPS": 10,
            "ANGVEL": 3.2,
            "LINVEL": 1.0,
            "MAX_LINACC": 1.5,
            "MAX_ANGA": 6.28,
            "BOUND": 1e3,
            "W_ANGVEL": 0.15,
            "W_DANGVEL": 0.7,
            "W_DA": 0.1,
            "W_LAG": 100,
            "W_CONTOUR": 0.8,
            "W_SPEED": 0.1,
            "REF_LENGTH": 6,
            "REF_SAMPLES": 11,
            "CLF_GAMMA": 0.5,
            "CLF_W_LAG": 1,
            "USE_CBF": True,
            "CBF_ALPHA_ABV": 0.05,
            "CBF_ALPHA_BLW": 0.05,
            "CBF_COLINEAR": 0.1,
            "CBF_PADDING": 0.1,
        }

        p0 = [0, 0]
        p1 = [4.2, 13.6]
        p2 = [5, -8.6]
        p3 = [10.0, 10.0]

        curve = BezierCurve(p0, p1, p2, p3)
        self.knots, self.traj_x, self.traj_y = curve.reparam_traj(M=20)

        # robot state: x, y, vx, vy
        self.robot_state = np.zeros(4, dtype=np.float64)
        self.robot_state[:2] = np.array([self.traj_x[0], self.traj_y[0]]).flatten()

        self.current_ref = (self.traj_x[0], self.traj_y[0])

        self._plot_init()
        self._mpc_init()

        self.reset()

    def step(self, action):

        self.len_start = self.mpc.get_s_from_pose(self.robot_state[:2])

        if self.len_start > self.knots[-1] - 1e-2:
            self.robot_state[2] = 0.0
            self.robot_state[3] = 0.0
        else:
            s_dot = min(
                max((self.len_start - self.prev_s) / self.dt, 0),
                np.sqrt(2 * self.v_max**2),
            )
            self.prev_s = self.len_start

            action = unnormalize(action, 0.0, 3.0)
            self.params["CBF_ALPHA_ABV"] += action[0] * self.dt
            self.params["CBF_ALPHA_BLW"] += action[1] * self.dt
            self.mpc.load_params(self.params)

            state = np.concatenate((self.robot_state, np.array([0, s_dot])))
            u = self.mpc.solve(state, False)

            self.robot_state[2] = u[0]
            self.robot_state[3] = u[1]

        # apply dynamics
        self.robot_state[0] += self.robot_state[2] * self.dt
        self.robot_state[1] += self.robot_state[3] * self.dt

        self.robot_state[2] = max(min(self.robot_state[2], self.v_max), -self.v_max)
        self.robot_state[3] = max(min(self.robot_state[3], self.v_max), -self.v_max)

        idx = (np.abs(self.knots.flatten() - self.len_start)).argmin()
        self.current_ref = (self.traj_x[idx], self.traj_y[idx])

        return (self._get_obs(), self._get_reward(), False, {})

    def reset(self):
        self.robot_state = np.zeros(4, dtype=np.float64)
        obs = np.zeros(13, dtype=np.float64)
        obs[4] = 0.5
        obs[5] = -0.5
        obs[10] = 0.05
        obs[11] = 0.05

        return obs

    def _get_obs(self):
        mpc_state = self.mpc.get_state()
        mpc_input = self.mpc.get_mpc_command()
        solver_status = self.mpc.get_solver_status()

        cbf_data_abv = self.mpc.get_cbf_data(mpc_state, mpc_input, True)
        cbf_data_blw = self.mpc.get_cbf_data(mpc_state, mpc_input, False)

        alpha_abv = self.mpc.get_params()["CBF_ALPHA_ABV"]
        alpha_blw = self.mpc.get_params()["CBF_ALPHA_BLW"]

        curr_progress = mpc_state[5] / np.sqrt(2 * self.v_max**2)

        state_limits = self.mpc.get_state_limits()
        input_limits = self.mpc.get_input_limits()

        obs = np.zeros(13, dtype=np.float64)
        obs[0] = normalize(mpc_state[2], state_limits[0][2], state_limits[1][2])
        obs[1] = normalize(mpc_state[3], state_limits[0][3], state_limits[1][3])
        obs[2] = normalize(mpc_input[0], input_limits[0][0], input_limits[1][0])
        obs[3] = normalize(mpc_input[1], input_limits[0][1], input_limits[1][0])
        obs[4] = normalize(cbf_data_abv[1], 0, 1.0)
        obs[5] = normalize(cbf_data_blw[1], 0, 1.0)
        obs[6] = normalize(cbf_data_abv[2], -np.pi, np.pi)
        obs[7] = curr_progress
        obs[8] = normalize(cbf_data_abv[0], 0, 100)
        obs[9] = normalize(cbf_data_blw[0], 0, 100)
        obs[10] = normalize(alpha_abv, 0, 5)
        obs[11] = normalize(alpha_blw, 0, 5)
        obs[12] = float(solver_status)

        return obs

    def _get_reward(self):
        return 1

    def _mpc_init(self):
        self.tube_degree = 7
        upper_coeffs = np.zeros(self.tube_degree)
        lower_coeffs = np.zeros(self.tube_degree)
        upper_coeffs[0] = 0.5
        lower_coeffs[0] = -0.5

        self.dt = self.params["DT"]
        self.v_max = self.params["LINVEL"]
        self.ref_len = self.params["REF_LENGTH"]

        self.prev_s = 0.0

        self.mpc = MPCCore("double_integrator")
        self.mpc.load_params(self.params)
        self.mpc.set_tubes([upper_coeffs, lower_coeffs])
        self.mpc.set_trajectory(self.traj_x, self.traj_y, 3, self.knots)

    def _plot_init(self):
        # visualization setup
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")

        margin = 2.0
        x_min, x_max = float(np.min(self.traj_x)), float(np.max(self.traj_x))
        y_min, y_max = float(np.min(self.traj_y)), float(np.max(self.traj_y))

        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)

        (self.traj_line,) = self.ax.plot(
            self.traj_x, self.traj_y, "k--", label="trajectory"
        )
        self.robot_patch = Circle((0, 0), 0.2, color="blue", label="robot")
        (self.ref_point,) = self.ax.plot([], [], "ro", label="reference")

        # tubes (initialized as empty lines)
        (self.upper_tube_line,) = self.ax.plot([], [], "r-", label="upper tube")
        (self.lower_tube_line,) = self.ax.plot([], [], "b-", label="lower tube")

        self.ax.add_patch(self.robot_patch)
        self.ax.legend()

    def _plot_tubes(self, tube_radius=0.5):
        # figure out where to start and stop the tubes
        if self.len_start > self.knots[-1] - 1e-2:
            return

        len_stop = min(self.len_start + self.ref_len, self.knots[-1])

        mask = (self.knots >= self.len_start) & (self.knots <= len_stop)
        traj = np.vstack([self.traj_x[mask], self.traj_y[mask]]).T

        try:
            # compute tangents using gradient
            tangents = np.gradient(traj, axis=0)
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        except Exception:
            return

        # compute normals
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        # offset trajectory
        upper = traj + tube_radius * normals
        lower = traj - tube_radius * normals

        # plot tubes
        self.upper_tube_line.set_data(upper[:, 0], upper[:, 1])
        self.lower_tube_line.set_data(lower[:, 0], lower[:, 1])

    def render(self, mode="human"):
        # update robot position
        self.robot_patch.center = (self.robot_state[0], self.robot_state[1])
        self.ref_point.set_data(self.current_ref[0], self.current_ref[1])

        self._plot_tubes()

        plt.pause(0.001)
        return self.ax

    def get_all_task_idx(self):
        return [0]

    def reset_task(self, idx):
        self.reset()


if __name__ == "__main__":
    env = RobotEnv()

    for _ in range(5):
        env.step([0, 0])
        env.render()

    env._get_obs()

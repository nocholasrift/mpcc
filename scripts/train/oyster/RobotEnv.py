import os
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from Plotter import Plotter
from Bezier import BezierCurve
from Tubes import TubeGenerator
from RobotMPC import RobotMPC
from ParamLoader import ParameterLoader


def normalize(val, min, max):
    return (val - min) / (max - min)


def unnormalize(val, min, max):
    return val * (max - min) + min


class RobotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, task={}, n_tasks=2, randomize_tasks=False, n_obs=100):

        super(RobotEnv, self).__init__()

        # each task is loaded from parameter list
        self.task_loader = self.load_tasks()

        if len(self.task_loader) == 0:
            raise ValueError("No parameter files found!")

        self.task_idx = 0

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

        # p0 = [0, 0]
        # p1 = [4.2, 13.6]
        # p2 = [5, -8.6]
        # p3 = [10.0, 10.0]
        p0 = [0, 0]
        p1 = [10,11.8]
        p2 = [10,-9.9]
        p3 = [10.0, 10.0]

        self.curve = BezierCurve(p0, p1, p2, p3)
        self.current_ref = self.curve.pos(0.0)

        self._obs_init(n_obs, min_dist=0.6)

        self.tube_gen = TubeGenerator(
            self.obstacles, (self.curve.knots, self.curve.xs, self.curve.ys)
        )
        (
            self.d_parallel_p,
            self.perp_dists_p,
            self.d_parallel_n,
            self.perp_dists_n,
        ) = self.tube_gen.get_dists()

        self.upper_coeffs, self.lower_coeffs = self.tube_gen.generate_corridor()

        self.plotter = None
        self.set_mpc(self.task_loader[self.task_idx])
        self.reset()

    def set_mpc(self, params):

        self.params = copy.deepcopy(params)
        self.dynamic_model = self.params["DYNAMIC_MODEL"]

        self.mpc = RobotMPC(self.curve.pos(0.0), self.params)
        self.robot_state = self.mpc.get_robot_state()


        self.mpc.set_trajectory(
            self.curve.xs,
            self.curve.ys,
            self.curve.knots,
        )

    def load_tasks(self):
        param_path = os.path.join(os.path.dirname(__file__), "configs")
        fnames = []
        for file in os.listdir(param_path):
            if file.endswith(".yaml"):
                fnames.append(os.path.join(param_path, file))

        return ParameterLoader(fnames)

    def step(self, action):

        len_start = self.mpc.get_s_from_pose(self.robot_state[:2])

        upper_coeffs, lower_coeffs = self.tube_gen.shift_poly_parameter(
            len_start, self.mpc.get_params()["REF_LENGTH"]
        )

        self.mpc.set_tubes(upper_coeffs, lower_coeffs)

        action[0] = unnormalize(action[0], 0.0, 3.0)
        action[1] = unnormalize(action[1], 0.0, 3.0)

        self.params = self.mpc.get_params()

        dt = self.params["DT"]
        exceeded_bounds_abv = False
        exceeded_bounds_blw = False
        alpha_abv = self.params["CBF_ALPHA_ABV"] + action[0] * dt
        alpha_blw = self.params["CBF_ALPHA_BLW"] + action[1] * dt

        if alpha_abv < self.params["MIN_ALPHA"] or alpha_abv > self.params["MAX_ALPHA"]:
            exceeded_bounds_abv = True

        if alpha_blw < self.params["MIN_ALPHA"] or alpha_blw > self.params["MAX_ALPHA"]:
            exceeded_bounds_blw = True

        self.params["CBF_ALPHA_ABV"] = np.clip(
            alpha_abv, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        )
        self.params["CBF_ALPHA_BLW"] = np.clip(
            alpha_blw, self.params["MIN_ALPHA"], self.params["MAX_ALPHA"]
        )

        self.mpc.load_params(self.params)

        u = self.mpc.get_control(len_start)
        self.robot_state = self.mpc.apply_control(u)

        idx = (np.abs(self.curve.knots.flatten() - len_start)).argmin()
        self.current_ref = (self.curve.xs[idx], self.curve.ys[idx])

        obs = self._get_obs()

        # check if done
        # figure out which side of trajectory we are on
        tangent = self.curve.vel(len_start)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])

        traj_p = np.array([self.curve.trajx(len_start), self.curve.trajy(len_start)])
        to_robot = self.robot_state[:2] - traj_p
        side = np.sign(np.dot(to_robot, normal))

        dist = self._dist_from_traj(self.robot_state[:2])

        is_colliding = False
        if side < 0:
            is_colliding = dist > np.polyval(self.lower_coeffs[::-1], len_start)
        else:
            is_colliding = dist > np.polyval(self.upper_coeffs[::-1], len_start)


        len_start = self.mpc.get_s_from_pose(self.robot_state[:2])
        is_done = len_start > self.curve.knots[-1] - 2e-1


        return (
            obs,
            self._get_reward(obs, exceeded_bounds_abv, exceeded_bounds_blw, is_colliding),
            # self._get_reward(obs, exceeded_bounds_blw or exceeded_bounds_abv, is_colliding),
            is_colliding or is_done,
            {},
        )

    def reset(self):
        plt.close("all")

        self.set_mpc(self.task_loader[self.task_idx])

        self.plotter = None
        self.robot_state = np.zeros(4, dtype=np.float64)
        self.mpc.set_mpc_state(self.robot_state)

        obs = np.zeros(13, dtype=np.float64)
        obs[4] = 1.0
        obs[5] = -1.0
        obs[10] = 0.0
        obs[11] = 0.0

        return obs

    def render(self, mode="human", close=False):
        if close:
            if self.plotter:
                plt.close(self.plotter.fig)
                self.plotter = None
            return None

        if self.plotter is None:
            self.plotter = Plotter(
                self.curve,
                self.upper_coeffs,
                self.lower_coeffs,
                self.obstacles,
                self.dynamic_model,
                0.25,
            )

        self.plotter.add_state_to_path(self.robot_state[:2])
        self.plotter.render(
            self.robot_state, self.current_ref, self.curve, self.tube_gen, self.mpc
        )
        plt.pause(0.001)

        return self.plotter.ax

    def get_all_task_idx(self):
        return range(len(self.task_loader))

    def reset_task(self, idx):
        plt.close("all")

        self.task_idx = idx

        self.reset()

    def _get_obs(self):
        mpc_state = self.mpc.get_mpc_state()
        mpc_input = self.mpc.get_mpc_command()
        solver_status = self.mpc.get_solver_status()

        cbf_data_abv = self.mpc.get_cbf_data(mpc_state, mpc_input, True)
        cbf_data_blw = self.mpc.get_cbf_data(mpc_state, mpc_input, False)

        params = self.mpc.get_params()
        alpha_abv = params["CBF_ALPHA_ABV"]
        alpha_blw = params["CBF_ALPHA_BLW"]

        v_max = params["LINVEL"]
        curr_progress = mpc_state[5] / np.sqrt(2 * v_max**2)

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
        obs[8] = normalize(cbf_data_abv[0], -100, 100)
        obs[9] = normalize(cbf_data_blw[0], -100, 100)
        obs[10] = normalize(alpha_abv, 0, 5)
        obs[11] = normalize(alpha_blw, 0, 5)
        obs[12] = float(solver_status)

        return obs

    def _get_reward(self, obs, exceeded_bounds_abv, exceeded_bounds_blw, is_done):
        progress = obs[7]

        h_abv = obs[8]
        h_blw = obs[9]

        mpc_success = bool(obs[12])

        # mpc_failed = bool(obs[12])
        # weights
        w_feas = 10
        w_safety = 6
        w_progress = 10
        w_collision = 50
        w_alpha_exceeded = 10

        safety_abv = self.safety_penalty(h_abv)
        safety_blw = self.safety_penalty(h_blw)

        bounds_penalty = 0
        if exceeded_bounds_abv:
            bounds_penalty -= w_alpha_exceeded
        if exceeded_bounds_blw:
            bounds_penalty -= w_alpha_exceeded

        collision = -w_collision if is_done else 0

        feasibility = 0
        if not mpc_success:
            feasibility = -w_feas

        return float(
            w_safety * safety_abv + 
            w_safety * safety_blw + 
            # w_progress * (1 - progress) +
            w_progress * progress +
            bounds_penalty +
            collision + 
            feasibility
        )

    def safety_penalty(self, h, min_val=-10.0, max_val=1.0):
        penalty = -np.exp(-10 * (h-0.5)) + 1
        return np.clip(penalty, min_val, max_val)

    # def _get_reward(self, obs, exceeded_bounds, is_done):
    #     len_start = self.mpc.get_s_from_pose(self.robot_state[:2])
    #
    #     tangent = self.curve.vel(len_start)
    #     tangent /= np.linalg.norm(tangent)
    #     normal = np.array([-tangent[1], tangent[0]])
    #
    #     # figure out which side of trajectory we are on
    #     traj_p = np.array([self.curve.trajx(len_start), self.curve.trajy(len_start)])
    #     to_robot = self.robot_state[:2] - traj_p
    #     side = np.sign(np.dot(to_robot, normal))
    #
    #     dist = self._dist_from_traj(self.robot_state[:2])
    #
    #     is_colliding = False
    #     if side < 0:
    #         is_colliding = dist > np.polyval(self.lower_coeffs[::-1], len_start)
    #     else:
    #         is_colliding = dist > np.polyval(self.upper_coeffs[::-1], len_start)
    #
    #     reward = 0.0
    #     if not is_colliding:
    #         reward = 5 * obs[4] * obs[5]
    #
    #     reward -= 5 * (1 - obs[7])
    #     mid_alpha = (self.params["MIN_ALPHA"] + self.params["MAX_ALPHA"]) / 2.0
    #     reward -= 5 * (obs[10] - mid_alpha) ** 2
    #     reward -= 5 * (obs[11] - mid_alpha) ** 2
    #     reward -= 30 * int(exceeded_bounds)
    #
    #     if obs[8] > 0.0:
    #         reward += 7 * obs[8]
    #     if obs[9] > 0.0:
    #         reward += 7 * obs[9]
    #
    #     if bool(obs[12]):
    #         reward -= 25
    #
    #     if is_done:
    #         reward -= 25
    #
    #     return reward

    def _obs_init(self, n_obs, min_dist):
        # obs = [[6.12, 2.3], [2.75, 4.45]]
        obs = []
        # obs = [[7.54, 9.32]]
        needed = n_obs

        # get trajectory points
        traj = self.curve.fill(np.linspace(0,1,100))

        np.random.seed(43)
        while len(obs) < n_obs:
            x_min, x_max = float(np.min(self.curve.xs)), float(np.max(self.curve.xs))
            y_min, y_max = float(np.min(self.curve.ys)), float(np.max(self.curve.ys))

            # oversample to reduce resampling loops
            cand_x = np.random.rand(5 * needed) * (x_max - x_min) + x_min
            cand_y = np.random.rand(5 * needed) * (y_max - y_min) + y_min
            cand = np.vstack([cand_x, cand_y]).T

            # compute distances to trajectory (brute force)
            # None index adds a new axis
            dists = np.min(
                np.linalg.norm(cand[:, None, :] - traj[None, :, :], axis=2), axis=1
            )
            valid = cand[dists > min_dist]

            obs.extend(valid.tolist())
            needed = n_obs - len(obs)

        self.obstacles = np.array(obs[:n_obs])

    def _dist_from_traj(self, point):
        dists = np.linalg.norm(self.curve.pts - point[None, :], axis=1)
        return np.min(dists)


if __name__ == "__main__":
    env = RobotEnv(n_obs=150)

    i = 0
    done = False
    env.reset_task(1)
    while not done:
        _, reward, done, _ = env.step([0, 0])
        env.render()
        i += 1

    i = 0
    done = False
    env.reset_task(2)
    while not done and i < 200:
        _, _, done, _ = env.step([0, 0])
        env.render()
        i += 1

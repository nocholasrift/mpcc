import numpy as np
import matplotlib.pyplot as plt

from RobotMPC import Dynamics
from matplotlib.patches import Circle, Polygon


class Plotter:
    def __init__(
        self,
        curve,
        upper_coeffs,
        lower_coeffs,
        obstacles,
        dynamics,
        robot_size=0.2,
        theta=np.pi / 3,
    ):
        self.dynamics = dynamics
        self.robot_size = robot_size
        self.robot_footprint = [
            (robot_size, 0),
            (
                -robot_size * np.sin(theta),
                -robot_size * np.cos(theta),
            ),
            (
                -robot_size * np.sin(theta),
                robot_size * np.cos(theta),
            ),
        ]

        self.prev_states = []

        self.plot_init(curve, upper_coeffs, lower_coeffs, obstacles)

    def plot_init(self, curve, upper_coeffs, lower_coeffs, obstacles):

        # visualization setup
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect("equal")

        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch = Circle(
                (0, 0), self.robot_size, color="blue", label="robot"
            )
        else:
            self.robot_patch = Polygon(
                self.robot_footprint,
                closed=True,
                color="blue",
                label="robot",
            )

        margin = 2.0
        x_min, x_max = float(np.min(curve.xs)), float(np.max(curve.xs))
        y_min, y_max = float(np.min(curve.ys)), float(np.max(curve.ys))
        # x_min, x_max = -1, 1
        # y_min, y_max = 1, 1

        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)

        (self.traj_line,) = self.ax.plot(curve.xs, curve.ys, "k--", label="trajectory")
        (self.ref_point,) = self.ax.plot([], [], "ro", label="reference")

        (self.path_line,) = self.ax.plot([], [], "b-", linewidth=1.5)

        # tubes (initialized as empty lines)
        (self.upper_tube_line,) = self.ax.plot(
            [], [], "r-", label="upper tube", linewidth=2.5
        )
        (self.lower_tube_line,) = self.ax.plot(
            [], [], "b-", label="lower tube", linewidth=2.5
        )

        ss = np.linspace(0, curve.get_arclen(), 100)
        traj = np.vstack([curve.trajx(ss), curve.trajy(ss)]).T

        tangents = np.column_stack([curve.trajx_d(ss), curve.trajy_d(ss)])
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        # compute normals
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        upper_d = np.polyval(upper_coeffs[::-1], ss)
        lower_d = np.polyval(lower_coeffs[::-1], ss)

        # offset trajectory
        upper = traj + upper_d[:, None] * normals
        lower = traj - lower_d[:, None] * normals

        self.ax.plot(
            upper[:, 0], upper[:, 1], "r-", label="upper tube", alpha=0.3, linewidth=2.5
        )
        self.ax.plot(
            lower[:, 0], lower[:, 1], "r-", label="lower tube", alpha=0.3, linewidth=2.5
        )

        self.ax.plot(
            obstacles[:, 0],
            obstacles[:, 1],
            "ko",
            label="obstacles_p",
        )

        self.ax.plot(
            obstacles[:, 0],
            obstacles[:, 1],
            "bo",
            label="obstacles_n",
        )

        self.ax.add_patch(self.robot_patch)

    def plot_tubes(self, curve, tube_gen, robot_state, mpc):

        # figure out where to start and stop the tubes
        len_start = mpc.get_s_from_pose(robot_state[:2])
        curve_len = curve.get_arclen()
        if len_start > curve_len - 1e-2:
            return

        ref_len = mpc.get_params()["REF_LENGTH"]
        len_stop = min(len_start + ref_len, curve_len)

        ss = np.linspace(len_start, len_stop, 100)
        traj = np.vstack([curve.trajx(ss), curve.trajy(ss)]).T

        tangents = np.column_stack([curve.trajx_d(ss), curve.trajy_d(ss)])
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        # compute normals
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        upper_coeffs, lower_coeffs = tube_gen.shift_poly_parameter(len_start, ref_len)

        tau = np.linspace(0, len_stop - len_start, 100)
        upper_d = np.polyval(upper_coeffs[::-1], tau)
        lower_d = np.polyval(lower_coeffs[::-1], tau)

        # offset trajectory
        upper = traj + upper_d[:, None] * normals
        lower = traj - lower_d[:, None] * normals

        # plot tubes
        self.upper_tube_line.set_data(upper[:, 0], upper[:, 1])
        self.lower_tube_line.set_data(lower[:, 0], lower[:, 1])

    def render(self, robot_state, current_ref, curve, tube_gen, mpc):
        if self.dynamics == Dynamics.DOUBLE_INTEGRATOR:
            self.robot_patch.center = (robot_state[0], robot_state[1])
        else:
            theta = robot_state[2]
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            footprint = np.dot(self.robot_footprint, R.T)
            # print((footprint + robot_state[:2]).reshape(-1, 2))
            self.robot_patch.set_xy((footprint + robot_state[:2]).reshape(-1, 2))

        self.ref_point.set_data(current_ref[0], current_ref[1])

        if len(self.prev_states) > 1:
            path = np.array(self.prev_states)
            # if len(self.prev_states) == 100:
            #     print(path)
            #     exit(0)
            self.path_line.set_data(path[:, 0], path[:, 1])

        self.plot_tubes(
            curve,
            tube_gen,
            robot_state,
            mpc,
        )

    def add_state_to_path(self, robot_state):
        # if more than 60 points, remove oldest
        if len(self.prev_states) > 100:
            self.prev_states.pop(0)
        self.prev_states.append(list(robot_state))

    def clear_state_path(self):
        self.prev_states.clear()

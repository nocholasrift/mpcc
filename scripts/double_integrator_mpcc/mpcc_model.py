import numpy as np
from acados_template import AcadosModel
from casadi import (
    MX,
    vertcat,
    horzcat,
    sin,
    cos,
    atan2,
    sqrt,
    exp,
    jacobian,
    interpolant,
    bspline,
    Function,
    DM,
)


class mpcc_ode_model:
    def __init__(self):
        self.model_name = "double_integrator_mpcc"

    def create_model(self, params) -> AcadosModel:
        self.init_x_and_u()

        self.setup_mpcc(params)

        self.setup_x_dot()

        self.clf()

        self.cbf(params)

        # cost expr
        self.Q_c = MX.sym("Q_c")  # 0.1
        self.Q_l = MX.sym("Q_l")  # 100
        self.Q_s = MX.sym("Q_s")  # 0.5
        self.Q_a = 1.0
        self.Q_sdd = 1

        self.cost_expr = (
            self.Q_c * self.e_c**2
            + self.Q_l * self.e_l**2
            + self.Q_a * self.ax**2
            + self.Q_a * self.ay**2
            + self.Q_sdd * self.sddot**2
            - self.Q_s * self.sdot1
        )

        self.cost_expr_e = (
            self.Q_c * self.e_c**2 + self.Q_l * self.e_l**2 - self.Q_s * self.sdot1
        )

        # params
        self.p = vertcat(
            self.x_coeff,
            self.y_coeff,
            self.d_abv_coeff,
            self.d_blw_coeff,
            self.Q_c,
            self.Q_l,
            self.Q_s,
            self.alpha_abv,
            self.alpha_blw,
            self.Ql_c,
            self.Ql_l,
            self.gamma,
        )

        self.compute_spline_x = self.spline_x
        self.compute_spline_y = self.spline_y

        self.compute_xr= Function(
            "xr",
            [self.x, self.x_coeff],
            [self.xr],
        )

        self.compute_yr= Function(
            "yr",
            [self.x, self.y_coeff],
            [self.yr],
        )

        self.compute_obs_dirx = Function(
            "obs_dirx",
            [self.x, self.x_coeff, self.y_coeff],
            [self.obs_dirx],
        )

        self.compute_obs_diry = Function(
            "obs_diry",
            [self.x, self.x_coeff, self.y_coeff],
            [self.obs_diry],
        )

        self.compute_hdot_abv  = Function(
            "hdot_abv",
            [self.x, self.d_abv_coeff, self.x_coeff, self.y_coeff],
            [self.h_dot_abv],
        )

        self.compute_hdot_blw = Function(
            "hdot_blw",
            [self.x, self.d_blw_coeff, self.x_coeff, self.y_coeff],
            [self.h_dot_blw],
        )

        self.compute_cbf_abv = Function(
            "h_abv",
            [self.x, self.d_abv_coeff, self.x_coeff, self.y_coeff],
            [self.h_abv],
        )

        self.compute_lfh_abv = Function(
            "lfh_abv",
            [self.x, self.d_abv_coeff, self.x_coeff, self.y_coeff],
            [self.Lfh_abv],
        )

        Lgh_abv = self.h_dot_abv @ self.g
        self.compute_lgh_abv = Function(
            "lgh_abv",
            [self.x, self.d_abv_coeff, self.x_coeff, self.y_coeff],
            [Lgh_abv],
        )

        self.compute_yrdot= Function(
            "yr_dot",
            [self.x, self.x_coeff, self.y_coeff],
            [self.yr_dot],
        )

        self.compute_xrdot= Function(
            "xr_dot",
            [self.x, self.x_coeff, self.y_coeff],
            [self.xr_dot],
        )

        self.compute_signed_d = Function(
            "signed_d",
            [self.x, self.x_coeff, self.y_coeff],
            [self.signed_d],
        )

        self.compute_p_abv = Function(
            "p_abv",
            [self.x, self.x_coeff, self.y_coeff],
            [self.p_abv],
        )

        self.compute_p_blw = Function(
            "p_blw",
            [self.x, self.x_coeff, self.y_coeff],
            [self.p_blw],
        )

        self.compute_cbf_blw = Function(
            "h_blw",
            [self.x, self.d_blw_coeff, self.x_coeff, self.y_coeff],
            [self.h_blw],
        )

        self.compute_d_abv = Function(
            "d_abv",
            [self.x, self.d_abv_coeff], 
            [self.d_abv],
        )

        self.compute_d_blw = Function(
            "d_blw",
            [self.x, self.d_blw_coeff],
            [self.d_blw],
        )

        self.compute_lfh_blw = Function(
            "lfh_blw",
            [self.x, self.d_blw_coeff, self.x_coeff, self.y_coeff],
            [self.Lfh_blw],
        )

        Lgh_blw = self.h_dot_blw @ self.g
        self.compute_lgh_blw = Function(
            "lgh_blw",
            [self.x, self.d_blw_coeff, self.x_coeff, self.y_coeff],
            [Lgh_blw],
        )

        self.model = AcadosModel()

        self.model.f_impl_expr = self.f_impl
        self.model.f_expl_expr = self.f_expl
        self.model.x = self.x
        self.model.u = self.u
        self.model.p = self.p
        self.model.xdot = self.x_dot
        self.model.name = self.model_name

        self.model.cost_expr_ext_cost = self.cost_expr
        self.model.cost_expr_ext_cost_e = self.cost_expr_e

        self.model.con_h_expr_0 = vertcat(
            self.lyap_con, self.cbf_con_abv, self.cbf_con_blw
        )
        self.model.con_h_expr = vertcat(
            self.lyap_con, self.cbf_con_abv, self.cbf_con_blw
        )

        # store meta information
        self.model.x_labels = [
            "$x$ [m]",
            "$y$ [m]",
            "$v_x$ [m/s]",
            "$v_y$ [m/s]",
            "$s$ []",
            "$sdot$ []",
        ]
        self.model.u_labels = ["$ax$", "$ay$", "$sddot$"]
        self.model.t_label = "$t$ [s]"

        return self.model

    def init_x_and_u(self):

        self.x1 = MX.sym("x1")
        self.y1 = MX.sym("y1")
        self.vx1 = MX.sym("vx1")
        self.vy1 = MX.sym("vy1")
        self.s1 = MX.sym("s1")
        self.sdot1 = MX.sym("s1_dot")

        self.x = vertcat(self.x1, self.y1, self.vx1, self.vy1, self.s1, self.sdot1)

        self.ax = MX.sym("ax")
        self.ay = MX.sym("ay")
        self.sddot = MX.sym("sddot")

        self.u = vertcat(self.ax, self.ay, self.sddot)

    def setup_mpcc(self, params):

        self.v = MX.sym("v")
        self.x_coeff = MX.sym("x_coeffs", 11)
        self.y_coeff = MX.sym("y_coeffs", 11)
        # arc_len_knots = DM([1.0] * 11)
        # arc_len_knots = MX.sym("knots", 11)

        self.arc_len_knots = np.linspace(0, params["ref_length_size"], 11)
        # arc_len_knots = np.linspace(0, 17.0385372, 11)
        self.arc_len_knots = np.concatenate(
            (
                np.ones((4,)) * self.arc_len_knots[0],
                self.arc_len_knots[2:-2],
                np.ones((4,)) * self.arc_len_knots[-1],
            )
        )

        # 1 denotes the multiplicity of the knots at the ends
        # don't need clamped so leave as 1
        self.x_spline_mx = bspline(
            self.v, self.x_coeff, [list(self.arc_len_knots)], [3], 1, {}
        )
        self.y_spline_mx = bspline(
            self.v, self.y_coeff, [list(self.arc_len_knots)], [3], 1, {}
        )

        self.spline_x = Function("xr", [self.v, self.x_coeff], [self.x_spline_mx], {})
        self.spline_y = Function("yr", [self.v, self.y_coeff], [self.y_spline_mx], {})

        self.xr = self.spline_x(self.s1, self.x_coeff)
        self.yr = self.spline_y(self.s1, self.y_coeff)

        self.xr_dot = jacobian(self.xr, self.s1)
        self.yr_dot = jacobian(self.yr, self.s1)

        # self.phi_r = atan2(self.xr_dot, self.yr_dot)
        self.phi_r = atan2(self.yr_dot, self.xr_dot)

        self.e_c = sin(self.phi_r) * (self.x1 - self.xr) - cos(self.phi_r) * (
            self.y1 - self.yr
        )
        self.e_l = -cos(self.phi_r) * (self.x1 - self.xr) - sin(self.phi_r) * (
            self.y1 - self.yr
        )

    def setup_x_dot(self):
        # xdot
        self.x1_dot = MX.sym("x1_dot")
        self.y1_dot = MX.sym("y1_dot")
        self.vx1_dot = MX.sym("vx1_dot")
        self.vy1_dot = MX.sym("vy1_dot")
        self.s1_dot = MX.sym("s1_dot")
        self.sdot1_dot = MX.sym("sdot1_dot")

        self.x_dot = vertcat(
            self.x1_dot,
            self.y1_dot,
            self.vx1_dot,
            self.vy1_dot,
            self.s1_dot,
            self.sdot1_dot,
        )

        # dynamics
        self.f_expl = vertcat(
            self.vx1,
            self.vy1,
            self.ax,
            self.ay,
            self.sdot1,
            self.sddot,
        )

        self.f_impl = self.x_dot - self.f_expl

    def clf(self):
        self.Ql_c = MX.sym("Ql_c")
        self.Ql_l = MX.sym("Ql_l")
        self.gamma = MX.sym("gamma")

        self.f = vertcat(self.vx1, self.vy1, 0, 0, self.sdot1, 0)
        self.g = vertcat(
            horzcat(0, 0, 0),
            horzcat(0, 0, 0),
            horzcat(1, 0, 0),
            horzcat(0, 1, 0),
            horzcat(0, 0, 0),
            horzcat(0, 0, 1),
        )

        self.v = self.Ql_c * self.e_c**2 + self.Ql_l * self.e_l**2
        self.v_dot = (
            jacobian(self.v, self.x) @ self.f
            + jacobian(self.v, self.x) @ self.g @ self.u
        )

        self.lyap_con = self.v_dot + self.gamma * self.v

    def cbf(self, params):
        self.d_abv_coeff = MX.sym("d_above_coeffs", params["tube_poly_degree"] + 1)
        self.d_blw_coeff = MX.sym("d_below_coeffs", params["tube_poly_degree"] + 1)

        self.d_abv = 0
        self.d_blw = 0
        for i in range(params["tube_poly_degree"] + 1):
            self.d_abv = self.d_abv + (self.d_abv_coeff[i] * self.s1**i)
            self.d_blw = self.d_blw + (self.d_blw_coeff[i] * self.s1**i)

        self.alpha_abv = MX.sym("alpha_abv")
        self.alpha_blw = MX.sym("alpha_blw")

        # self.obs_dirx = -self.yr_dot / sqrt(self.xr_dot**2 + self.yr_dot**2)
        # self.obs_diry = self.xr_dot / sqrt(self.xr_dot**2 + self.yr_dot**2)

        self.obs_dirx = -sin(self.phi_r)
        self.obs_diry = cos(self.phi_r)

        self.signed_d = (self.x1 - self.xr) * self.obs_dirx + (
            self.y1 - self.yr
        ) * self.obs_diry

        # theta = atan2(self.vx1, self.vy1)
        theta = atan2(self.vy1, self.vx1)
        vel = sqrt(self.vx1**2 + self.vy1**2)

        # self.p_abv = (
        #     self.obs_dirx * self.vx1 + self.obs_diry * self.vy1
        # ) / vel + vel * 0.05
        self.p_abv = (
            self.obs_dirx * cos(theta) + self.obs_diry * sin(theta)
        ) + vel * 0.05
        self.h_abv = (self.d_abv - self.signed_d) * exp(-self.p_abv)

        # self.p_blw = (
        #     -self.obs_dirx * self.vx1 - self.obs_diry * self.vy1
        # ) / vel + vel * 0.05
        self.p_blw = (
            -self.obs_dirx * cos(theta) - self.obs_diry * sin(theta)
        ) + vel * 0.05
        self.h_blw = (self.signed_d - self.d_blw) * exp(-self.p_blw)

        self.h_dot_abv = jacobian(self.h_abv, self.x)
        self.Lfh_abv = self.h_dot_abv @ self.f

        self.h_dot_blw = jacobian(self.h_blw, self.x)
        self.Lfh_blw = self.h_dot_blw @ self.f

        self.cbf_con_abv = (
            self.Lfh_abv
            + self.h_dot_abv @ self.g @ self.u
            + self.alpha_abv * self.h_abv
        )
        self.cbf_con_blw = (
            self.Lfh_blw
            + self.h_dot_blw @ self.g @ self.u
            + self.alpha_blw * self.h_blw
        )

if __name__ == "__main__":
    params = {"tube_poly_degree": 6}
    model = mpcc_ode_model()
    model.create_model(params)

    f = model.compute_cbf_abv

    print("n_in =", f.n_in())
    for i in range(f.n_in()):
        print(i,
              f.name_in(i),
              f.size_in(i),
              f.sparsity_in(i))

    print("n_out =", f.n_out())
    for i in range(f.n_out()):
        print(i,
              f.name_out(i),
              f.size_out(i))

    x = [-2.25e+00, -2.50e+00,  0.05e+00,  1.00e-6,  0.00e+00,  0.00e+00]
    xs = [-2.24999693, -2.11674265, -1.97900936, -1.84491102, -1.71762874, -1.58263046,
 -1.4541721,  -1.32337543, -1.24702939, -1.24702939, -1.24702939]
    ys = [-2.49999123, -2.11986464, -1.72696107, -1.34442668, -0.98133606, -0.59623445,
 -0.22978891,  0.14332701,  0.36111483,  0.36111483,  0.36111483]
    coeffs = [ 0.36863125, -0.14984056,  0.44477208, -0.53646524,  0.30815822, -0.0849333,
  0.00906838]

    # if np.linalg.norm(x[2:4]) < 1e-9:
    #     x[3] = 1e-3

    print("h", f(x, coeffs, xs, ys))
    print("lfh", model.compute_lfh_abv(x, coeffs, xs, ys))
    print("lgh", model.compute_lgh_abv(x, coeffs, xs, ys))
    print("p_abv", model.compute_p_abv(x, coeffs, xs, ys))
    print("h_dot", model.compute_hdot_abv(x, coeffs, xs, ys))
    

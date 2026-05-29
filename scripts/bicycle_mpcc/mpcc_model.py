import os
import numpy as np

from acados_template import AcadosModel
from casadi import (
    MX,
    vertcat,
    horzcat,
    sin,
    cos,
    tan,
    atan,
    atan2,
    sqrt,
    exp,
    jacobian,
    interpolant,
    Function,
    CodeGenerator,
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ParamVector import ParamVector


class DebugRegistry:
    def __init__(self):
        self.exprs = {}

    def add(self, name, expr):
        self.exprs[name] = expr

    def build_functions(self, inputs):
        funcs = {}
        for name, expr in self.exprs.items():
            funcs[name] = Function(name, inputs, [expr])
        return funcs

    def generate_c(self, filename, inputs):
        opts = {"cpp": True, "with_header": True}
        C = CodeGenerator(filename, opts)
        for name, expr in self.exprs.items():
            f = Function(name, inputs, [expr])
            C.add(f)
        C.generate()


class mpcc_ode_model:
    def __init__(self):
        self.model_name = "bicycle_mpcc"

    def create_model(self, params, output_dir="") -> AcadosModel:
        self.init_x_and_u()

        self.setup_mpcc(params)

        self.setup_x_dot()

        self.clf()

        self.cbf(params)

        # cost expr
        self.Q_c = MX.sym("Q_c")  
        self.Q_l = MX.sym("Q_l")  
        self.Q_s = MX.sym("Q_s")  
        self.Q_a = 1.0
        self.Q_delta = 1.0
        self.Q_sdd = 1

        self.cost_expr = (
            self.Q_c * self.e_c**2
            + self.Q_l * self.e_l**2
            + self.Q_a * self.a**2
            # + self.Q_delta * self.delta**2
            + self.Q_delta * self.kappa**2
            + self.Q_sdd * self.sddot**2
            - self.Q_s * self.sdot1
        )

        self.cost_expr_e = (
            self.Q_c * self.e_c**2 + self.Q_l * self.e_l**2 - self.Q_s * self.sdot1
        )

        self.pv = ParamVector()
        self.pv.add(str(self.x_coeff), self.x_coeff)
        self.pv.add(str(self.y_coeff), self.y_coeff)
        self.pv.add(str(self.d_abv_coeff), self.d_abv_coeff)
        self.pv.add(str(self.d_blw_coeff), self.d_blw_coeff)
        self.pv.add(str(self.Q_c), self.Q_c)
        self.pv.add(str(self.Q_l), self.Q_l)
        self.pv.add(str(self.Q_s), self.Q_s)
        self.pv.add(str(self.alpha_abv), self.alpha_abv)
        self.pv.add(str(self.alpha_blw), self.alpha_blw)
        self.pv.add(str(self.Ql_c), self.Ql_c)
        self.pv.add(str(self.Ql_l), self.Ql_l)
        self.pv.add(str(self.gamma), self.gamma)
        self.pv.add(str(self.L_path), self.L_path)
        self.pv.add(str(self.body_length), self.body_length)

        self.p = self.pv.as_casadi_vector()

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
            "$\\theta$ [rad]",
            "$v$ [m/s]",
            "$s$ []",
            "$sdot$ []",
        ]
        self.model.u_labels = ["$a$", "$\\delta$", "$sddot$"]
        self.model.t_label = "$t$ [s]"

        self.add_debugs_and_params(output_dir)

        return self.model

    def init_x_and_u(self):
        # States: x, y, heading theta, velocity v, path parameter s, path speed sdot
        self.x1 = MX.sym("x1")
        self.y1 = MX.sym("y1")
        self.theta = MX.sym("theta")
        self.v = MX.sym("v")
        self.s1 = MX.sym("s1")
        self.sdot1 = MX.sym("s1_dot")

        self.x = vertcat(self.x1, self.y1, self.theta, self.v, self.s1, self.sdot1)

        # Controls: acceleration a, steering angle delta, path acceleration sddot
        self.a = MX.sym("a")
        # self.delta = MX.sym("delta")
        self.kappa = MX.sym("kappa")
        self.sddot = MX.sym("sddot")

        # self.u = vertcat(self.a, self.delta, self.sddot)
        self.u = vertcat(self.a, self.kappa, self.sddot)

    def setup_mpcc(self, params):
        n_knots = params["mpc_ref_samples"]
        xspl = MX.sym("xspl", 1, 1)
        yspl = MX.sym("yspl", 1, 1)
        self.x_coeff = MX.sym("x_coeffs", n_knots)
        self.y_coeff = MX.sym("y_coeffs", n_knots)

        self.arc_len_knots = np.linspace(0, 1, params["mpc_ref_samples"])
        self.L_path = MX.sym("L_path")

        self.interp_x = interpolant("interp_x", "bspline", [self.arc_len_knots.tolist()])
        self.interp_exp_x = self.interp_x(xspl, self.x_coeff)
        self.xr_func = Function("xr", [xspl, self.x_coeff], [self.interp_exp_x])

        self.interp_y = interpolant("interp_y", "bspline", [self.arc_len_knots.tolist()])
        self.interp_exp_y = self.interp_y(yspl, self.y_coeff)
        self.yr_func = Function("yr", [yspl, self.y_coeff], [self.interp_exp_y])

        s_norm = self.s1 / self.L_path
        self.xr = self.xr_func(s_norm, self.x_coeff)
        self.yr = self.yr_func(s_norm, self.y_coeff)

        self.xr_dot = jacobian(self.xr, self.s1)
        self.yr_dot = jacobian(self.yr, self.s1)

        self.phi_r = atan2(self.yr_dot, self.xr_dot)

        self.e_c = sin(self.phi_r) * (self.x1 - self.xr) - cos(self.phi_r) * (self.y1 - self.yr)
        self.e_l = -cos(self.phi_r) * (self.x1 - self.xr) - sin(self.phi_r) * (self.y1 - self.yr)

    def setup_x_dot(self):
        self.x1_dot = MX.sym("x1_dot")
        self.y1_dot = MX.sym("y1_dot")
        self.theta_dot = MX.sym("theta_dot")
        self.v_dot_sym = MX.sym("v_dot_sym")
        self.s1_dot = MX.sym("s1_dot")
        self.sdot1_dot = MX.sym("sdot1_dot")

        self.x_dot = vertcat(
            self.x1_dot,
            self.y1_dot,
            self.theta_dot,
            self.v_dot_sym,
            self.s1_dot,
            self.sdot1_dot,
        )

        self.body_length = MX.sym("body_length")

        # Kinematic Bicycle Constants
        lf = 0.16  # Front wheelbase length
        lr = 0.15  # Rear wheelbase length
        # beta = atan((lr / (lf + lr)) * self.delta)

        # Explict dynamics formulation
        self.f_expl = vertcat(
            self.v * cos(self.theta), # + beta),
            self.v * sin(self.theta), # + beta),
            (self.v * self.kappa / self.body_length),
            self.a,
            self.sdot1,
            self.sddot,
        )

        self.f_impl = self.x_dot - self.f_expl

    def clf(self):
        self.Ql_c = MX.sym("Ql_c")
        self.Ql_l = MX.sym("Ql_l")
        self.gamma = MX.sym("gamma")

        # Drift dynamic vector fields f and control matrix g
        lf = 0.16
        lr = 0.15
        # beta = atan((lr / (lf + lr)) * self.delta)

        self.f = vertcat(
            self.v * cos(self.theta), # + beta),
            self.v * sin(self.theta), # + beta),
            # (self.v / lr) * sin(beta),
            0, 
            0,
            self.sdot1,
            0
        )
        self.g = vertcat(
            horzcat(0, 0, 0),
            horzcat(0, 0, 0),
            horzcat(0, self.v / self.body_length, 0),
            horzcat(1, 0, 0),
            horzcat(0, 0, 0),
            horzcat(0, 0, 1),
        )

        self.v_lyap = self.Ql_c * self.e_c**2 + self.Ql_l * self.e_l**2

        self.lfv = jacobian(self.v_lyap, self.x) @ self.f
        self.lgv = jacobian(self.v_lyap, self.x) @ self.g
        self.lgvu = self.lgv @ self.u
        self.v_dot = self.lfv + self.lgvu
        self.lyap_con = self.v_dot + self.gamma * self.v_lyap

    def cbf(self, params):
        self.d_abv_coeff = MX.sym("d_above_coeffs", params["tube_poly_degree"] + 1)
        self.d_blw_coeff = MX.sym("d_below_coeffs", params["tube_poly_degree"] + 1)

        self.d_abv = 0
        self.d_blw = 0
        s_norm = self.s1 / self.L_path
        for i in range(params["tube_poly_degree"] + 1):
            self.d_abv = self.d_abv + (self.d_abv_coeff[i] * s_norm**i)
            self.d_blw = self.d_blw + (self.d_blw_coeff[i] * s_norm**i)

        self.alpha_abv = MX.sym("alpha_abv")
        self.alpha_blw = MX.sym("alpha_blw")

        self.obs_dirx = -sin(self.phi_r)
        self.obs_diry = cos(self.phi_r)

        self.signed_d = (self.x1 - self.xr) * self.obs_dirx + (self.y1 - self.yr) * self.obs_diry

        # Adaptation to track vehicle heading orientation
        self.p_abv = (self.obs_dirx * cos(self.theta) + self.obs_diry * sin(self.theta)) + self.v * 0.05
        self.h_abv = (self.d_abv - self.signed_d) * exp(-self.p_abv)

        self.p_blw = (-self.obs_dirx * cos(self.theta) - self.obs_diry * sin(self.theta)) + self.v * 0.05
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
        # self.cbf_con_abv = 1
        # self.cbf_con_blw = 1

    def add_debugs_and_params(self, output_dir):
        self.debug = DebugRegistry()

        self.debug.add("bicycle_xr", self.xr)
        self.debug.add("bicycle_yr", self.yr)
        self.debug.add("bicycle_xr_dot", self.xr_dot)
        self.debug.add("bicycle_yr_dot", self.yr_dot)
        self.debug.add("bicycle_phi_r", self.phi_r)
        self.debug.add("bicycle_e_c", self.e_c)
        self.debug.add("bicycle_e_l", self.e_l)
        self.debug.add("bicycle_signed_d", self.signed_d)
        self.debug.add("bicycle_p_abv", self.p_abv)
        self.debug.add("bicycle_p_blw", self.p_blw)
        self.debug.add("bicycle_d_abv", self.d_abv)
        self.debug.add("bicycle_d_blw", self.d_blw)
        self.debug.add("bicycle_h_abv", self.h_abv)
        self.debug.add("bicycle_h_blw", self.h_blw)
        self.debug.add("bicycle_Lfh_abv", self.Lfh_abv)
        self.debug.add("bicycle_Lfh_blw", self.Lfh_blw)
        self.debug.add("bicycle_Lghu_abv", self.h_dot_abv @ self.g @ self.u)
        self.debug.add("bicycle_Lghu_blw", self.h_dot_blw @ self.g @ self.u)
        self.debug.add("bicycle_Lfv", self.lfv)
        self.debug.add("bicycle_Lgv", self.lgv)
        self.debug.add("bicycle_Lgvu", self.lgvu)
        self.debug.add("bicycle_lyap_const", self.lyap_con)

        debug_inputs = [
            self.x,
            self.u,
            self.x_coeff,
            self.y_coeff,
            self.d_abv_coeff,
            self.d_blw_coeff,
            self.Ql_c,
            self.Ql_l,
            self.gamma,
            self.L_path,
            self.body_length,
        ]

        if output_dir == "":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "cpp_generated_code")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        current_dir = os.getcwd()
        os.chdir(output_dir)

        fname = "casadi_bicycle_mpcc_internals"
        self.debug.generate_c(f"{fname}.cpp", debug_inputs)
        os.system(f"gcc -fPIC -shared {fname}.cpp -o lib{fname}.so")

        self.pv.write_cpp_header(
            "bicycle_mpcc_param_indices.h",
            namespace="mpcc::bicycle_param",
        )

        os.chdir(current_dir)


if __name__ == "__main__":
    params = {"tube_poly_degree": 6, "mpc_ref_samples": 11}
    model = mpcc_ode_model()
    model.create_model(params)

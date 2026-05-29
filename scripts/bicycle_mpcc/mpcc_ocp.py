#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import numpy as np
from mpcc_model import mpcc_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver


def create_ocp(yaml_file, casadi_dir):
    ocp = AcadosOcp()

    params = None
    if yaml_file != "":
        with open(yaml_file) as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("ERROR:", e, file=sys.stderr)
                exit(1)
    else:
        print("ERROR: YAML file must be provided!", file=sys.stderr)
        exit(1)

    # set model
    mpcc_model = mpcc_ode_model()
    model = mpcc_model.create_model(params, casadi_dir)
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    nparams = model.p.rows()
    N = 10

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.dims.N = N
    ocp.parameter_values = np.zeros((nparams,))

    ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    # con_upper_bounds = np.array([0, 1e6, 1e6])
    # con_lower_bounds = np.array([-1e6, 0, 0])

    con_upper_bounds = np.array([1e6, 1e6, 1e6])
    con_lower_bounds = np.array([-1e6, 0, 0])

    ocp.constraints.uh_0 = con_upper_bounds
    ocp.constraints.lh_0 = con_lower_bounds
    ocp.constraints.uh = con_upper_bounds
    ocp.constraints.lh = con_lower_bounds

    # set soft constraints
    nsh = 3
    ocp.constraints.lsh_0 = np.zeros((nsh,))
    ocp.constraints.ush_0 = np.zeros((nsh,))
    ocp.constraints.idxsh_0 = np.array([0, 1, 2])

    ocp.constraints.lsh = np.zeros((nsh,))
    ocp.constraints.ush = np.zeros((nsh,))
    ocp.constraints.idxsh = np.array([0, 1, 2])

    grad_cost = 100
    hess_cost = 1

    ocp.cost.Zl_0 = hess_cost * np.ones((nsh,))
    ocp.cost.Zu_0 = hess_cost * np.ones((nsh,))
    ocp.cost.zl_0 = grad_cost * np.ones((nsh,))
    ocp.cost.zu_0 = grad_cost * np.ones((nsh,))

    ocp.cost.Zl = hess_cost * np.ones((nsh,))
    ocp.cost.Zu = hess_cost * np.ones((nsh,))
    ocp.cost.zl = grad_cost * np.ones((nsh,))
    ocp.cost.zu = grad_cost * np.ones((nsh,))

    grad_cost = 1e4
    hess_cost = 1e2

    num_cbf = nsh - 1
    ocp.cost.Zl_0[-num_cbf:] = hess_cost * np.ones((num_cbf,))
    ocp.cost.Zu_0[-num_cbf:] = hess_cost * np.ones((num_cbf,))
    ocp.cost.zl_0[-num_cbf:] = grad_cost * np.ones((num_cbf,))
    ocp.cost.zu_0[-num_cbf:] = grad_cost * np.ones((num_cbf,))

    ocp.cost.Zl[-num_cbf:] = hess_cost * np.ones((num_cbf,))
    ocp.cost.Zu[-num_cbf:] = hess_cost * np.ones((num_cbf,))
    ocp.cost.zl[-num_cbf:] = grad_cost * np.ones((num_cbf,))
    ocp.cost.zu[-num_cbf:] = grad_cost * np.ones((num_cbf,))

    # Control Constraints: [Acceleration, Steering Angle, Path Acceleration]
    max_acc = 3.0
    max_steer = np.tan(0.52)   # ~30 degrees max steer angle bounds
    ocp.constraints.lbu = np.array([-max_acc, -max_steer, -max_acc])
    ocp.constraints.ubu = np.array([max_acc, max_steer, max_acc])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    # State Constraints: [x, y, theta, v, s, sdot]
    max_vel = 1.5
    ocp.constraints.lbx = np.array([-1e6, -1e6, -2*np.pi, 0.0, 0, 0])
    ocp.constraints.ubx = np.array([1e6, 1e6, 2*np.pi, max_vel, 1e6, max_vel])
    ocp.constraints.idxbx = np.array(range(nx))

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N
    ocp.solver_options.shooting_nodes = np.linspace(0, Tf, N + 1)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.regularize_method = "MIRROR"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.globalization_line_search_use_sufficient_descent = True
    ocp.solver_options.hpipm_mode = "ROBUST"

    return ocp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bicycle model mpcc")
    parser.add_argument("--yaml", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--casadi_dir", type=str, default="")

    args = parser.parse_args()

    ocp = create_ocp(args.yaml, args.casadi_dir)
    if args.output_dir != "":
        ocp.code_export_directory = args.output_dir

    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

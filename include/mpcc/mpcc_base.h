#pragma once

#include <mpcc/types.h>
#include <mpcc/utils.h>
#include <mpcc/termcolor.hpp>

#include <Eigen/Core>
//
// acados
#include "acados_c/ocp_nlp_interface.h"

#include <cmath>
#include <map>

namespace mpcc {
using TrajectoryView = types::Trajectory::View;

// Interface assumes the use of acados
// Using CRTP so MPCBase gets access to derived class members.
// Downside is it is unclear what each MPCImpl needs to implement.
// using C++17 and below so no concepts unfortunately to supplement that...
template <typename MPCImpl>
class MPCBase {

 public:
  virtual ~MPCBase() = default;
  // virtual void load_params(const std::map<std::string, double>& params) = 0;
  void load_params(const std::map<std::string, double>& params) {
    static_cast<MPCImpl*>(this)->load_params();
  }

  void set_odom(const Eigen::VectorXd& odom) {
    _odom = odom;

    if (!_odom_init) {
      _odom_init = true;
      reset_horizon();
    }
  }
  virtual void reset_horizon() = 0;

  bool set_solver_parameters(const TrajectoryView& reference,
                             const unsigned int num_params) {
    // double params[num_params];
    std::vector<double> params;
    params.resize(num_params);
    auto& ctrls_x = reference.xs;
    auto& ctrls_y = reference.ys;

    int provided_params = ctrls_x.size() + ctrls_y.size() + _tubes[0].size() +
                          _tubes[1].size() + 8;
    if (provided_params != num_params) {
      std::cerr << termcolor::yellow << "[MPCC] provided param count"
                << provided_params << " does not match acados parameter size "
                << num_params << termcolor::reset << std::endl;

      return false;
    }

    params[num_params - 8] = _w_qc;
    params[num_params - 7] = _w_ql;
    params[num_params - 6] = _w_q_speed;
    params[num_params - 5] = _alpha_abv;
    params[num_params - 4] = _alpha_blw;
    params[num_params - 3] = _w_qc_lyap;
    params[num_params - 2] = _w_ql_lyap;
    params[num_params - 1] = _gamma;

    for (int i = 0; i < ctrls_x.size(); ++i) {
      params[i]                  = ctrls_x[i];
      params[i + ctrls_x.size()] = ctrls_y[i];
    }

    for (int i = 0; i < _tubes[0].size(); ++i) {
      params[i + 2 * ctrls_x.size()]                    = _tubes[0](i);
      params[i + 2 * ctrls_x.size() + _tubes[0].size()] = _tubes[1](i);
    }

    MPCImpl& impl = static_cast<MPCImpl&>(*this);
    for (int i = 0; i < _mpc_steps + 1; ++i) {
      impl.acados_capsule_update_params(params, i);
    }

    return true;
  }

  void set_tubes(const std::array<Eigen::VectorXd, 2>& tubes) {
    _tubes = tubes;
  }

  const std::array<Eigen::VectorXd, 2>& get_tubes() const { return _tubes; }

  virtual const Eigen::VectorXd& get_state() const                      = 0;
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits() const = 0;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits() const = 0;

  decltype(auto) get_horizon() const {
    return static_cast<MPCImpl*>(this)->get_horizon();
  }

  virtual Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control,
                                       bool is_abv) const = 0;

  const bool get_solver_status() const { return _solve_success; }

  double limit(double prev_val, double input, double max_rate,
               double dt) const {
    double ret = input;
    if (fabs(prev_val - input) / dt > max_rate) {
      if (input > prev_val)
        ret = prev_val + max_rate * dt;
      else
        ret = prev_val - max_rate * dt;
    }

    return ret;
  }

  std::array<double, 2> solve(const Eigen::VectorXd& state,
                              const types::Trajectory::View& reference,
                              bool is_reverse) {
    MPCImpl& impl  = static_cast<MPCImpl&>(*this);
    _solve_success = false;

    /*************************************
  ********** INITIAL CONDITION *********
  **************************************/
    if (!is_solver_ready(state)) {
      return {0, 0};
    }

    Eigen::VectorXd x0 = impl.prepare_initial_state(state);
    set_acados_initial_constraints(x0);

    /*************************************
  ********* INITIALIZE SOLUTION ********
  **************************************/
    std::cout << "init state is: " << x0.transpose() << "\n";
    warm_start_mpc(x0);

    /*************************************
  ********* SET REFERENCE PARAMS *******
  **************************************/
    if (!set_solver_parameters(reference, MPCImpl::kNP)) {
      return {0, 0};
    }

    /*************************************
  ************* RUN SOLVER *************
  **************************************/
    _solve_success = impl.run_acados_solver(x0);

    /*************************************
  *********** PROCESS OUTPUT ***********
  **************************************/

    Eigen::VectorXd xtraj;
    Eigen::VectorXd utraj;
    xtraj.resize((_mpc_steps + 1) * MPCImpl::kNX);
    utraj.resize(_mpc_steps * MPCImpl::kNU);
    process_solver_output(xtraj, utraj);

    _has_run = true;

    if (!_solve_success) {
      std::cout << "[MPCC] SOLVER STATUS WAS INFEASIBLE!\n";
    }

    std::array<double, 2> cmd = impl.compute_mpc_vel_command(xtraj, utraj);

    _prev_x0 = xtraj;
    _prev_u0 = utraj;

    return cmd;
  }

 protected:
  bool is_solver_ready(const Eigen::VectorXd& state) {
    if (!static_cast<MPCImpl*>(this)->is_acados_ready()) {
      std::cerr << termcolor::yellow << "[MPCC] Acados not yet initialized!"
                << termcolor::reset << std::endl;
    }

    if (_tubes[0].size() == 0 || _tubes[1].size() == 0) {
      std::cerr << "[MPCC] tubes are not set yet, mpc cannot run" << std::endl;
      return false;
    }

    if (state.size() != MPCImpl::kNBX0) {
      std::cerr << termcolor::yellow << "[MPCC] state size passed has size "
                << state.size() << " but should be " << MPCImpl::kNBX0
                << termcolor::reset << std::endl;
      return false;
    }

    return true;
  }

  void set_acados_initial_constraints(const Eigen::VectorXd& initial_state) {

    double lbx0[MPCImpl::kNBX0];
    double ubx0[MPCImpl::kNBX0];

    memcpy(lbx0, &initial_state[0], MPCImpl::kNBX0 * sizeof(double));
    memcpy(ubx0, &initial_state[0], MPCImpl::kNBX0 * sizeof(double));

    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "lbx",
                                  lbx0);
    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "ubx",
                                  ubx0);
  }

  void warm_start_mpc(const Eigen::VectorXd& initial_state) {
    // in our case kNX = kNBX0

    // should be true by construction
    assert(_prev_x0.size() == (_mpc_steps + 1) * MPCImpl::kNX);

    Eigen::Vector2d prev_pos = _prev_x0.segment(MPCImpl::kNX, 2);
    Eigen::Vector2d curr_pos = initial_state.head(2);

    if (!_is_shift_warm) {
      std::cout << "using no u warm start\n";
      warm_start_no_u(initial_state);
    } else {
      std::cout << "using shifted warm start\n";
      // warm_start_shifted_u(false, x0);
      warm_start_shifted_u((prev_pos - curr_pos).norm() > 5e-2, initial_state);
    }
  }

  // From linear to nonlinear MPC: bridging the gap via the real-time iteration,
  // Gros et. al.
  void warm_start_shifted_u(bool correct_perturb,
                            const Eigen::VectorXd& state) {
    double starting_s = _prev_x0[1 * MPCImpl::kNX + MPCImpl::kIndS];
    if (correct_perturb) {
      std::cout << termcolor::red << "[MPCC] Guess pos. too far, correcting"
                << termcolor::reset << std::endl;

      Eigen::VectorXd curr = state;

      // project forward previous control inputs, starting from true current
      // state
      for (int i = 0; i < _mpc_steps - 1; ++i) {
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", &curr[0]);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u",
                        &_prev_u0[(i + 1) * MPCImpl::kNU]);
        curr = static_cast<MPCImpl*>(this)->next_state(
            curr, _prev_u0.segment((i + 1) * MPCImpl::kNU, MPCImpl::kNU));
        // std::cout << curr.transpose() << std::endl;
      }

      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x",
                      &curr[0]);
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                      &_prev_u0[(_mpc_steps - 1) * MPCImpl::kNU]);

      curr = static_cast<MPCImpl*>(this)->next_state(
          curr, _prev_u0.tail(MPCImpl::kNU));
      // std::cout << curr.transpose() << std::endl;

      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x",
                      &curr[0]);
      // exit(0);
    } else {
      for (int i = 1; i < _mpc_steps; ++i) {
        Eigen::VectorXd warm_state =
            _prev_x0.segment(i * MPCImpl::kNX, MPCImpl::kNX);
        warm_state(MPCImpl::kIndS) -= starting_s;
        // warm_state(MPCImpl::kIndS) = std::max(warm_state(MPCImpl::kIndS), 1e-6);

        std::cout << i << " " << warm_state.transpose() << "\n";
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "x",
                        &warm_state[0]);
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "u",
                        &_prev_u0[i * MPCImpl::kNU]);
      }

      Eigen::VectorXd xN_prev = _prev_x0.tail(MPCImpl::kNX);
      xN_prev(MPCImpl::kIndS) -= starting_s;
      // xN_prev(MPCImpl::kIndS) = std::max(xN_prev(MPCImpl::kIndS), 1e-6);

      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x",
                      &xN_prev[0]);
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                      &_prev_u0[(_mpc_steps - 1) * MPCImpl::kNU]);

      Eigen::VectorXd uN_prev = _prev_u0.tail(MPCImpl::kNU);
      Eigen::VectorXd xN =
          static_cast<MPCImpl*>(this)->next_state(xN_prev, uN_prev);

      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x",
                      &xN[0]);
    }
  }

  void warm_start_no_u(const Eigen::VectorXd& initial_state) {
    double x_init[MPCImpl::kNX];
    memcpy(x_init, &initial_state[0], MPCImpl::kNX * sizeof(double));

    double u_init[MPCImpl::kNU];
    u_init[0] = 0.0;
    u_init[1] = 0.0;
    u_init[2] = 0.0;

    // x_init[5] = x_init[3];

    for (int i = 0; i < _mpc_steps; ++i) {
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", x_init);
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u", u_init);
    }

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", x_init);
  }

  void process_solver_output(Eigen::VectorXd& xtraj, Eigen::VectorXd& utraj) {
    // stored as x0, y0,..., x1, y1, ..., xN, yN, ...
    for (int i = 0; i < _mpc_steps; ++i) {
      // std::array<double, 3> u;
      ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "x",
                      &xtraj[i * MPCImpl::kNX]);
      ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "u",
                      &utraj[i * MPCImpl::kNU]);
      // ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "u", &u[0]);
      // std::cout << i << " " << u[0] << " " << u[1] << " " << u[2] << "\n";
    }

    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x",
                    &xtraj[_mpc_steps * MPCImpl::kNX]);

    std::cout << "xtraj is:\n";
    for (int i = 0; i < xtraj.size(); i += MPCImpl::kNX) {
      std::cout << i << " " << xtraj.segment(i, MPCImpl::kNX).transpose()
                << "\n";
    }

    std::cout << "utraj is:\n";
    for (int i = 0; i < utraj.size(); i += MPCImpl::kNU) {
      std::cout << i << " " << utraj.segment(i, MPCImpl::kNU).transpose()
                << "\n";
    }

    // _prev_x0 = xtraj;
    // _prev_u0 = utraj;

    static_cast<MPCImpl*>(this)->map_trajectory_to_buffers(xtraj, utraj);
  }

 protected:
  sim_config* _sim_config;
  sim_in* _sim_in;
  sim_out* _sim_out;
  void* _sim_dims;

  ocp_nlp_in* _nlp_in;
  ocp_nlp_out* _nlp_out;
  ocp_nlp_dims* _nlp_dims;
  ocp_nlp_config* _nlp_config;
  ocp_nlp_solver* _nlp_solver;
  void* _nlp_opts;

  double* _new_time_steps;

  Eigen::VectorXd _state;
  Eigen::VectorXd _odom;

  Eigen::VectorXd _prev_x0;
  Eigen::VectorXd _prev_u0;

  std::map<std::string, double> _params;

  std::array<Eigen::VectorXd, 2> _tubes;

  int _mpc_steps;
  int _ref_samples;

  double _dt;
  double _ref_len_sz;

  bool _has_run;
  bool _odom_init;
  bool _is_shift_warm;
  bool _solve_success;

  // params
  double _gamma;
  double _w_ql_lyap;
  double _w_qc_lyap;
  double _w_angvel;
  double _w_angvel_d;
  double _w_linvel_d;

  double _w_ql;
  double _w_qc;
  double _w_q_speed;
  double _alpha_abv;
  double _alpha_blw;
  double _colinear;
  double _padding;

  bool _use_cbf;
};
}  // namespace mpcc

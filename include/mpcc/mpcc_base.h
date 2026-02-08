#pragma once

#include <mpcc/types.h>
#include <mpcc/utils.h>
#include <mpcc/termcolor.hpp>

#include <Eigen/Core>
//
// acados
#include <mpcc/acados_interface.h>

#include <optional>
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

  bool set_solver_parameters(const types::Corridor& corridor,
                             const unsigned int num_params) {
    using Side = types::Corridor::Side;
    std::vector<double> params;
    params.resize(num_params);

    const TrajectoryView& traj_view = corridor.get_trajectory().view();
    const auto& ctrls_x             = traj_view.xs;
    const auto& ctrls_y             = traj_view.ys;

    // this shoudl never happen... but just in case.
    if (corridor.get_above_poly().get_degree() !=
        corridor.get_below_poly().get_degree()) {
      std::cerr << termcolor::yellow
                << "[MPCC] tube degrees do not match for above and below!"
                << num_params << termcolor::reset << std::endl;
      return false;
    }

    int N_tube_coeffs = corridor.get_above_poly().get_coeffs().size();
    int provided_params =
        ctrls_x.size() + ctrls_y.size() + 2 * N_tube_coeffs + 9;
    if (provided_params != num_params) {
      std::cerr << termcolor::yellow << "[MPCC] provided param count"
                << provided_params << " does not match acados parameter size "
                << num_params << termcolor::reset << std::endl;

      return false;
    }

    params[num_params - 9] = _w_qc;
    params[num_params - 8] = _w_ql;
    params[num_params - 7] = _w_q_speed;
    params[num_params - 6] = _alpha_abv;
    params[num_params - 5] = _alpha_blw;
    params[num_params - 4] = _w_qc_lyap;
    params[num_params - 3] = _w_ql_lyap;
    params[num_params - 2] = _gamma;
    params[num_params - 1] = traj_view.arclen;

    int N_ctrls = ctrls_x.size();
    for (int i = 0; i < N_ctrls; ++i) {
      params[i]           = ctrls_x[i];
      params[i + N_ctrls] = ctrls_y[i];
    }

    const auto& above_coeffs = corridor.get_tube_coeffs(Side::kAbove);
    const auto& below_coeffs = corridor.get_tube_coeffs(Side::kBelow);

    for (int i = 0; i < N_tube_coeffs; ++i) {
      params[i + 2 * N_ctrls]                 = above_coeffs(i);
      params[i + 2 * N_ctrls + N_tube_coeffs] = below_coeffs(i);
    }

    for (int step = 0; step < _mpc_steps + 1; ++step) {
      _acados_solver.update_params(step, params);
    }

    return true;
  }

  const Eigen::VectorXd& get_state() const {return _state;}
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits() const = 0;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits() const = 0;

  decltype(auto) get_horizon() const {
    return static_cast<MPCImpl*>(this)->get_horizon();
  }

  std::optional<std::array<double, 2>> presolve_hook(
      const Eigen::VectorXd& state, const types::Corridor& reference) const {
    return std::nullopt;
  }

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
                              types::Corridor& corridor, bool is_reverse) {

    MPCImpl& impl  = static_cast<MPCImpl&>(*this);
    _solve_success = false;

    std::optional<std::array<double, 2>> pre_cmd =
        impl.presolve_hook(state, corridor);

    if (pre_cmd) {
      return *pre_cmd;
    }

    /*************************************
  ********** INITIAL CONDITION *********
  **************************************/
    if (!is_solver_ready(state, corridor)) {
      return {0, 0};
    }

    Eigen::VectorXd x0 = impl.prepare_initial_state(state, corridor);
    _state = x0;

    set_acados_initial_constraints(x0);

    /*************************************
  ********* INITIALIZE SOLUTION ********
  **************************************/
    std::cout << "warm starting with init state: " << x0.transpose() << "\n";
    warm_start_mpc(x0);

    /*************************************
  ********* SET REFERENCE PARAMS *******
  **************************************/
    if (!set_solver_parameters(corridor, MPCImpl::kNP)) {
      return {0, 0};
    }

    /*************************************
  ************* RUN SOLVER *************
  **************************************/
    _solve_success = run_acados_solver(x0);

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
  bool is_solver_ready(const Eigen::VectorXd& state,
                       const types::Corridor& corridor) {
    using Side = types::Corridor::Side;

    if (!_acados_solver.is_initialized()) {
      std::cerr << termcolor::yellow << "[MPCC] Acados not yet initialized!"
                << termcolor::reset << std::endl;
      return false;
    }

    const auto& above_coeffs = corridor.get_tube_coeffs(Side::kAbove);
    const auto& below_coeffs = corridor.get_tube_coeffs(Side::kBelow);
    if (above_coeffs.size() == 0 || below_coeffs.size() == 0) {
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

    unsigned int initial_stage = 0;
    _acados_solver.set_model_constraint(initial_stage, "lbx",
                                        initial_state.data());
    _acados_solver.set_model_constraint(initial_stage, "ubx",
                                        initial_state.data());
  }

  void warm_start_mpc(const Eigen::VectorXd& initial_state) {

    // should be true by construction
    assert(_prev_x0.size() == (_mpc_steps + 1) * MPCImpl::kNX);

    Eigen::Vector2d prev_pos = _prev_x0.segment(MPCImpl::kNX, 2);
    Eigen::Vector2d curr_pos = initial_state.head(2);

    std::cout << termcolor::yellow << "previous state: "
              << _prev_x0.segment(MPCImpl::kNX, MPCImpl::kNX).transpose()
              << termcolor::reset << "\n";

    if (!_is_shift_warm) {
      std::cout << "using no u warm start\n";
      warm_start_no_u(initial_state);
    } else {
      std::cout << "using shifted warm start\n";
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
      for (int step = 0; step < _mpc_steps - 1; ++step) {
        _acados_solver.set_output(step, "x", curr.data());
        _acados_solver.set_output(step, "u",
                                  &_prev_u0[(step + 1) * MPCImpl::kNU]);
        curr = static_cast<MPCImpl*>(this)->next_state(
            curr, _prev_u0.segment((step + 1) * MPCImpl::kNU, MPCImpl::kNU));
      }

      _acados_solver.set_output(_mpc_steps - 1, "x", curr.data());
      _acados_solver.set_output(_mpc_steps - 1, "u",
                                &_prev_u0[(_mpc_steps - 1) * MPCImpl::kNU]);

      curr = static_cast<MPCImpl*>(this)->next_state(
          curr, _prev_u0.tail(MPCImpl::kNU));

      _acados_solver.set_output(_mpc_steps, "x", curr.data());
    } else {
      for (int step = 1; step < _mpc_steps; ++step) {
        Eigen::VectorXd warm_state =
            _prev_x0.segment(step * MPCImpl::kNX, MPCImpl::kNX);
        warm_state(MPCImpl::kIndS) -= starting_s;

        _acados_solver.set_output(step, "x", warm_state.data());
        _acados_solver.set_output(step, "u", &_prev_u0[step * MPCImpl::kNU]);
        // std::cout << "x " << step << "\t" << warm_state.transpose() << "\n";
        // std::cout
        //     << "u " << step << "\t"
        //     << _prev_u0.segment(step * MPCImpl::kNU, MPCImpl::kNU).transpose()
        //     << "\n";
      }

      Eigen::VectorXd xN_prev = _prev_x0.tail(MPCImpl::kNX);
      xN_prev(MPCImpl::kIndS) -= starting_s;

      _acados_solver.set_output(_mpc_steps, "x", xN_prev.data());
      // Eigen::VectorXd zero_u = Eigen::VectorXd::Zero(MPCImpl::kNU);
      _acados_solver.set_output(_mpc_steps, "u",
                                _prev_u0.tail(MPCImpl::kNU).data());
      // _acados_solver.set_output(_mpc_steps, "u", zero_u.data());

      // std::cout << "x " << _mpc_steps << "\t" << xN_prev.transpose() << "\n";
      // std::cout << "u " << _mpc_steps << "\t"
      //           << _prev_u0.tail(MPCImpl::kNU).transpose() << "\n";
      // std::cout << "u " << _mpc_steps << "\t" << zero_u.transpose() << "\n";

      Eigen::VectorXd uN_prev = _prev_u0.tail(MPCImpl::kNU);
      Eigen::VectorXd xN =
          static_cast<MPCImpl*>(this)->next_state(xN_prev, uN_prev);

      // std::cout << "x " << _mpc_steps + 1 << "\t" << xN.transpose() << "\n";
      // std::cout << "x " << _mpc_steps + 1 << "\t" << xN_prev.transpose()
      //           << "\n";

      _acados_solver.set_output(_mpc_steps + 1, "x", xN.data());
      // _acados_solver.set_output(_mpc_steps + 1, "x", xN_prev.data());
    }
  }

  void warm_start_no_u(const Eigen::VectorXd& initial_state) {
    double x_init[MPCImpl::kNX];
    memcpy(x_init, &initial_state[0], MPCImpl::kNX * sizeof(double));

    Eigen::VectorXd u_init = Eigen::VectorXd::Zero(MPCImpl::kNU);

    for (int step = 0; step < _mpc_steps; ++step) {
      _acados_solver.set_output(step, "x", initial_state.data());
      _acados_solver.set_output(step, "u", u_init.data());
    }

    _acados_solver.set_output(_mpc_steps, "x", x_init);
  }

  void process_solver_output(Eigen::VectorXd& xtraj, Eigen::VectorXd& utraj) {
    // stored as x0, y0,..., x1, y1, ..., xN, yN, ...
    for (int step = 0; step < _mpc_steps; ++step) {
      _acados_solver.get_output(step, "x", &xtraj[step * MPCImpl::kNX]);
      _acados_solver.get_output(step, "u", &utraj[step * MPCImpl::kNU]);
    }

    _acados_solver.get_output(_mpc_steps, "x",
                              &xtraj[_mpc_steps * MPCImpl::kNX]);

    static_cast<MPCImpl*>(this)->map_trajectory_to_buffers(xtraj, utraj);
  }

  bool run_acados_solver(const Eigen::VectorXd& initial_state) {
    // run at most 2 times, if first fails, try with simple initialization
    int status               = 0;
    unsigned int num_retries = 2;
    for (int i = 0; i < num_retries; ++i) {
      status = _acados_solver.solve();

      if (status == ACADOS_SUCCESS) {
        _is_shift_warm = true;
        _solve_success = true;
        break;
      } else {
        _is_shift_warm = false;
        warm_start_no_u(initial_state);
      }
    }

    return status == ACADOS_SUCCESS;
  }

 protected:
  using SolverTraits = types::SolverTraits<MPCImpl>;
  AcadosInterface<SolverTraits> _acados_solver;

  Eigen::VectorXd _state;
  Eigen::VectorXd _odom;

  Eigen::VectorXd _prev_x0;
  Eigen::VectorXd _prev_u0;

  std::map<std::string, double> _params;

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

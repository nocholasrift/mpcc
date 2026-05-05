#include <mpcc/unicycle/casadi_unicycle_interface.h>
#include <mpcc/unicycle/mpcc_unicycle_acados.h>
#include <mpcc/common/termcolor.hpp>

#include <unicycle_model_mpcc_param_indices.h>

#include <array>
#include <csignal>
#include <iostream>
#include <optional>
#include <stdexcept>

using namespace mpcc;

// #ifdef FOUND_PYBIND11
// the declarations in the header cannot be referenced by pybind...
// need to define them
const uint16_t UnicycleMPCC::kNX;
const uint16_t UnicycleMPCC::kNP;
const uint16_t UnicycleMPCC::kNU;
const uint16_t UnicycleMPCC::kNBX0;
// #endif

UnicycleMPCC::UnicycleMPCC(std::shared_ptr<MPCConfig> cfg) {
  _has_run   = false;
  _odom_init = false;

  _traj_alignment_threshold = .1;
  _alignment_p_gain         = 1;

  _state = Eigen::VectorXd::Zero(kNX);
  _odom  = Eigen::VectorXd(3);

  _use_dyna_obs  = false;
  _is_shift_warm = false;
  _solve_success = false;

  load_params(cfg);

  // cpg_update_A_mat(0, -1.1);
}

UnicycleMPCC::~UnicycleMPCC() {}

void UnicycleMPCC::load_params(std::shared_ptr<MPCConfig> cfg) {
  _mpc_cfg = cfg;

  // std::cout << "ACCESSING MPC STEPS\n";
  double mpc_steps = _mpc_cfg->steps;
  // std::cout << "DONE\n";
  // int status = _acados_solver.initialize(mpc_steps);

  mpc_x.resize(mpc_steps + 1);
  mpc_y.resize(mpc_steps + 1);
  mpc_theta.resize(mpc_steps + 1);
  mpc_linvels.resize(mpc_steps + 1);
  mpc_s.resize(mpc_steps + 1);
  mpc_s_dot.resize(mpc_steps + 1);

  mpc_angvels.resize(mpc_steps);
  mpc_linaccs.resize(mpc_steps);
  mpc_s_ddots.resize(mpc_steps);

  reset_horizon();

  if (_prev_x0.size() != (mpc_steps + 1) * kNX) {
    std::cout << termcolor::yellow
              << "x0, u0 size differs from mpc_steps size, resizing and "
                 "zeroing out\n";
    _prev_x0 = Eigen::VectorXd::Zero((mpc_steps + 1) * kNX);
    _prev_u0 = Eigen::VectorXd::Zero(mpc_steps * kNU);
  }

  bool status = _acados_solver.initialize(mpc_steps);
  if (status) {
    throw std::runtime_error("Acados initialization failed with status + " +
                             std::to_string(status) + "!");
  }

  std::cout << "!! MPC Obj parameters updated !! " << std::endl;
  std::cout << "!! ACADOS model instantiated !! " << std::endl;
}

bool UnicycleMPCC::set_solver_parameters(const types::Corridor& corridor) {
  using Side      = types::Corridor::Side;
  namespace Param = mpcc::unicycle_param;

  std::vector<double> params;
  params.resize(Param::kNP);

  const TrajectoryView& traj_view = corridor.get_trajectory().view();
  const auto& ctrls_x             = traj_view.xs;
  const auto& ctrls_y             = traj_view.ys;

  // this shoudl never happen... but just in case.
  if (corridor.get_above_poly().get_degree() !=
      corridor.get_below_poly().get_degree()) {
    std::cerr << termcolor::yellow
              << "[MPCC] tube degrees do not match for above and below!"
              << Param::kNP << termcolor::reset << std::endl;
    return false;
  }

  int N_tube_coeffs = corridor.get_above_poly().get_coeffs().size();
  int provided_params =
      ctrls_x.size() + ctrls_y.size() + 2 * N_tube_coeffs + Param::kNP_Scalar;
  if (provided_params != Param::kNP) {
    std::cerr << termcolor::yellow << "[MPCC] provided param count "
              << provided_params << " does not match acados parameter size "
              << Param::kNP << termcolor::reset << std::endl;

    return false;
  }

  params[Param::k_Q_c]       = _mpc_cfg->weights.w_contour_e;
  params[Param::k_Q_l]       = _mpc_cfg->weights.w_lag_e;
  params[Param::k_Q_s]       = _mpc_cfg->weights.w_speed;
  params[Param::k_alpha_abv] = _mpc_cfg->cbf.alpha_abv;
  params[Param::k_alpha_blw] = _mpc_cfg->cbf.alpha_blw;
  params[Param::k_Ql_c]      = _mpc_cfg->clf.w_contour_e;
  params[Param::k_Ql_l]      = _mpc_cfg->clf.w_lag_e;
  params[Param::k_gamma]     = _mpc_cfg->clf.gamma;
  params[Param::k_L_path]    = traj_view.arclen;

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

  double mpc_steps = _mpc_cfg->steps;
  for (int step = 0; step < mpc_steps + 1; ++step) {
    _acados_solver.update_params(step, params);
  }

  return true;
}

Eigen::VectorXd UnicycleMPCC::next_state(const Eigen::VectorXd& current_state,
                                         const Eigen::VectorXd& control) {
  Eigen::VectorXd ret(kNX);

  // Extracting current state values
  double x1     = current_state(kIndX);
  double y1     = current_state(kIndY);
  double theta1 = current_state(kIndTheta);
  double v1     = current_state(kIndV);
  double s1     = current_state(kIndS);
  double sdot1  = current_state(kIndSDot);

  // Extracting control inputs
  double a     = control(kIndLinAcc);
  double w     = control(kIndAngVel);
  double sddot = control(kIndSDDot);

  // needed params
  double dt         = _mpc_cfg->dt;
  double max_linvel = _mpc_cfg->constraints.max_linvel;

  // Dynamics equations
  // for numerical reasons, theta is not wrapped before going into solver.
  ret(kIndX)     = x1 + v1 * cos(theta1) * dt;
  ret(kIndY)     = y1 + v1 * sin(theta1) * dt;
  ret(kIndTheta) = theta1 + w * dt;
  ret(kIndV)     = std::max(std::min(v1 + a * dt, max_linvel), -max_linvel);
  ret(kIndS)     = s1 + sdot1 * dt;
  ret(kIndSDot) =
      std::max(std::min(sdot1 + sddot * dt, max_linvel), -max_linvel);

  return ret;
}

void UnicycleMPCC::map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                             const Eigen::VectorXd& utraj) {

  double mpc_steps = _mpc_cfg->steps;
  for (int i = 0; i <= mpc_steps; ++i) {
    mpc_x[i]       = xtraj[kIndX + i * kIndStateInc];
    mpc_y[i]       = xtraj[kIndY + i * kIndStateInc];
    mpc_theta[i]   = xtraj[kIndTheta + i * kIndStateInc];
    mpc_linvels[i] = xtraj[kIndV + i * kIndStateInc];
    mpc_s[i]       = xtraj[kIndS + i * kIndStateInc];
    mpc_s_dot[i]   = xtraj[kIndSDot + i * kIndStateInc];
  }

  for (int i = 0; i < mpc_steps; ++i) {
    mpc_angvels[i] = utraj[kIndAngVel + i * kIndInputInc];
    mpc_linaccs[i] = utraj[kIndLinAcc + i * kIndInputInc];
    mpc_s_ddots[i] = utraj[kIndSDDot + i * kIndInputInc];
  }
}

Eigen::VectorXd UnicycleMPCC::prepare_initial_state(
    const Eigen::VectorXd& state, const types::Corridor& corridor) {
  Eigen::VectorXd x0 = state;
  if (_has_run) {
    Eigen::VectorXd prev_x0 = _prev_x0.head(kNX);
    double etheta           = x0(kIndTheta) - prev_x0(kIndTheta);
    if (etheta > M_PI)
      x0(kIndTheta) -= 2 * M_PI;
    if (etheta < -M_PI)
      x0(kIndTheta) += 2 * M_PI;
  }

  // x0(kIndSDot) = x0(kIndV);
  double min_sdot = 0.1;
  x0(kIndSDot)    = std::max(_prev_x0(kNX + kIndSDot), min_sdot);

  return x0;
}

std::array<double, 2> UnicycleMPCC::compute_mpc_vel_command(
    const Eigen::VectorXd& state, const Eigen::VectorXd& u) {

  // double curr_angvel =
  //     limit(_prev_u0[kIndAngVel], u[kIndAngVel], _max_anga, _dt);
  double curr_angvel = u[kIndAngVel];

  // make sure velocity does not violate acc bounds, then cap
  double dt      = _mpc_cfg->dt;
  double new_vel = limit(state[kIndV], state[kIndV] + u[kIndLinAcc] * dt,
                         _mpc_cfg->constraints.max_linacc, dt);
  new_vel = std::max(-_mpc_cfg->constraints.max_linvel,
                     std::min(new_vel, _mpc_cfg->constraints.max_linvel));

  return {new_vel, curr_angvel};
}

void UnicycleMPCC::reset_horizon() {
  double mpc_steps = _mpc_cfg->steps;
  for (int i = 0; i < mpc_steps + 1; ++i) {
    mpc_x[i]       = _state(kIndX);
    mpc_y[i]       = _state(kIndY);
    mpc_theta[i]   = 0;
    mpc_linvels[i] = 0;
    mpc_s[i]       = 1e-2;
    mpc_s_dot[i]   = 0;
  }

  for (int i = 0; i < mpc_steps; ++i) {
    mpc_angvels[i] = 0;
    mpc_linaccs[i] = 0;
    mpc_s_ddots[i] = 0;
  }
}

const std::array<Eigen::VectorXd, 2> UnicycleMPCC::get_state_limits() const {
  Eigen::VectorXd xmin(kNX), xmax(kNX);
  double max_linvel  = _mpc_cfg->constraints.max_linvel;
  double bound_value = _mpc_cfg->constraints.bound_value;
  xmin << -bound_value, -bound_value, -M_PI, -max_linvel, 0, -max_linvel;
  xmax << bound_value, bound_value, M_PI, max_linvel, _mpc_cfg->ref_length,
      max_linvel;

  return {xmin, xmax};
}

const std::array<Eigen::VectorXd, 2> UnicycleMPCC::get_input_limits() const {
  Eigen::VectorXd umin(kNU), umax(kNU);
  double max_angvel = _mpc_cfg->constraints.max_angvel;
  double max_linacc = _mpc_cfg->constraints.max_linacc;

  umin << -max_angvel, -max_linacc, -max_linacc;
  umax << max_angvel, max_linacc, max_linacc;

  return {umin, umax};
}

Eigen::VectorXd UnicycleMPCC::get_cbf_data(const types::Corridor& corridor,
                                           size_t horizon_idx) const {
  Eigen::VectorXd state = _prev_x0.segment(horizon_idx * kNX, kNX);
  Eigen::VectorXd input = _prev_u0.segment(horizon_idx * kNU, kNU);

  // if s > traj len, we get nan reports, not good.
  // s >= 1e-2 already through mpc construction
  constexpr double eps = 1e-2;
  double arclen        = corridor.get_trajectory().get_arclen();
  state[kIndS]         = std::min(state[kIndS], arclen - eps);
  state[kIndS]         = std::max(state[kIndS], eps);

  CasadiUnicycleInterface::Params params;
  params.qc_lyap    = _mpc_cfg->clf.w_contour_e;
  params.ql_lyap    = _mpc_cfg->clf.w_lag_e;
  params.gamma_lyap = _mpc_cfg->clf.gamma;

  CasadiUnicycleInterface casadi_interface;
  double h_abv = casadi_interface.get_h_abv(state, input, corridor, params);
  double hdot_abv =
      casadi_interface.get_h_dot_abv(state, input, corridor, params);

  double h_blw = casadi_interface.get_h_blw(state, input, corridor, params);
  double hdot_blw =
      casadi_interface.get_h_dot_blw(state, input, corridor, params);

  auto is_nan = [](const std::vector<double>& candidates) {
    for (auto candidate : candidates) {
      if (std::isnan(candidate))
        return true;
    }
    return false;
  };

  if (is_nan({h_abv, hdot_abv, h_blw, hdot_blw})) {
    std::cout << "traj arclen: " << arclen << std::endl;
    std::cout << "state is: " << state.transpose() << std::endl;
    std::cout << "h_abv is: " << h_abv << std::endl;
    std::cout << "h_dot_abv is: " << hdot_abv << std::endl;
    std::cout << "h_blw is: " << h_blw << std::endl;
    std::cout << "h_dot_blw is: " << hdot_blw << std::endl;
  }

  return Eigen::Vector4d(h_abv, hdot_abv, h_blw, hdot_blw);
}

UnicycleMPCC::MPCHorizon UnicycleMPCC::get_horizon() const {

  double mpc_steps = _mpc_cfg->steps;
  UnicycleMPCC::MPCHorizon horizon;

  horizon.states.xs          = utils::vector_to_eigen(mpc_x);
  horizon.states.ys          = utils::vector_to_eigen(mpc_y);
  horizon.states.thetas      = utils::vector_to_eigen(mpc_theta);
  horizon.states.vs          = utils::vector_to_eigen(mpc_linvels);
  horizon.states.arclens     = utils::vector_to_eigen(mpc_s);
  horizon.states.arclens_dot = utils::vector_to_eigen(mpc_s_dot);

  horizon.inputs.angvels      = utils::vector_to_eigen(mpc_angvels);
  horizon.inputs.linaccs      = utils::vector_to_eigen(mpc_linaccs);
  horizon.inputs.arclens_ddot = utils::vector_to_eigen(mpc_s_ddots);
  horizon.length              = mpc_steps + 1;

  const auto N = mpc_steps + 1;
  assert(horizon.states.xs.size() == N);
  assert(horizon.states.ys.size() == N);
  assert(horizon.states.thetas.size() == N);
  assert(horizon.states.vs.size() == N);
  assert(horizon.states.arclens.size() == N);
  assert(horizon.states.arclens_dot.size() == N);

  assert(horizon.inputs.angvels.size() == N - 1);
  assert(horizon.inputs.linaccs.size() == N - 1);
  assert(horizon.inputs.arclens_ddot.size() == N - 1);

  return horizon;
}

// object is orientable (see orientable.h), so we must check trajectory alignemnt
// before
std::optional<std::array<double, 2>> UnicycleMPCC ::presolve_hook(
    const Eigen::VectorXd& state, const types::Corridor& corridor) {
  double eps_s = .05;

  const auto& reference = corridor.get_trajectory();
  double current_s      = reference.get_closest_s(state.head(2));

  // only attempt to align if we are near beginning of trajectory, otherwise just let robot
  // run. Otherwise we will be stopping a lot along the trajectory whenever we are off, even
  // if CBF is engaged to go off course of trajectory.
  if (current_s > eps_s || reference.get_arclen() < eps_s) {
    return std::nullopt;
  }

  _prev_x0[kIndX]     = state(kIndX);
  _prev_x0[kIndY]     = state(kIndY);
  _prev_x0[kIndTheta] = state(kIndTheta);
  _prev_x0[kIndV]     = state(kIndV);
  _prev_x0[kIndS]     = state(kIndS);
  _prev_x0[kIndSDot]  = state(kIndSDot);

  reset_horizon();

  Eigen::Vector2d point =
      reference(current_s + eps_s, types::Trajectory::kFirstOrder);

  double traj_heading = atan2(point[1], point[0]);

  // std::cout << "theta is: " << state[kIndTheta] << "\n";
  // std::cout << "prop thresh is: " << _mpc_cfg->prop.gain_thresh << "\n";
  if (!is_aligned(traj_heading, state[kIndTheta], _mpc_cfg->prop.gain_thresh)) {
    std::cout << termcolor::yellow
              << "Unicycle model is executing presolve hook!\n"
              << termcolor::reset << std::endl;
    double max_angvel = _mpc_cfg->constraints.max_angvel;
    double desired_angvel =
        get_orient_control(traj_heading, state[kIndTheta], _mpc_cfg->prop.gain,
                           -max_angvel, max_angvel);

    // std::cout << "prop gain: " << _mpc_cfg->prop.gain << "\n";
    // std::cout << "desired angvel: " << desired_angvel << "\n";
    double desired_vel = 0;

    _prev_u0[kIndLinAcc] = 0.;
    _prev_u0[kIndAngVel] = 0.;
    _prev_u0[kIndSDDot]  = 0.;

    return std::optional<std::array<double, 2>>({desired_vel, desired_angvel});
  }

  return std::nullopt;
}

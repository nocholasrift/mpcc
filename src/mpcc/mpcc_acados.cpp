#include <mpcc/mpcc_acados.h>

#include <array>
#include <csignal>
#include <iostream>
#include <mpcc/termcolor.hpp>
#include <stdexcept>

using namespace mpcc;

#ifdef FOUND_PYBIND11
// the declarations in the header cannot be referenced by pybind...
// need to define them
const uint16_t UnicycleMPCC::kNX;
const uint16_t UnicycleMPCC::kNP;
const uint16_t UnicycleMPCC::kNU;
const uint16_t UnicycleMPCC::kNBX0;
#endif

UnicycleMPCC::UnicycleMPCC() {
  // Set default value
  _dt          = .1;
  _mpc_steps   = 10;
  _max_angvel  = 3.0;  // Maximal angvel radian (~30 deg)
  _max_linvel  = 2.0;  // Maximal linvel accel
  _max_linacc  = 4.0;  // Maximal linacc accel
  _max_anga    = 2 * M_PI;
  _bound_value = 1.0e3;  // Bound value for other variables

  _w_ql      = 50.0;
  _w_qc      = .1;
  _w_q_speed = .3;
  _w_ql_lyap = 1;
  _w_qc_lyap = 1;

  _gamma     = .5;
  _use_cbf   = false;
  _alpha_abv = 1.0;
  _alpha_blw = 1.0;
  _colinear  = 0.01;
  _padding   = .05;

  _use_eigen   = false;
  _ref_samples = 10;
  _ref_len_sz  = 4.0;

  _has_run   = false;
  _odom_init = false;

  _traj_alignment_threshold = .1;
  _alignment_p_gain         = 1;

  _s_dot = 0;

  _state = Eigen::VectorXd::Zero(kNX);

  _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
  _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);

  mpc_x.resize(_mpc_steps + 1);
  mpc_y.resize(_mpc_steps + 1);
  mpc_theta.resize(_mpc_steps + 1);
  mpc_linvels.resize(_mpc_steps + 1);
  mpc_s.resize(_mpc_steps + 1);
  mpc_s_dot.resize(_mpc_steps + 1);

  mpc_angvels.resize(_mpc_steps);
  mpc_linaccs.resize(_mpc_steps);
  mpc_s_ddots.resize(_mpc_steps);

  _use_dyna_obs  = false;
  _is_shift_warm = false;
  _solve_success = false;

  _odom = Eigen::VectorXd(3);

  _acados_solver.initialize(_mpc_steps);

  // cpg_update_A_mat(0, -1.1);
}

UnicycleMPCC::~UnicycleMPCC() {}

void UnicycleMPCC::load_params(const std::map<std::string, double>& params) {
  _params = params;

  // Init parameters for MPC object
  _dt = params.find("DT") != params.end() ? params.at("DT") : _dt;
  _mpc_steps =
      _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;
  _max_angvel = _params.find("ANGVEL") != _params.end() ? _params.at("ANGVEL")
                                                        : _max_angvel;
  _max_linvel = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL")
                                                        : _max_linvel;
  _max_linacc = _params.find("MAX_LINACC") != _params.end()
                    ? _params.at("MAX_LINACC")
                    : _max_linacc;
  _max_anga = _params.find("MAX_ANGA") != _params.end() ? _params.at("MAX_ANGA")
                                                        : _max_anga;

  _bound_value = _params.find("BOUND") != _params.end() ? _params.at("BOUND")
                                                        : _bound_value;

  _w_angvel   = params.find("W_ANGVEL") != params.end() ? params.at("W_ANGVEL")
                                                        : _w_angvel;
  _w_angvel_d = params.find("W_DANGVEL") != params.end()
                    ? params.at("W_DANGVEL")
                    : _w_angvel_d;
  _w_linvel_d =
      params.find("W_DA") != params.end() ? params.at("W_DA") : _w_linvel_d;

  _w_ql = params.find("W_LAG") != params.end() ? params.at("W_LAG") : _w_ql;
  _w_qc =
      params.find("W_CONTOUR") != params.end() ? params.at("W_CONTOUR") : _w_qc;
  _w_q_speed = params.find("W_SPEED") != params.end() ? params.at("W_SPEED")
                                                      : _w_q_speed;

  _ref_len_sz  = params.find("REF_LENGTH") != params.end()
                     ? params.at("REF_LENGTH")
                     : _ref_len_sz;
  _ref_samples = params.find("REF_SAMPLES") != params.end()
                     ? params.at("REF_SAMPLES")
                     : _ref_samples;

  _gamma     = params.find("CLF_GAMMA") != params.end() ? params.at("CLF_GAMMA")
                                                        : _gamma;
  _w_ql_lyap = params.find("CLF_W_LAG") != params.end() ? params.at("CLF_W_LAG")
                                                        : _w_ql_lyap;
  _w_qc_lyap = params.find("CLF_W_CONTOUR") != params.end()
                   ? params.at("CLF_W_CONTOUR")
                   : _w_ql_lyap;

  _use_cbf =
      params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
  _alpha_abv = params.find("CBF_ALPHA_ABV") != params.end()
                   ? params.at("CBF_ALPHA_ABV")
                   : _alpha_abv;
  _alpha_blw = params.find("CBF_ALPHA_BLW") != params.end()
                   ? params.at("CBF_ALPHA_BLW")
                   : _alpha_blw;
  _colinear  = params.find("CBF_COLINEAR") != params.end()
                   ? params.at("CBF_COLINEAR")
                   : _colinear;
  _padding   = params.find("CBF_PADDING") != params.end()
                   ? params.at("CBF_PADDING")
                   : _padding;

  utils::get_param(params, "ANGLE_GAIN", _alignment_p_gain);
  utils::get_param(params, "ANGLE_THRESH", _traj_alignment_threshold);

  int status = _acados_solver.initialize(_mpc_steps);
  if (status) {
    throw std::runtime_error("Acados initialization failed with status + " +
                             std::to_string(status) + "!");
  }

  mpc_x.resize(_mpc_steps + 1);
  mpc_y.resize(_mpc_steps + 1);
  mpc_theta.resize(_mpc_steps + 1);
  mpc_linvels.resize(_mpc_steps + 1);
  mpc_s.resize(_mpc_steps + 1);
  mpc_s_dot.resize(_mpc_steps + 1);

  mpc_angvels.resize(_mpc_steps);
  mpc_linaccs.resize(_mpc_steps);
  mpc_s_ddots.resize(_mpc_steps);

  if (_prev_x0.size() != (_mpc_steps + 1) * kNX) {
    std::cout << termcolor::yellow
              << "x0, u0 size differs from mpc_steps size, resizing and "
                 "zeroing out\n";
    _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
    _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);
  }

  std::cout << "!! MPC Obj parameters updated !! " << std::endl;
  std::cout << "!! ACADOS model instantiated !! " << std::endl;
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

  // Dynamics equations
  // for numerical reasons, theta is not wrapped before going into solver.
  ret(kIndX)     = x1 + v1 * cos(theta1) * _dt;
  ret(kIndY)     = y1 + v1 * sin(theta1) * _dt;
  ret(kIndTheta) = theta1 + w * _dt;
  ret(kIndV)     = std::max(std::min(v1 + a * _dt, _max_linvel), -_max_linvel);
  ret(kIndS)     = s1 + sdot1 * _dt;
  ret(kIndSDot) =
      std::max(std::min(sdot1 + sddot * _dt, _max_linvel), -_max_linvel);

  return ret;
}

void UnicycleMPCC::map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                             const Eigen::VectorXd& utraj) {

  for (int i = 0; i <= _mpc_steps; ++i) {
    mpc_x[i]       = xtraj[kIndX + i * kIndStateInc];
    mpc_y[i]       = xtraj[kIndY + i * kIndStateInc];
    mpc_theta[i]   = xtraj[kIndTheta + i * kIndStateInc];
    mpc_linvels[i] = xtraj[kIndV + i * kIndStateInc];
    mpc_s[i]       = xtraj[kIndS + i * kIndStateInc];
    mpc_s_dot[i]   = xtraj[kIndSDot + i * kIndStateInc];
  }

  for (int i = 0; i < _mpc_steps; ++i) {
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
  x0(kIndSDot) = _prev_x0(kNX + kIndSDot);

  return x0;
}

std::array<double, 2> UnicycleMPCC::compute_mpc_vel_command(
    const Eigen::VectorXd& state, const Eigen::VectorXd& u) {

  // double curr_angvel =
  //     limit(_prev_u0[kIndAngVel], u[kIndAngVel], _max_anga, _dt);
  double curr_angvel = u[kIndAngVel];

  // make sure velocity does not violate acc bounds, then cap
  double new_vel =
      limit(state[kIndV], state[kIndV] + u[kIndLinAcc] * _dt, _max_linacc, _dt);
  new_vel = std::max(-_max_linvel, std::min(new_vel, _max_linvel));

  return {new_vel, curr_angvel};
}

void UnicycleMPCC::reset_horizon() {
  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_x[i]       = _odom(0);
    mpc_y[i]       = _odom(1);
    mpc_theta[i]   = 0;
    mpc_linvels[i] = 0;
    mpc_s[i]       = 0;
    mpc_s_dot[i]   = 0;
  }

  for (int i = 0; i < _mpc_steps - 1; ++i) {
    mpc_angvels[i] = 0;
    mpc_linaccs[i] = 0;
    mpc_s_ddots[i] = 0;
  }
}

const std::array<Eigen::VectorXd, 2> UnicycleMPCC::get_state_limits() const {
  Eigen::VectorXd xmin(kNX), xmax(kNX);
  xmin << -_bound_value, -_bound_value, -M_PI, -_max_linvel, 0, -_max_linvel;
  xmax << _bound_value, _bound_value, M_PI, _max_linvel, _ref_len_sz,
      _max_linvel;

  return {xmin, xmax};
}

const std::array<Eigen::VectorXd, 2> UnicycleMPCC::get_input_limits() const {
  Eigen::VectorXd umin(kNU), umax(kNU);
  umin << -_max_angvel, -_max_linacc, -_max_linacc;
  umax << _max_angvel, _max_linacc, _max_linacc;

  return {umin, umax};
}

Eigen::VectorXd UnicycleMPCC::get_cbf_data(const Eigen::VectorXd& state,
                                           const Eigen::VectorXd& control,
                                           bool is_abv) const {
  return Eigen::Vector3d(0., 0., 0.);
}

UnicycleMPCC::MPCHorizon UnicycleMPCC::get_horizon() const {

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
  horizon.length              = _mpc_steps + 1;

  const auto N = _mpc_steps + 1;
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
    const Eigen::VectorXd& state, const types::Corridor& corridor) const {
  double eps_s = .05;

  const auto& reference = corridor.get_trajectory();
  double current_s      = reference.get_closest_s(state.head(2));

  // only attempt to align if we are near beginning of trajectory, otherwise just let robot
  // run. Otherwise we will be stopping a lot along the trajectory whenever we are off, even
  // if CBF is engaged to go off course of trajectory.
  if (current_s > eps_s || reference.get_arclen() < eps_s) {
    return std::nullopt;
  }

  Eigen::Vector2d point =
      reference(current_s + eps_s, types::Trajectory::kFirstOrder);

  double traj_heading = atan2(point[1], point[0]);

  if (!is_aligned(traj_heading, state[kIndTheta], _traj_alignment_threshold)) {
    std::cout << termcolor::yellow
              << "Unicycle model is executing presolve hook!\n"
              << termcolor::reset << std::endl;
    double desired_angvel =
        get_orient_control(traj_heading, state[kIndTheta], _alignment_p_gain,
                           -_max_angvel, _max_angvel);
    double desired_vel = 0;

    return std::optional<std::array<double, 2>>({desired_vel, desired_angvel});
  }

  return std::nullopt;
}

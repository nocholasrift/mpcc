#include "mpcc/mpcc_core.h"

#include <chrono>
#include <cmath>

#include "mpcc/termcolor.hpp"

using namespace mpcc;

MPCCore::MPCCore() {
  // using emplace because all MPC classes have non-copyable
  // solver interface inherited from the base class
  _mpc.emplace<UnicycleMPCC>();
}

MPCCore::MPCCore(const MPCType& mpc_input_type) {
  _mpc_input_type = mpc_input_type;

  if (_mpc_input_type == MPCType::kUnicycle) {
    std::cout << termcolor::green << "[MPC Core] Using unicycle model"
              << termcolor::reset << std::endl;
    _mpc.emplace<UnicycleMPCC>();
  } else if (_mpc_input_type == MPCType::kDoubleIntegrator) {
    std::cout << termcolor::green << "[MPC Core] Using double integrator model"
              << termcolor::reset << std::endl;
    _mpc.emplace<DIMPCC>();
  } else {
    throw std::runtime_error(
        "Invalid MPC input type: " +
        std::to_string(static_cast<unsigned int>(_mpc_input_type)));
  }
}

MPCCore::~MPCCore() {}

void MPCCore::load_params(const std::map<std::string, double>& params) {
  utils::get_param(params, "DT", _dt);
  utils::get_param(params, "MAX_ANGA", _max_anga);
  utils::get_param(params, "MAX_LINACC", _max_linacc);
  utils::get_param(params, "LINVEL", _max_vel);
  utils::get_param(params, "ANGVEL", _max_angvel);
  utils::get_param(params, "ANGLE_GAIN", _prop_gain);
  utils::get_param(params, "ANGLE_THRESH", _prop_angle_thresh);

  _params = params;

  // _mpc->load_params(params);
  call_mpc([&](auto& mpc) { mpc.load_params(params); });
}

void MPCCore::set_odom(const Eigen::Vector3d& odom) {
  _odom = odom;
  // _mpc->set_odom(odom);
  call_mpc([&](auto& mpc) { mpc.set_odom(odom); });
}

void MPCCore::set_trajectory(const Eigen::VectorXd& x_pts,
                             const Eigen::VectorXd& y_pts,
                             const Eigen::VectorXd& knot_parameters) {

  // need to extend trajectory to REF_LENGTH because acados MPC can only
  // handle a fixed length trajectory,which has been fixed at REF_LENGTH
  int N                      = knot_parameters.size();
  double required_mpc_length = _params["REF_LENGTH"];
  _trajectory                = types::Trajectory(knot_parameters, x_pts, y_pts);
  _trajectory.extend_to_length(required_mpc_length);

  _ref_length      = _trajectory.get_extended_length();
  _true_ref_length = _trajectory.get_true_length();

  std::cout << "received trajectory of length: " << _ref_length << "\n";
  std::cout << "trajectory has " << N << " knots\n";

  _is_set = true;
}

// void MPCCore::set_tubes(const std::vector<Spline1D>& tubes)
void MPCCore::set_tubes(const std::array<Eigen::VectorXd, 2>& tubes) {
  // _mpc->set_tubes(tubes);
  call_mpc([&](auto& mpc) { mpc.set_tubes(tubes); });
}

const std::array<Eigen::VectorXd, 2>& MPCCore::get_tubes() const {
  // return _mpc->get_tubes();
  return call_mpc([&](auto& mpc) -> decltype(auto) { return mpc.get_tubes(); });
}

std::array<double, 2> MPCCore::solve(const Eigen::VectorXd& state,
                                     bool is_reverse) {
  if (!_is_set) {
    std::cout << termcolor::yellow << "[MPC Core] trajectory not set!"
              << termcolor::reset << std::endl;
    return {0, 0};
  }

  _traj_reset = false;

  double time_to_solve = 0.;

  double current_s = std::max(_trajectory.get_closest_s(state.head(2)), 1e-6);
  double s_dot = std::min(std::max((current_s - _prev_s) / _dt, 0.), _max_vel);

  int N_virtual_states = 2;
  Eigen::VectorXd mpcc_state(state.size() + N_virtual_states);
  mpcc_state << state, 0., s_dot;

  // Just like with trajectory length,acados MPC also has fixed number of knots
  // that can be used for the trajectory due to a quirk with Casadi Splines.
  unsigned int required_mpc_knots =
      static_cast<unsigned int>(_params["REF_SAMPLES"]);
  types::Trajectory adjusted_traj =
      _trajectory.get_adjusted_traj(current_s, required_mpc_knots);

  auto start                        = std::chrono::high_resolution_clock::now();
  std::array<double, 2> mpc_command = call_mpc([&](auto& mpc) {
    return mpc.solve(mpcc_state, adjusted_traj, is_reverse);
  });
  // std::array<double, 2> mpc_command = call_mpc([&](auto& mpc) {
  //   return mpc.solve(state, adjusted_traj.view(), is_reverse);
  // });

  auto end = std::chrono::high_resolution_clock::now();

  time_to_solve =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  _has_run     = true;
  _curr_vel    = mpc_command[0];
  _curr_angvel = mpc_command[1];

  _prev_s = current_s;

  return mpc_command;
}

Eigen::VectorXd MPCCore::get_cbf_data(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control,
                                      bool is_abv) const {
  // return _mpc->get_cbf_data(state, control, is_abv);
  return call_mpc(
      [&](auto& mpc) { return mpc.get_cbf_data(state, control, is_abv); });
}

const bool MPCCore::get_solver_status() const {
  // return _mpc->get_solver_status();
  return call_mpc([&](auto& mpc) { return mpc.get_solver_status(); });
}

const double MPCCore::get_true_ref_len() const {
  return _true_ref_length;
}

const Eigen::VectorXd& MPCCore::get_state() const {
  // return _mpc->get_state();
  return call_mpc([&](auto& mpc) -> decltype(auto) { return mpc.get_state(); });
}

const std::array<Eigen::VectorXd, 2> MPCCore::get_state_limits() const {
  // return _mpc->get_state_limits();
  return call_mpc([&](auto& mpc) { return mpc.get_state_limits(); });
}

const std::array<Eigen::VectorXd, 2> MPCCore::get_input_limits() const {
  return call_mpc([&](auto& mpc) { return mpc.get_input_limits(); });
}

MPCCore::AnyHorizon MPCCore::get_horizon() const {

  // return _mpc->get_horizon();
  return call_mpc([&](auto& mpc) -> AnyHorizon { return mpc.get_horizon(); });
}

const std::map<std::string, double>& MPCCore::get_params() const {
  return _params;
}

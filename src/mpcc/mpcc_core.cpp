#include <mpcc/mpcc_core.h>
#include <mpcc/termcolor.hpp>

#include <chrono>
#include <stdexcept>
#include "mpcc/utils.h"

using namespace mpcc;

MPCCore::MPCCore() {
  // using emplace because all MPC classes have non-copyable
  // solver interface inherited from the base class
  _mpc.emplace<UnicycleMPCC>();
  _tube_generator.set_verbose(false);
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
  _tube_generator.set_verbose(false);
}

MPCCore::~MPCCore() {}

void MPCCore::load_params(const std::map<std::string, double>& params) {
  utils::get_param(params, "DT", _dt);
  utils::get_param(params, "REF_LENGTH", _tube_horizon);
  utils::get_param(params, "MAX_ANGA", _max_anga);
  utils::get_param(params, "MAX_LINACC", _max_linacc);
  utils::get_param(params, "LINVEL", _max_vel);
  utils::get_param(params, "ANGVEL", _max_angvel);
  utils::get_param(params, "ANGLE_GAIN", _prop_gain);
  utils::get_param(params, "ANGLE_THRESH", _prop_angle_thresh);

  // hack for now, will come back to fix...
  double tube_degree{0.}, tube_samples{0.};
  utils::get_param(params, "TUBE_DEGREE", tube_degree);
  utils::get_param(params, "TUBE_SAMPLES", tube_samples);
  utils::get_param(params, "MAX_TUBE_WIDTH", _max_tube_width);
  utils::get_param(params, "REF_LENGTH", _tube_horizon);

  _tube_degree  = static_cast<int>(tube_degree);
  _tube_samples = static_cast<int>(tube_samples);
  _max_tube_width /= 2.;

  double mpc_steps = _mpc_steps;
  utils::get_param(params, "STEPS", mpc_steps);
  _mpc_steps = static_cast<int>(mpc_steps);

  tube::TubeGenerator::Settings tube_settings;
  tube_settings.degree       = _tube_degree;
  tube_settings.num_samples  = _tube_samples;
  tube_settings.max_distance = _max_tube_width;

  _tube_generator.update_settings(tube_settings);

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
  int N = knot_parameters.size();
  // double required_mpc_length = _params["REF_LENGTH"];
  _trajectory              = types::Trajectory(knot_parameters, x_pts, y_pts);
  _non_extended_trajectory = _trajectory;

  // _trajectory.extend_to_length(required_mpc_length);

  double traj_len = _trajectory.get_arclen();
  double horizon  = std::min(_tube_horizon, traj_len);

  // tube::TubeGenerator::Settings tube_settings;
  // tube_settings.degree       = _tube_degree;
  // tube_settings.num_samples  = _tube_samples;
  // tube_settings.max_distance = _max_tube_width;
  // tube_settings.len_start    = 0;
  // tube_settings.horizon      = horizon;
  //
  // _is_tube_generated =
  //     _tube_generator.generate(*_map_util, _trajectory, tube_settings);

  std::cout << "received trajectory of length: " << _trajectory.get_arclen()
            << "\n";
  std::cout << "trajectory has " << N << " knots\n";
  std::cout << "start point: " << _trajectory(0).transpose() << "\n";

  _is_traj_set = true;
  _traj_reset  = true;
}

std::array<double, 2> MPCCore::solve(const Eigen::VectorXd& state,
                                     bool is_reverse) {

  if (!_is_traj_set || !_is_map_util_set) {
    std::cout << termcolor::yellow << "[MPC Core] trajectory or map not set!"
              << termcolor::reset << std::endl;
    return {0, 0};
  }

  // get s value on taj closest to robot position
  double current_s = std::max(_trajectory.get_closest_s(state.head(2)), 1e-6);

  // define the tubes right before solving.
  if (_has_run) {
    AnyHorizon horizon = get_horizon();

    size_t horizon_steps =
        std::visit([&](const auto& arg) { return arg.length; }, horizon);
    double last_s = std::visit(
        [&](const auto& arg) {
          return arg.get_arclen_at_step(horizon_steps - 1);
        },
        horizon);

    double max_possible_horizon_dist = _max_vel * _dt * _mpc_steps;
    std::cout << "LAST S WAS: " << last_s << "\n";
    std::cout << "remaining len: " << _trajectory.get_arclen() - current_s
              << "\n";
    std::cout << "MAX POSSIBLE HORIZ DIST " << max_possible_horizon_dist
              << "\n";
    if (std::abs(_trajectory.get_arclen() - current_s - last_s) <
        1.5 * max_possible_horizon_dist) {
      double extend_len = _trajectory.get_arclen() + max_possible_horizon_dist;
      _trajectory       = utils::extend_trajectory(_trajectory, extend_len);
      std::cout << "near trajectory end, extending to len: "
                << _trajectory.get_arclen() << "\n";
    }
  }

  std::cout << "non extended traj length: "
            << _non_extended_trajectory.get_arclen() << "\n";

  /*double extend_len = _trajectory.get_arclen() + 2.0;*/
  /*types::Trajectory extended_trajectory =*/
  /*    utils::extend_trajectory(_trajectory, extend_len);*/

  // Just like with trajectory length,acados MPC also has fixed number of knots
  // that can be used for the trajectory due to a quirk with Casadi Splines.
  unsigned int required_mpc_knots =
      static_cast<unsigned int>(_params["REF_SAMPLES"]);

  /*types::Trajectory adjusted_traj =*/
  /*    extended_trajectory.get_adjusted_traj(current_s, required_mpc_knots);*/
  types::Trajectory adjusted_traj =
      _trajectory.get_adjusted_traj(current_s, required_mpc_knots);

  // bool status =
  //     _tube_generator.generate(*_map_util, _trajectory, current_s, horizon);

  // std::cout << "adjusted traj length: " << adjusted_traj.get_arclen() << "\n";

  /*double horizon = std::min(extend_len, adjusted_traj.get_arclen());*/
  double horizon = adjusted_traj.get_arclen();
  // double traj_len = adjusted_traj.get_arclen();
  // if (horizon > traj_len) {
  //   horizon = traj_len;
  // }

  // passing in 0 here because, by construction, adjusted traj will start at s = 0
  double tube_starting_s = 0;
  // bool status            = _tube_generator.generate(*_map_util, adjusted_traj,
  // tube_starting_s, horizon);
  bool status =
      _tube_generator.generate(*_map_util, adjusted_traj, tube_starting_s,
                               _non_extended_trajectory.get_arclen());

  // if (!_is_tube_generated) {
  //
  //   _is_tube_generated =
  //       _tube_generator.generate(*_map_util, _trajectory, tube_settings);
  //
  //   if (!_is_tube_generated) {
  //     std::cout << "[MPCC Code] cannot generate tubes\n";
  //     return {0., 0.};
  //   }
  // }

  // types::Polynomial abv, blw;
  // _tube_generator.shift_tube_domain(current_s, horizon, abv, blw);

  int N_virtual_states = 2;
  double s_dot = std::min(std::max((current_s - _prev_s) / _dt, 0.), _max_vel);
  // std::cout << "prev_s: " << _prev_s << "\tcurrent_s: " << current_s << "\n";
  _prev_s = current_s;

  // if traj reset and had a previous trajectory, use previous traj for first iter
  // of s_dot calculation
  if (_traj_reset && current_s < 2e-2) {
    // double curr_s = std::max(_prev_traj.get_closest_s(state.head(2)), 1e-6);
    // s_dot         = std::min(std::max((curr_s - _prev_s) / _dt, 0.), _max_vel);
    // s_dot = state[3];
    // _prev_s       = curr_s;
  } else {

    _traj_reset = false;
  }

  double mpc_s_offset = 0.;
  Eigen::VectorXd mpcc_state(state.size() + N_virtual_states);
  mpcc_state << state, mpc_s_offset, s_dot;

  // types::Trajectory adjusted_traj =
  //     _trajectory.get_adjusted_traj(current_s, required_mpc_knots);

  types::Corridor corridor(
      adjusted_traj,
      _tube_generator.get_side_poly(tube::TubeGenerator::Side::kAbove),
      _tube_generator.get_side_poly(tube::TubeGenerator::Side::kBelow),
      mpc_s_offset);
  // types::Corridor corridor(adjusted_traj, _mpc_tube[0], _mpc_tube[1],
  //                          mpc_s_offset);

  auto start                        = std::chrono::high_resolution_clock::now();
  std::array<double, 2> mpc_command = call_mpc(
      [&](auto& mpc) { return mpc.solve(mpcc_state, corridor, is_reverse); });

  auto end = std::chrono::high_resolution_clock::now();

  double time_to_solve =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  _has_run     = true;
  _curr_vel    = mpc_command[0];
  _curr_angvel = mpc_command[1];

  return mpc_command;
}

Eigen::VectorXd MPCCore::get_cbf_data(size_t horizon_idx) const {
  // return _mpc->get_cbf_data(state, control, is_abv);
  if (!_has_run) {
    std::cout << "has not run, returning 0s for cbf get data\n";
    return Eigen::VectorXd::Zero(4);
  }

  std::cout << "getting horizon\n";
  MPCCore::AnyHorizon horizon = get_horizon();
  std::cout << "done\n";

  size_t horizon_steps =
      std::visit([](const auto& arg) { return arg.length; }, horizon);

  std::cout << "checking horizon steps\n";

  if (horizon_idx >= horizon_steps) {
    throw std::invalid_argument(
        "[get_cbf_data] passed in horizon_idx exceeds total number of horizon "
        "steps!");
  }

  Eigen::VectorXd init_state = std::visit(
      [&](const auto& arg) { return arg.get_state_at_step(0); }, horizon);

  std::cout << "bulding adjusted traj\n";
  double current_s = std::max(_trajectory.get_closest_s(init_state), 1e-6);
  unsigned int num_samples =
      static_cast<unsigned int>(_params.at("REF_SAMPLES"));
  std::cout << "adjusted traj\n";
  types::Trajectory adjusted_traj =
      _trajectory.get_adjusted_traj(current_s, num_samples);
  std::cout << "done\n";

  std::cout << adjusted_traj.get_ctrls_x() << "\n";
  std::cout << adjusted_traj.get_ctrls_y() << "\n";

  types::Corridor corridor(
      adjusted_traj,
      _tube_generator.get_side_poly(tube::TubeGenerator::Side::kAbove),
      _tube_generator.get_side_poly(tube::TubeGenerator::Side::kBelow),
      current_s);
  std::cout << "finished corridor\n";

  return call_mpc(
      [&](auto& mpc) { return mpc.get_cbf_data(corridor, horizon_idx); });
}

const bool MPCCore::get_solver_status() const {
  // return _mpc->get_solver_status();
  return call_mpc([&](auto& mpc) { return mpc.get_solver_status(); });
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

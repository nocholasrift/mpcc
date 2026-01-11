#include "mpcc/mpcc_core.h"

#include <chrono>
#include <cmath>

#include "mpcc/termcolor.hpp"

using namespace mpcc;

MPCCore::MPCCore() {
  _mpc = std::make_unique<MPCC>();
}

MPCCore::MPCCore(const MPCType& mpc_input_type) {
  _mpc_input_type = mpc_input_type;

  if (_mpc_input_type == MPCType::kUnicycle) {
    std::cout << termcolor::green << "[MPC Core] Using unicycle model"
              << termcolor::reset << std::endl;
    _mpc = std::make_unique<MPCC>();
  } else if (_mpc_input_type == MPCType::kDoubleIntegrator) {
    std::cout << termcolor::green << "[MPC Core] Using double integrator model"
              << termcolor::reset << std::endl;
    _mpc = std::make_unique<DIMPCC>();
  } else {
    throw std::runtime_error(
        "Invalid MPC input type: " +
        std::to_string(static_cast<unsigned int>(_mpc_input_type)));
  }
}

MPCCore::~MPCCore() {}

void MPCCore::get_param(const std::map<std::string, double>& params,
                        const std::string& key, double& value) {
  if (auto it = params.find(key); it != params.end()) {
    value = params.at(key);
  }
}

void MPCCore::load_params(const std::map<std::string, double>& params) {
  get_param(params, "DT", _dt);
  get_param(params, "MAX_ANGA", _max_anga);
  get_param(params, "MAX_LINACC", _max_linacc);
  get_param(params, "LINVEL", _max_vel);
  get_param(params, "ANGVEL", _max_angvel);
  get_param(params, "ANGLE_GAIN", _prop_gain);
  get_param(params, "ANGLE_THRESH", _prop_angle_thresh);

  _params = params;

  _mpc->load_params(params);
}

void MPCCore::set_odom(const Eigen::Vector3d& odom) {
  _odom = odom;
  _mpc->set_odom(odom);
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
  _mpc->set_tubes(tubes);
}

const std::array<Eigen::VectorXd, 2>& MPCCore::get_tubes() const {
  return _mpc->get_tubes();
}

bool MPCCore::orient_robot() {
  // calculate heading error between robot and trajectory start
  // use 1st point as most times first point has 0 velocity

  double start = _trajectory.get_closest_s(_odom.head(2));
  double eps_s = .05;

  Eigen::Vector2d point =
      _trajectory(start + eps_s, mpcc::types::Trajectory::kFirstOrder);
  double traj_heading = atan2(point[1], point[0]);

  // wrap between -pi and pi
  double e = atan2(sin(traj_heading - _odom(2)), cos(traj_heading - _odom(2)));

  if (std::isnan(e)) {
    std::cout << termcolor::red << "[MPC Core] heading error nan, returning"
              << termcolor::reset << std::endl;
    return false;
  }

  std::cout << termcolor::yellow
            << "[MPC Core] trajectory reset, checking if we need to align... "
               "error = "
            << e * 180. / M_PI << " deg" << termcolor::reset << std::endl;

  // if error is larger than _prop_angle_thresh use proportional controller to
  // align
  if (fabs(e) > _prop_angle_thresh) {
    _mpc->reset_horizon();
    _curr_vel = 0;
    _curr_angvel =
        std::max(-_max_angvel, std::min(_max_angvel, _prop_gain * e));

    return true;
  }

  return false;
}

std::array<double, 2> MPCCore::solve(const Eigen::VectorXd& state,
                                     bool is_reverse) {
  if (!_is_set) {
    std::cout << termcolor::yellow << "[MPC Core] trajectory not set!"
              << termcolor::reset << std::endl;
    return {0, 0};
  }

  if (_ref_length > .1 && _traj_reset) {
    if (_mpc_input_type == MPCType::kUnicycle && orient_robot())
      return {_curr_vel, _curr_angvel};
  }

  _traj_reset = false;

  double new_vel;
  double time_to_solve = 0.;

  double current_s = _trajectory.get_closest_s(state.head(2));

  // Just like with trajectory length,acados MPC also has fixed number of knots
  // that can be used for the trajectory due to a quirk with Casadi Splines.
  unsigned int required_mpc_knots =
      static_cast<unsigned int>(_params["REF_SAMPLES"]);
  types::Trajectory adjusted_traj =
      _trajectory.get_adjusted_traj(current_s, required_mpc_knots);

  for (double s = 0; s < adjusted_traj.get_extended_length(); s += 0.1) {
    Eigen::Vector2d der = adjusted_traj(s, 1);
    std::cout << s << ":\t" << der.transpose() << "\t" << der.norm() << "\n";
  }

  std::cout << "cpp xs " << adjusted_traj.get_ctrls_x() << "\n";
  std::cout << "cpp ys " << adjusted_traj.get_ctrls_y() << "\n";

  auto start = std::chrono::high_resolution_clock::now();
  std::array<double, 2> mpc_command =
      _mpc->solve(state, adjusted_traj.view(), is_reverse);

  auto end = std::chrono::high_resolution_clock::now();

  time_to_solve =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  _has_run     = true;
  _curr_vel    = mpc_command[0];
  _curr_angvel = mpc_command[1];

  return mpc_command;
}

Eigen::VectorXd MPCCore::get_cbf_data(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control,
                                      bool is_abv) const {
  return _mpc->get_cbf_data(state, control, is_abv);
}

const bool MPCCore::get_solver_status() const {
  return _mpc->get_solver_status();
}

const double MPCCore::get_true_ref_len() const {
  return _true_ref_length;
}

const Eigen::VectorXd& MPCCore::get_state() const {
  return _mpc->get_state();
}

const std::array<Eigen::VectorXd, 2> MPCCore::get_state_limits() const {
  return _mpc->get_state_limits();
}

const std::array<Eigen::VectorXd, 2> MPCCore::get_input_limits() const {
  return _mpc->get_input_limits();
}

MPCHorizon MPCCore::get_horizon() const {

  return _mpc->get_horizon();

  // TODO: This horizon situation is a disaster and needs to be refactored
  // std::vector<Eigen::VectorXd> ret;
  // if (_mpc_input_type == MPCType::kUnicycle) {
  //   MPCC* _mpc_unicycle = dynamic_cast<MPCC*>(_mpc.get());
  //   ret.reserve(_mpc_unicycle->mpc_x.size());
  //   if (_mpc_unicycle->mpc_x.size() == 0)
  //     return ret;
  //
  //   double t = 0;
  //   for (int i = 0; i < _mpc_unicycle->mpc_x.size() - 1; ++i) {
  //     ret.emplace_back(7);
  //     ret.back() << t, _mpc_unicycle->mpc_x[i], _mpc_unicycle->mpc_y[i],
  //         _mpc_unicycle->mpc_theta[i], _mpc_unicycle->mpc_linvels[i],
  //         _mpc_unicycle->mpc_linaccs[i], _mpc_unicycle->mpc_s[i];
  //     t += _dt;
  //   }
  // } else if (_mpc_input_type == MPCType::kDoubleIntegrator) {
  //   DIMPCC* _mpc_double_integrator = dynamic_cast<DIMPCC*>(_mpc.get());
  //   ret.reserve(_mpc_double_integrator->mpc_x.size());
  //   if (_mpc_double_integrator->mpc_x.size() == 0)
  //     return ret;
  //   double t = 0;
  //   for (int i = 0; i < _mpc_double_integrator->mpc_x.size() - 1; ++i) {
  //     ret.emplace_back(9);
  //     ret.back() << t, _mpc_double_integrator->mpc_x[i],
  //         _mpc_double_integrator->mpc_y[i], _mpc_double_integrator->mpc_vx[i],
  //         _mpc_double_integrator->mpc_vy[i], _mpc_double_integrator->mpc_s[i],
  //         _mpc_double_integrator->mpc_s_dot[i],
  //         _mpc_double_integrator->mpc_ax[i], _mpc_double_integrator->mpc_ay[i];
  //     t += _dt;
  //   }
  // }
  //
  // return ret;
}

const std::map<std::string, double>& MPCCore::get_params() const {
  return _params;
}

const std::array<double, 2> MPCCore::get_mpc_command() const {
  return _mpc->get_command();
}

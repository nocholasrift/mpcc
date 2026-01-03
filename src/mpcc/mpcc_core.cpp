#include "mpcc/mpcc_core.h"

#include <chrono>
#include <cmath>

#include "mpcc/termcolor.hpp"

MPCCore::MPCCore() {
  _mpc            = std::make_unique<MPCC>();
  _mpc_input_type = MPCType::kUnicycle;

  _curr_vel    = 0;
  _curr_angvel = 0;

  _is_set     = false;
  _has_run    = false;
  _traj_reset = false;

  _ref_length = 0;
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

  _curr_vel    = 0;
  _curr_angvel = 0;

  _is_set     = false;
  _has_run    = false;
  _traj_reset = false;

  _ref_length = 0;
}

MPCCore::~MPCCore() {}

void MPCCore::load_params(const std::map<std::string, double>& params) {
  _dt         = params.find("DT") != params.end() ? params.at("DT") : _dt;
  _max_anga   = params.find("MAX_ANGA") != params.end() ? params.at("MAX_ANGA")
                                                        : _max_anga;
  _max_linacc = params.find("MAX_LINACC") != params.end()
                    ? params.at("MAX_LINACC")
                    : _max_linacc;

  _max_vel =
      params.find("LINVEL") != params.end() ? params.at("LINVEL") : _max_vel;
  _max_angvel =
      params.find("ANGVEL") != params.end() ? params.at("ANGVEL") : _max_angvel;

  _prop_gain         = params.find("ANGLE_GAIN") != params.end()
                           ? params.at("ANGLE_GAIN")
                           : _prop_gain;
  _prop_angle_thresh = params.find("ANGLE_THRESH") != params.end()
                           ? params.at("ANGLE_THRESH")
                           : _prop_angle_thresh;

  _params = params;

  _mpc->load_params(params);
}

void MPCCore::set_alpha(const std::array<double, 2>& alphas) {}

void MPCCore::set_dyna_obs(const Eigen::MatrixXd& dyna_obs) {
  /*_mpc->set_dyna_obs(dyna_obs);*/
}

void MPCCore::set_odom(const Eigen::Vector3d& odom) {
  _odom = odom;
  _mpc->set_odom(odom);
}

void MPCCore::set_trajectory(const std::array<Spline1D, 2>& ref, double ref_len,
                             double true_ref_len) {
  _ref             = ref;
  _ref_length      = ref_len;
  _true_ref_length = true_ref_len;
  _is_set          = true;
  _traj_reset      = true;
  _mpc->set_reference(ref, ref_len);
}

void MPCCore::set_trajectory(const Eigen::VectorXd& x_pts,
                             const Eigen::VectorXd& y_pts, int degree,
                             const Eigen::VectorXd& knot_parameters) {
  Spline1D splineX(utils::Interp(x_pts, degree, knot_parameters));
  Spline1D splineY(utils::Interp(y_pts, degree, knot_parameters));

  int N               = knot_parameters.size();
  double true_ref_len = knot_parameters.tail(1)[0];
  double ref_len      = _params["REF_LENGTH"];

  _true_ref_length = true_ref_len;
  if (true_ref_len < ref_len) {
    double end = true_ref_len - 1e-1;
    double px  = splineX(end).coeff(0);
    double py  = splineY(end).coeff(0);
    double dx  = splineX.derivatives(end, 1).coeff(1);
    double dy  = splineY.derivatives(end, 1).coeff(1);

    /*ROS_WARN("(%.2f, %.2f)\t(%.2f, %.2f)", px, py, dx, dy);*/

    double ds = ref_len / (N - 1);

    Eigen::RowVectorXd ss, xs, ys;
    ss.resize(N);
    xs.resize(N);
    ys.resize(N);

    for (int i = 0; i < N; ++i) {
      double s = ds * i;
      ss(i)    = s;

      if (s < true_ref_len) {
        xs(i) = splineX(s).coeff(0);
        ys(i) = splineY(s).coeff(0);
      } else {
        xs(i) = dx * (s - true_ref_len) + px;
        ys(i) = dy * (s - true_ref_len) + py;
      }
    }

    true_ref_len = ss.tail(1)[0];

    const auto fitX = utils::Interp(xs, 3, ss);
    splineX         = Spline1D(fitX);

    const auto fitY = utils::Interp(ys, 3, ss);
    splineY         = Spline1D(fitY);
  }

  std::cout << "received trajectory of length: " << true_ref_len << "\n";
  std::cout << "trajectory has " << N << " knots\n";

  _ref[0]     = splineX;
  _ref[1]     = splineY;
  _ref_length = true_ref_len;
  _is_set     = true;
  _mpc->set_reference(_ref, true_ref_len);
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

  double start = get_s_from_pose(_odom);
  double eps_s = .05;

  /*std::cout << "start is " << start + eps_s << std::endl;*/

  double traj_heading = atan2(_ref[1].derivatives(start + eps_s, 1).coeff(1),
                              _ref[0].derivatives(start + eps_s, 1).coeff(1));

  /*std::cout << "dx " << _ref[0].derivatives(start + eps_s, 1).coeff(1) <<
   * std::endl;*/
  /*std::cout << "dy " << _ref[1].derivatives(start + eps_s, 1).coeff(1) <<
   * std::endl;*/
  /*std::cout << "traj_heading is " << traj_heading << std::endl;*/

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

double MPCCore::get_s_from_pose(const Eigen::VectorXd& pose) const {
  // find the s which minimizes dist to robot
  double s            = 0;
  double min_dist     = 1e6;
  Eigen::Vector2d pos = pose.head(2);
  for (double i = 0.0; i < _ref_length; i += .01) {
    Eigen::Vector2d p =
        Eigen::Vector2d(_ref[0](i).coeff(0), _ref[1](i).coeff(0));

    double d = (pos - p).squaredNorm();
    if (d < min_dist) {
      min_dist = d;
      s        = i;
    }
  }

  return s;
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

  // if (_has_run)
  //   _prev_cmd = _mpc_command.getCommand();

  // if (is_reverse)
  //   _curr_vel = -1 * state(3);
  auto start = std::chrono::high_resolution_clock::now();
  // Eigen::VectorXd state(4);
  // state << _odom(0), _odom(1), _odom(2), _curr_vel;
  std::array<double, 2> mpc_command = _mpc->solve(state, is_reverse);

  // _mpc_command = _mpc->Solve(_state, _reference);
  auto end = std::chrono::high_resolution_clock::now();

  // if (is_reverse)
  //   _mpc_command[1] *= -1;

  time_to_solve =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  /*std::cout << "[MPC Core] Solve time: " << time_to_solve << std::endl;*/

  // std::array<double, 2> input = mpc_command.getCommand();

  // if (time_to_solve > _dt * 1000) {
  //
  //   if (mpc_command.getType() == CommandType::kUnicycle) {
  //     input[1] = _has_run ? _prev_cmd[1] : 0.;
  //
  //     if (mpc_command.getOrder == CommandOrder::kAccel)
  //       input[0] = 0.;
  //     else if (_mpc_command.getOrder == CommandOrder::kVel)
  //       input[0] = _has_run ? prev_cmd[0] : 0;
  //   }
  //
  //   _mpc_command.setCommand(input[0], input[1]);
  // }

  _has_run     = true;
  _curr_vel    = mpc_command[0];
  _curr_angvel = mpc_command[1];

  return mpc_command;

  // new_vel = _curr_vel + _mpc_command[1] * _dt;
  //
  // _curr_angvel = limit(_curr_angvel, _mpc_command[0], _max_anga);
  // _curr_vel = limit(_curr_vel, new_vel, _max_linacc);
  //
  // // ensure vel is between -max and max and ang vel is between -max and max
  // _curr_vel = std::max(-_max_vel, std::min(_max_vel, _curr_vel));
  // _curr_angvel = std::max(-_max_angvel, std::min(_max_angvel, _curr_angvel));
  //
  // std::cerr << "[MPC Core] curr vel: " << _curr_vel
  //           << ", curr ang vel: " << _curr_angvel << std::endl;
  //
  // return {_curr_vel, _curr_angvel};
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

std::vector<Eigen::VectorXd> MPCCore::get_horizon() const {
  std::vector<Eigen::VectorXd> ret;
  if (_mpc_input_type == MPCType::kUnicycle) {
    MPCC* _mpc_unicycle = dynamic_cast<MPCC*>(_mpc.get());
    ret.reserve(_mpc_unicycle->mpc_x.size());
    if (_mpc_unicycle->mpc_x.size() == 0)
      return ret;

    double t = 0;
    for (int i = 0; i < _mpc_unicycle->mpc_x.size() - 1; ++i) {
      ret.emplace_back(7);
      ret.back() << t, _mpc_unicycle->mpc_x[i], _mpc_unicycle->mpc_y[i],
          _mpc_unicycle->mpc_theta[i], _mpc_unicycle->mpc_linvels[i],
          _mpc_unicycle->mpc_linaccs[i], _mpc_unicycle->mpc_s[i];
      t += _dt;
    }
  } else if (_mpc_input_type == MPCType::kDoubleIntegrator) {
    DIMPCC* _mpc_double_integrator = dynamic_cast<DIMPCC*>(_mpc.get());
    ret.reserve(_mpc_double_integrator->mpc_x.size());
    if (_mpc_double_integrator->mpc_x.size() == 0)
      return ret;
    double t = 0;
    for (int i = 0; i < _mpc_double_integrator->mpc_x.size() - 1; ++i) {
      ret.emplace_back(9);
      ret.back() << t, _mpc_double_integrator->mpc_x[i],
          _mpc_double_integrator->mpc_y[i], _mpc_double_integrator->mpc_vx[i],
          _mpc_double_integrator->mpc_vy[i], _mpc_double_integrator->mpc_s[i],
          _mpc_double_integrator->mpc_s_dot[i],
          _mpc_double_integrator->mpc_ax[i], _mpc_double_integrator->mpc_ay[i];
      t += _dt;
    }
  }

  return ret;
}

const std::map<std::string, double>& MPCCore::get_params() const {
  return _params;
}

const std::array<double, 2> MPCCore::get_mpc_command() const {
  return _mpc->get_command();
}

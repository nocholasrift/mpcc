#include <mpcc/logger.h>
#include <mpcc/utils.h>
#include <std_msgs/Float64.h>

#include <Eigen/Core>
#include <string>

#include <casadi_double_integrator_mpcc_internals.h>
#include <casadi_unicycle_model_mpcc_internals.h>


namespace logger {

double get_max_width(const std::array<Eigen::VectorXd, 2>& tubes, double length,
                     unsigned int n_samples = 100) {
  double champ       = -1e6;
  Eigen::VectorXd ss = Eigen::VectorXd::LinSpaced(n_samples, 0, length);

  for (size_t i = 0; i < ss.size(); ++i) {
    double d_abv = utils::eval_traj(tubes[0], ss[i]);
    double d_blw = utils::eval_traj(tubes[1], ss[i]);
    double width = d_abv - d_blw;

    if (width > champ) {
      champ = width;
    }
  }

  return champ;
}

RLLogger::RLLogger(ros::NodeHandle& nh,
                   const std::unordered_map<std::string, double>& params,
                   bool is_logging) {

  _nh            = nh;
  _min_alpha     = 0.01;
  _max_alpha     = 10.0;
  _min_alpha_dot = -1.0;
  _max_alpha_dot = 1.0;
  _max_obs_dist  = 1.0;

  load_params(params);

  _is_logging = is_logging;

  _alpha_pub_abv = _nh.advertise<std_msgs::Float64>("/cbf_alpha_abv", 100);
  _alpha_pub_blw = _nh.advertise<std_msgs::Float64>("/cbf_alpha_blw", 100);

  _sac_srv = nh.serviceClient<mpcc::QuerySAC>("/query_sac");

  _count         = 0;
  _is_done       = false;
  _is_first_iter = true;
  _is_colliding  = false;

  _exceeded_bounds = 0;

  _alpha_dot_abv = 0.;
  _alpha_dot_blw = 0.;
}

RLLogger::~RLLogger() {}

void RLLogger::load_params(
    const std::unordered_map<std::string, double>& params) {
  _min_alpha = params.find("MIN_ALPHA") != params.end() ? params.at("MIN_ALPHA")
                                                        : _min_alpha;

  _max_alpha = params.find("MAX_ALPHA") != params.end() ? params.at("MAX_ALPHA")
                                                        : _max_alpha;

  _min_alpha_dot = params.find("MIN_ALPHA_DOT") != params.end()
                       ? params.at("MIN_ALPHA_DOT")
                       : _min_alpha_dot;

  _max_alpha_dot = params.find("MAX_ALPHA_DOT") != params.end()
                       ? params.at("MAX_ALPHA_DOT")
                       : _max_alpha_dot;

  _min_h_val = params.find("MIN_H_VAL") != params.end() ? params.at("MIN_H_VAL")
                                                        : _min_h_val;

  _max_h_val = params.find("MAX_H_VAL") != params.end() ? params.at("MAX_H_VAL")
                                                        : _max_h_val;

  _max_obs_dist = params.find("MAX_OBS_DIST") != params.end()
                      ? params.at("MAX_OBS_DIST")
                      : _max_obs_dist;

  _task_id = params.find("TASK_ID") != params.end() ? params.at("TASK_ID") : -1;

  _num_samples = params.find("NUM_SAMPLES") != params.end()
                     ? params.at("NUM_SAMPLES")
                     : 1e6;

  _max_path_length = params.find("MAX_PATH_LENGTH") != params.end()
                         ? params.at("MAX_PATH_LENGTH")
                         : 1e6;

  _params = params;
}

bool RLLogger::request_alpha(mpcc::MPCCore& mpc_core) {

  // TODO: Modify the data handling to eventually remove the magic numbers...
  mpcc::QuerySAC req;

  fill_state(mpc_core, req.request.state);

  if (!_sac_srv.call(req)) {
    ROS_ERROR("Failed to call service query_sac");
    return false;
  }

  if (!req.response.success) {
    ROS_ERROR("SAC service failed");
    return false;
  }

  // received action is between -1 and 1, scale to min/max alpha_dot
  auto scale = [](double val, double min_val, double max_val) {
    if (fabs(min_val - max_val) < 1e-8) {
      std::cerr << "[Logger] Warning: min (" << min_val << ") and max ("
                << max_val
                << ") are too close for proper "
                   "normalization!"
                << std::endl;
      return 0.;
    }

    if (val < min_val) {
      std::cerr << "[Logger] Warning: value " << val << " is less than min "
                << min_val << "!" << std::endl;
      val = min_val;
    }
    return min_val + (val + 1) * 0.5 * (max_val - min_val);
  };

  _alpha_dot_abv =
      scale(req.response.alpha_dot[0], _min_alpha_dot, _max_alpha_dot);
  _alpha_dot_blw =
      scale(req.response.alpha_dot[1], _min_alpha_dot, _max_alpha_dot);

  // integrate alpha_dot into CBF_ALPHA
  // clip alpha to ensure it's within bounds
  std::map<std::string, double> mpc_params = mpc_core.get_params();
  double dt                                = mpc_params.at("DT");

  double alpha_abv = mpc_params["CBF_ALPHA_ABV"] + _alpha_dot_abv * dt;
  double alpha_blw = mpc_params["CBF_ALPHA_BLW"] + _alpha_dot_blw * dt;

  if (alpha_abv < _min_alpha || alpha_abv > _max_alpha)
    _exceeded_bounds++;
  if (alpha_blw < _min_alpha || alpha_blw > _max_alpha)
    _exceeded_bounds++;

  alpha_abv = std::max(_min_alpha, std::min(_max_alpha, alpha_abv));
  alpha_blw = std::max(_min_alpha, std::min(_max_alpha, alpha_blw));

  std_msgs::Float64 alpha_msg;
  alpha_msg.data = alpha_abv;
  _alpha_pub_abv.publish(alpha_msg);

  std_msgs::Float64 alpha_msg_blw;
  alpha_msg_blw.data = alpha_blw;
  _alpha_pub_blw.publish(alpha_msg_blw);

  mpc_params["CBF_ALPHA_ABV"] = alpha_abv;
  mpc_params["CBF_ALPHA_BLW"] = alpha_blw;
  mpc_core.load_params(mpc_params);

  return true;
}

void RLLogger::fill_state(const mpcc::MPCCore& mpc_core, mpcc::RLState& state) {
  state.state.resize(6);
  state.state.emplace_back(0.);
  state.state.emplace_back(0.);
  state.state.emplace_back(0.);
  state.state.emplace_back(0.);
  state.state.emplace_back(0.);
  state.state.emplace_back(0.);
}

double normalize(double val, double min, double max) {
  if (fabs(min - max) < 1e-8) {
    std::cerr << "[Logger] Warning: min (" << min << ") and max (" << max
              << ") are too close for proper "
                 "normalization!"
              << std::endl;
    return 0.;
  }

  if (val < min) {
    std::cerr << "[Logger] Warning: value " << val << " is less than min "
              << min << "!" << std::endl;
    val = min;
  }

  return (val - min) / (max - min);
}

}  // namespace logger

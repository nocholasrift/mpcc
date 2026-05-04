#include <mpcc/common/utils.h>
#include <mpcc_ros1/logger.h>
#include <std_msgs/Float64.h>
#include "mpcc/common/mpcc_core.h"

#include <Eigen/Core>
#include <string>

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
                   std::shared_ptr<NodeConfig> node_cfg,
                   std::shared_ptr<mpcc::MPCConfig> mpc_cfg,
                   bool is_logging) {

  _nh            = nh;
  _min_alpha     = 0.01;
  _max_alpha     = 10.0;
  _min_alpha_dot = -1.0;
  _max_alpha_dot = 1.0;
  _max_obs_dist  = 1.0;
  _mpc_steps     = 0;

  load_params(node_cfg, mpc_cfg);

  _is_logging = is_logging;

  _alpha_pub_abv = _nh.advertise<std_msgs::Float64>("/cbf_alpha_abv", 100);
  _alpha_pub_blw = _nh.advertise<std_msgs::Float64>("/cbf_alpha_blw", 100);

  _sac_srv = nh.serviceClient<mpcc::QuerySAC>("/query_sac");

  _count         = 0;
  _is_done       = false;
  _is_first_iter = true;
  _is_colliding  = false;

  _alpha_dot_abv = 0.;
  _alpha_dot_blw = 0.;
}

RLLogger::~RLLogger() {}

void RLLogger::load_params(std::shared_ptr<NodeConfig> node_cfg,
                           std::shared_ptr<mpcc::MPCConfig> mpc_cfg) {
  _node_cfg = node_cfg;
  _mpc_cfg  = mpc_cfg;
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

  // copying instead of using const auto& becuase we will modify this
  // map...
  double dt       = _mpc_cfg->dt;

  // double alpha_abv = mpc_params["CBF_ALPHA_ABV"] + _alpha_dot_abv * dt;
  // double alpha_blw = mpc_params["CBF_ALPHA_BLW"] + _alpha_dot_blw * dt;
  double alpha_abv = _mpc_cfg->cbf.alpha_abv + _alpha_dot_abv;
  double alpha_blw = _mpc_cfg->cbf.alpha_blw + _alpha_dot_blw;

  ROS_WARN("ALPHA_ABV: %.2f", alpha_abv);
  ROS_WARN("ALPHA_BLW: %.2f", alpha_blw);

  // alpha_abv = std::max(_min_alpha, std::min(_max_alpha, alpha_abv));
  // alpha_blw = std::max(_min_alpha, std::min(_max_alpha, alpha_blw));

  _mpc_cfg->cbf.alpha_abv = alpha_abv;
  _mpc_cfg->cbf.alpha_blw = alpha_blw;

  std_msgs::Float64 alpha_msg;
  alpha_msg.data = alpha_abv;
  _alpha_pub_abv.publish(alpha_msg);

  std_msgs::Float64 alpha_msg_blw;
  alpha_msg_blw.data = alpha_blw;
  _alpha_pub_blw.publish(alpha_msg_blw);

  return true;
}

void RLLogger::fill_state(const mpcc::MPCCore& mpc_core, mpcc::RLState& state) {

  int N       = 3;
  double step = (_mpc_steps) / (N - 1);
  state.state.reserve(4 * N + 3);

  for (size_t i = 0; i < N; ++i) {
    size_t idx               = static_cast<size_t>(i * step);
    Eigen::VectorXd cbf_data = mpc_core.get_cbf_data(idx);
    state.state.emplace_back(cbf_data(0));
    state.state.emplace_back(cbf_data(1));
    state.state.emplace_back(cbf_data(2));
    state.state.emplace_back(cbf_data(3));
  }

  state.state.emplace_back(mpc_core.get_state().tail(1)[0]);
  state.state.emplace_back(_mpc_cfg->cbf.alpha_abv);
  state.state.emplace_back(_mpc_cfg->cbf.alpha_blw);
  /*state.state.emplace_back(mpc_core.get_params().at("CBF_ALPHA_ABV"));*/
  /*state.state.emplace_back(mpc_core.get_params().at("CBF_ALPHA_BLW"));*/
}

}  // namespace logger

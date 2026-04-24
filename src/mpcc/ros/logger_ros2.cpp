#include <mpcc/common/utils.h>
#include <mpcc/ros/logger.h>

#include <std_msgs/msg/float64.hpp>

#include "mpcc/common/mpcc_core.h"

#include <Eigen/Core>
#include <chrono>
#include <string>

namespace logger {

// -------------------------
// Constructor
// -------------------------
RLLogger::RLLogger(rclcpp::Node::SharedPtr node,
                   const std::unordered_map<std::string, double>& params,
                   bool is_logging)
    : node_(node) {

  min_alpha_     = 0.01;
  max_alpha_     = 10.0;
  min_alpha_dot_ = -1.0;
  max_alpha_dot_ = 1.0;
  max_obs_dist_  = 1.0;
  mpc_steps_     = 0;

  load_params(params);

  is_logging_ = is_logging;

  alpha_pub_abv_ =
      node_->create_publisher<std_msgs::msg::Float64>("/cbf_alpha_abv", 10);
  alpha_pub_blw_ =
      node_->create_publisher<std_msgs::msg::Float64>("/cbf_alpha_blw", 10);

  sac_client_ = node_->create_client<mpcc::srv::QuerySAC>("/query_sac");

  count_         = 0;
  is_done_       = false;
  is_first_iter_ = true;
  is_colliding_  = false;

  alpha_dot_abv_ = 0.0;
  alpha_dot_blw_ = 0.0;
}

RLLogger::~RLLogger() {}

// -------------------------
// Params
// -------------------------
void RLLogger::load_params(
    const std::unordered_map<std::string, double>& params) {

  mpc_steps_ = params.count("STEPS") ? params.at("STEPS") : mpc_steps_;

  min_alpha_ = params.count("MIN_ALPHA") ? params.at("MIN_ALPHA") : min_alpha_;

  max_alpha_ = params.count("MAX_ALPHA") ? params.at("MAX_ALPHA") : max_alpha_;

  min_alpha_dot_ = params.count("MIN_ALPHA_DOT") ? params.at("MIN_ALPHA_DOT")
                                                 : min_alpha_dot_;

  max_alpha_dot_ = params.count("MAX_ALPHA_DOT") ? params.at("MAX_ALPHA_DOT")
                                                 : max_alpha_dot_;

  min_h_val_ = params.count("MIN_H_VAL") ? params.at("MIN_H_VAL") : min_h_val_;

  max_h_val_ = params.count("MAX_H_VAL") ? params.at("MAX_H_VAL") : max_h_val_;

  max_obs_dist_ =
      params.count("MAX_OBS_DIST") ? params.at("MAX_OBS_DIST") : max_obs_dist_;

  task_id_ = params.count("TASK_ID") ? params.at("TASK_ID") : -1;

  num_samples_ = params.count("NUM_SAMPLES") ? params.at("NUM_SAMPLES") : 1e6;

  max_path_length_ =
      params.count("MAX_PATH_LENGTH") ? params.at("MAX_PATH_LENGTH") : 1e6;

  params_ = params;
}

// -------------------------
// Service Call (ROS2 version)
// -------------------------
bool RLLogger::request_alpha(mpcc::MPCCore& mpc_core) {

  if (!sac_client_->wait_for_service(std::chrono::milliseconds(100))) {
    RCLCPP_ERROR(node_->get_logger(), "Service /query_sac not available");
    return false;
  }

  auto req = std::make_shared<mpcc::srv::QuerySAC::Request>();

  fill_state(mpc_core, req->state);

  auto future = sac_client_->async_send_request(req);

  // --- BLOCKING WAIT (closest to ROS1 behavior) ---
  if (rclcpp::spin_until_future_complete(node_, future) !=
      rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(), "Service call failed");
    return false;
  }

  auto resp = future.get();

  if (!resp->success) {
    RCLCPP_ERROR(node_->get_logger(), "SAC service returned failure");
    return false;
  }

  // scaling
  auto scale = [](double val, double min_val, double max_val) {
    if (fabs(min_val - max_val) < 1e-8)
      return 0.0;

    val = std::max(min_val, val);
    return min_val + (val + 1.0) * 0.5 * (max_val - min_val);
  };

  alpha_dot_abv_ = scale(resp->alpha_dot[0], min_alpha_dot_, max_alpha_dot_);
  alpha_dot_blw_ = scale(resp->alpha_dot[1], min_alpha_dot_, max_alpha_dot_);

  auto mpc_params = mpc_core.get_params();
  double dt       = mpc_params.at("DT");

  double alpha_abv = mpc_params["CBF_ALPHA_ABV"] + alpha_dot_abv_;
  double alpha_blw = mpc_params["CBF_ALPHA_BLW"] + alpha_dot_blw_;

  alpha_abv = std::clamp(alpha_abv, min_alpha_, max_alpha_);
  alpha_blw = std::clamp(alpha_blw, min_alpha_, max_alpha_);

  std_msgs::msg::Float64 msg;
  msg.data = alpha_abv;
  alpha_pub_abv_->publish(msg);

  msg.data = alpha_blw;
  alpha_pub_blw_->publish(msg);

  mpc_params["CBF_ALPHA_ABV"] = alpha_abv;
  mpc_params["CBF_ALPHA_BLW"] = alpha_blw;

  mpc_core.load_params(mpc_params);

  return true;
}

// -------------------------
// State packing
// -------------------------
void RLLogger::fill_state(const mpcc::MPCCore& mpc_core,
                          mpcc::msg::RLState& state) {

  int N       = 3;
  double step = (mpc_steps_) / (N - 1);

  state.state.reserve(4 * N + 3);

  for (int i = 0; i < N; ++i) {
    size_t idx = static_cast<size_t>(i * step);

    Eigen::VectorXd cbf_data = mpc_core.get_cbf_data(idx);

    state.state.emplace_back(cbf_data(0));
    state.state.emplace_back(cbf_data(1));
    state.state.emplace_back(cbf_data(2));
    state.state.emplace_back(cbf_data(3));
  }

  const auto& params = mpc_core.get_params();

  state.state.emplace_back(mpc_core.get_state().tail(1)[0]);
  state.state.emplace_back(params.at("CBF_ALPHA_ABV"));
  state.state.emplace_back(params.at("CBF_ALPHA_BLW"));
}

}  // namespace logger

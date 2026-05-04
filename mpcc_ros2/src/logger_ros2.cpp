#include <mpcc/common/utils.h>
#include <mpcc_ros2/logger_ros2.h>

#include <mpcc/common/mpcc_core.h>
#include <std_msgs/msg/float64.hpp>

#include <Eigen/Core>
#include <algorithm>  // for std::clamp
#include <chrono>
#include <string>

namespace logger {

// -------------------------
// Constructor
// -------------------------
RLLogger::RLLogger(rclcpp::Node::SharedPtr node,
                   std::shared_ptr<NodeConfig> node_cfg,
                   std::shared_ptr<mpcc::MPCConfig> mpc_cfg, bool is_logging)
    : node_(node), is_logging_(is_logging) {

  load_params(node_cfg, mpc_cfg);
  // Default Values
  min_alpha_     = 0.01;
  max_alpha_     = 10.0;
  min_alpha_dot_ = -1.0;
  max_alpha_dot_ = 1.0;
  max_obs_dist_  = 1.0;
  mpc_steps_     = 0;

  // 1. Create a Reentrant Callback Group
  // This allows the executor to process the service response on a different thread
  // even while request_alpha() is "blocking" its current thread.
  cb_group_ =
      node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  // 2. Define Publisher Options to use the callback group
  // Without this, the publishers might be blocked by the same thread waiting for the service.
  rclcpp::PublisherOptions pub_options;
  pub_options.callback_group = cb_group_;

  alpha_pub_abv_ = node_->create_publisher<std_msgs::msg::Float64>(
      "/cbf_alpha_abv", 10, pub_options);
  alpha_pub_blw_ = node_->create_publisher<std_msgs::msg::Float64>(
      "/cbf_alpha_blw", 10, pub_options);

  // 3. Assign the client to the Reentrant group
  sac_client_ = node_->create_client<mpcc::srv::QuerySAC>(
      "/query_sac", rmw_qos_profile_services_default, cb_group_);

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
void RLLogger::load_params(std::shared_ptr<NodeConfig> node_cfg,
                           std::shared_ptr<mpcc::MPCConfig> mpc_cfg) {
  node_cfg_ = node_cfg;
  mpc_cfg_  = mpc_cfg;

  // mpc_steps_ = params.count("STEPS") ? params.at("STEPS") : mpc_steps_;
  // min_alpha_ = params.count("MIN_ALPHA") ? params.at("MIN_ALPHA") : min_alpha_;
  // max_alpha_ = params.count("MAX_ALPHA") ? params.at("MAX_ALPHA") : max_alpha_;
  // min_alpha_dot_ = params.count("MIN_ALPHA_DOT") ? params.at("MIN_ALPHA_DOT")
  //                                                : min_alpha_dot_;
  // max_alpha_dot_ = params.count("MAX_ALPHA_DOT") ? params.at("MAX_ALPHA_DOT")
  //                                                : max_alpha_dot_;
  // min_h_val_ = params.count("MIN_H_VAL") ? params.at("MIN_H_VAL") : min_h_val_;
  // max_h_val_ = params.count("MAX_H_VAL") ? params.at("MAX_H_VAL") : max_h_val_;
  // max_obs_dist_ =
  //     params.count("MAX_OBS_DIST") ? params.at("MAX_OBS_DIST") : max_obs_dist_;
  // task_id_     = params.count("TASK_ID") ? params.at("TASK_ID") : -1;
  // num_samples_ = params.count("NUM_SAMPLES") ? params.at("NUM_SAMPLES") : 1e6;
  // max_path_length_ =
  //     params.count("MAX_PATH_LENGTH") ? params.at("MAX_PATH_LENGTH") : 1e6;

  // params_ = params;
}

// -------------------------
// Service Call (ROS2 Reentrant Version)
// -------------------------
bool RLLogger::request_alpha(mpcc::MPCCore& mpc_core) {

  if (!sac_client_->wait_for_service(std::chrono::milliseconds(500))) {
    RCLCPP_ERROR(node_->get_logger(), "Service /query_sac not available");
    return false;
  }

  auto req = std::make_shared<mpcc::srv::QuerySAC::Request>();
  fill_state(mpc_core, req->state);

  RCLCPP_INFO(node_->get_logger(), "SENT REQUEST!!!!!");
  auto future = sac_client_->async_send_request(req);

  // 4. WAIT FOR FUTURE
  // Using future.wait_for instead of spin_until_future_complete.
  // The MultiThreadedExecutor in your main.cpp will handle the work.
  if (future.wait_for(std::chrono::milliseconds(500)) !=
      std::future_status::ready) {
    RCLCPP_ERROR(node_->get_logger(), "SAC service call timed out");
    return false;
  }

  RCLCPP_INFO(node_->get_logger(), "GOT RESPONSE!!!!!");
  auto resp = future.get();

  if (!resp->success) {
    RCLCPP_ERROR(node_->get_logger(), "SAC service returned failure");
    return false;
  }

  double min_alpha_dot = mpc_cfg_->cbf.min_alpha_dot;
  double max_alpha_dot = mpc_cfg_->cbf.max_alpha_dot;
  double min_alpha     = mpc_cfg_->cbf.min_alpha;
  double max_alpha     = mpc_cfg_->cbf.max_alpha;

  // Scaling lambda
  auto scale = [](double val, double min_val, double max_val) {
    if (std::abs(min_val - max_val) < 1e-8)
      return 0.0;
    return min_val + (val + 1.0) * 0.5 * (max_val - min_val);
  };

  alpha_dot_abv_ = scale(resp->alpha_dot[0], min_alpha_dot, max_alpha_dot);
  alpha_dot_blw_ = scale(resp->alpha_dot[1], min_alpha_dot, max_alpha_dot);

  // auto mpc_params = mpc_core.get_params();
  // double dt       = mpc_params.at("DT");

  // Calculate new alpha values
  // double alpha_abv = mpc_params["CBF_ALPHA_ABV"] + alpha_dot_abv_ * dt;
  // double alpha_blw = mpc_params["CBF_ALPHA_BLW"] + alpha_dot_blw_ * dt;
  double alpha_abv = mpc_cfg_->cbf.alpha_abv + alpha_dot_abv_;
  double alpha_blw = mpc_cfg_->cbf.alpha_blw + alpha_dot_blw_;

  // Constraint
  mpc_cfg_->cbf.alpha_abv = std::clamp(alpha_abv, min_alpha, max_alpha);
  mpc_cfg_->cbf.alpha_blw = std::clamp(alpha_blw, min_alpha, max_alpha);

  // 5. PUBLISH
  std_msgs::msg::Float64 msg;
  msg.data = alpha_abv;
  alpha_pub_abv_->publish(msg);

  msg.data = alpha_blw;
  alpha_pub_blw_->publish(msg);

  // 6. Update Core
  // mpc_params["CBF_ALPHA_ABV"] = alpha_abv;
  // mpc_params["CBF_ALPHA_BLW"] = alpha_blw;
  // mpc_core.load_params(mpc_params);

  return true;
}

// -------------------------
// State packing
// -------------------------
void RLLogger::fill_state(const mpcc::MPCCore& mpc_core,
                          mpcc::msg::RLState& state) {

  int N            = 3;
  double mpc_steps = mpc_cfg_->steps;
  double step      = (mpc_steps > 1) ? (double(mpc_steps) / (N - 1)) : 0;

  state.state.clear();
  state.state.reserve(4 * N + 3);

  for (int i = 0; i < N; ++i) {
    size_t idx               = static_cast<size_t>(i * step);
    Eigen::VectorXd cbf_data = mpc_core.get_cbf_data(idx);

    state.state.emplace_back(cbf_data(0));  // h_val
    state.state.emplace_back(cbf_data(1));  // h_dot
    state.state.emplace_back(cbf_data(2));  // dist to obs
    state.state.emplace_back(cbf_data(3));  // relative vel
  }

  // const auto& params = mpc_core.get_params();
  state.state.emplace_back(mpc_core.get_state().tail(1)[0]);  // Progress/Theta
  state.state.emplace_back(mpc_cfg_->cbf.alpha_abv);
  state.state.emplace_back(mpc_cfg_->cbf.alpha_blw);
  // state.state.emplace_back(params.at("CBF_ALPHA_ABV"));
  // state.state.emplace_back(params.at("CBF_ALPHA_BLW"));
}

}  // namespace logger

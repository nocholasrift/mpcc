#pragma once

#include <mpcc/msg/rl_state.hpp>
#include <mpcc/srv/query_sac.hpp>
#include <mpcc/srv/query_sac_di.hpp>

#include <mpcc/common/mpcc_core.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace logger {

class RLLogger {
 public:
  RLLogger(rclcpp::Node::SharedPtr node,
           const std::unordered_map<std::string, double>& params,
           bool is_logging);

  void load_params(const std::unordered_map<std::string, double>& params);

  ~RLLogger();

  void log_transition(const mpcc::MPCCore& mpc_core, double len_start);
  bool request_alpha(mpcc::MPCCore& mpc_core);

 private:
  void fill_state(const mpcc::MPCCore& mpc_core, mpcc::msg::RLState& state);

  // -------------------------
  // ROS2 core
  // -------------------------
  rclcpp::Node::SharedPtr node_;

  // Publishers
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr logging_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr alpha_pub_abv_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr alpha_pub_blw_;

  // Service client
  rclcpp::Client<mpcc::srv::QuerySAC>::SharedPtr sac_client_;

  // -------------------------
  // RL state
  // -------------------------
  mpcc::msg::RLState prev_rl_state_;
  mpcc::msg::RLState curr_rl_state_;

  unsigned int count_;

  std::string table_name_;
  std::string topic_name_;

  int task_id_;
  int mpc_steps_;
  int num_samples_;
  int max_path_length_;

  double min_alpha_;
  double max_alpha_;
  double min_alpha_dot_;
  double max_alpha_dot_;
  double min_h_val_;
  double max_h_val_;

  double max_obs_dist_;

  double alpha_dot_abv_;
  double alpha_dot_blw_;

  bool is_done_;
  bool is_logging_;
  bool is_colliding_;
  bool is_first_iter_;

  uint8_t exceeded_bounds_;

  std::unordered_map<std::string, double> params_;
};

// utility
double normalize(double val, double min, double max);

}  // namespace logger

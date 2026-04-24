#pragma once

// ROS2 Core
#include <rclcpp/rclcpp.hpp>

// ROS2 Message Headers (Note the .hpp extension and folder changes)
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/empty.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

// Project Specific
#include <mpcc/common/mpcc_core.h>
#include <mpcc/ros/logger.h>  // Ensure this is ported to ROS2 as well

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class MPCCROS : public rclcpp::Node {
 public:
  MPCCROS();  // Node options usually handled internally or via constructor
  virtual ~MPCCROS();

 private:
  // --- Core Logic ---
  void publishMPCTrajectory();
  void publishReference();
  void mpcc_ctrl_loop();  // ROS2 timers don't strictly require the event object

  // --- Callbacks ---
  void odomcb(const nav_msgs::msg::Odometry::SharedPtr msg);
  void mapcb(const nav_msgs::msg::Occupancy_grid::SharedPtr msg);
  void goalcb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void trajectorycb(
      const trajectory_msgs::msg::Joint_trajectory::SharedPtr msg);

  void publishVel();
  void visualizeTubes();
  void visualizeTraj();

  // --- Services ---
  bool toggleBackup(const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                    std::shared_ptr<std_srvs::srv::Empty::Response> res);

  bool can_execute();

  // --- ROS2 Members ---

  // Subscribers
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr
      _trajSub;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr
      _trajNoResetSub;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr
      _obsSub;  // Fixed from placeholder logic
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr _alphaSub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _odomSub;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr _collisionSub;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr _mapSub;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr _velPub;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr _trajPub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr _pathPub;
  // ... (Other publishers would follow the same pattern)
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _donePub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr _tubeVizPub;

  // Services & Clients
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr _backup_srv;
  // Note: For _sac_srv, you'll need the specific ROS2 Service Type
  // rclcpp::Client<YourSrvType>::SharedPtr _sac_srv;

  // Timers
  rclcpp::TimerBase::SharedPtr _timer;
  rclcpp::TimerBase::SharedPtr _velPubTimer;

  // --- Parameters & State ---
  std::unique_ptr<mpcc::MPCCore> _mpc_core;
  std::unique_ptr<logger::RLLogger> _logger;

  Eigen::VectorXd _odom;
  trajectory_msgs::msg::JointTrajectory _trajectory;
  nav2_costmap_2d::Costmap2DROS* _local_costmap;

  std::vector<Eigen::Vector3d> poses;
  std::vector<double> mpc_results;

  std::map<std::string, double> _mpc_params;

  double _mpc_steps, _w_vel, _w_angvel, _w_linvel, _w_angvel_d, _w_linvel_d,
      _w_etheta, _max_angvel, _max_linvel, _bound_value, _x_goal, _y_goal,
      _theta_goal, _tol, _max_linacc, _max_anga, _w_cte, _w_pos, _w_qc, _w_ql,
      _w_q_speed;

  double _cbf_alpha_abv, _cbf_alpha_blw, _cbf_colinear, _cbf_padding;

  double _prop_gain, _prop_angle_thresh;

  double _clf_gamma;
  double _w_ql_lyap;
  double _w_qc_lyap;

  double _min_alpha;
  double _max_alpha;
  double _min_alpha_dot;
  double _max_alpha_dot;
  double _min_h_val;
  double _max_h_val;

  double _ref_len;
  double _true_ref_len;
  double _mpc_ref_len_sz;
  double _max_tube_width;

  double _dt, _curr_vel, _curr_ang_vel, _vel_pub_freq;
  bool _is_init, _is_goal, _teleop, _use_vicon, _estop, _is_at_goal, _use_cbf,
      _use_dynamic_alpha, _reverse_mode;

  bool _is_traj_set{false};
  bool _is_logging;
  bool _is_eval;

  int _task_id;
  int _num_samples;
  int _tube_degree;
  int _tube_samples;
  int _max_path_length;
  int _mpc_ref_samples;

  Eigen::MatrixX4d _poly;
  geometry_msgs::Twist _vel_msg;

  Eigen::VectorXd _prev_rl_state;
  Eigen::VectorXd _curr_rl_state;

  std::string _frame_id;
  std::string _logging_table_name;
  std::string _logging_topic_name;

  mpcc::MPCType _mpc_input_type;

  Eigen::MatrixX4d _poly;
  geometry_msgs::msg::Twist _vel_msg;

  std::string _frame_id;
  std::thread timer_thread;
  static constexpr double kMAX_ALPHA = 100.0;
};

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
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/empty.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Project Specific
// #include <mpcc/common/mpcc_config.h>
#include <mpcc/common/mpcc_core.h>
#include <mpcc/node/node_impl.h>
#include <mpcc_ros2/logger_ros2.h>

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct ROS2Traits {
  using OdomMsg        = nav_msgs::msg::Odometry;
  using MapMsg         = nav_msgs::msg::OccupancyGrid;
  using TrajMsg        = trajectory_msgs::msg::JointTrajectory;
  using TrajPointMsg   = trajectory_msgs::msg::JointTrajectoryPoint;
  using PoseStampedMsg = geometry_msgs::msg::PoseStamped;
  using PathMsg        = nav_msgs::msg::Path;
  using TwistMsg       = geometry_msgs::msg::Twist;
  using Marker         = visualization_msgs::msg::Marker;
  using MarkerArray    = visualization_msgs::msg::MarkerArray;
  using Point          = geometry_msgs::msg::Point;
  using ColorRGBA      = std_msgs::msg::ColorRGBA;
  using PointStamped   = geometry_msgs::msg::PointStamped;
  using Header         = std_msgs::msg::Header;

  static double to_seconds(const builtin_interfaces::msg::Duration& d) {
    return rclcpp::Duration(d).seconds();
  }

  static rclcpp::Time now() { return rclcpp::Clock().now(); }

  static double elapsed(const rclcpp::Time& t) {
    return (rclcpp::Clock().now() - t).seconds();
  }

  static rclcpp::Duration duration(double dur) {
    return rclcpp::Duration(std::chrono::duration<double>(dur));
  }

  static Eigen::Vector3d odom_to_state(const nav_msgs::msg::Odometry& msg) {
    tf2::Quaternion q(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);

    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return {msg.pose.pose.position.x, msg.pose.pose.position.y, yaw};
  }

  static Header make_header(const std::string& frame_id) {
    Header h;
    h.frame_id = frame_id;
    h.stamp    = rclcpp::Clock().now();
    return h;
  }
};

struct ParamLoader {
  rclcpp::Node* node;

  double getd(const char* name, double def) const {
    node->declare_parameter(name, def);
    std::cout << name << ": " << node->get_parameter(name).as_double() << "\n";
    return node->get_parameter(name).as_double();
  }

  int geti(const char* name, int def) const {
    node->declare_parameter(name, def);
    std::cout << name << ": " << node->get_parameter(name).as_int() << "\n";
    return node->get_parameter(name).as_int();
  }

  bool getb(const char* name, bool def) const {
    node->declare_parameter(name, def);
    std::cout << name << ": " << node->get_parameter(name).as_bool() << "\n";
    return node->get_parameter(name).as_bool();
  }

  std::string gets(const char* name, const char* def) const {
    node->declare_parameter(name, std::string(def));
    std::cout << name << ": " << node->get_parameter(name).as_string() << "\n";
    return node->get_parameter(name).as_string();
  }
};

class MPCCROS : public rclcpp::Node,
                public mpcc_node::MPCCNodeImpl<MPCCROS, ROS2Traits> {
 public:
  MPCCROS();  // Node options usually handled internally or via constructor
  void init();

  mpcc::MPCConfig loadMPCConfig(ParamLoader& p_loader);
  mpcc_node::NodeConfig loadNodeConfig(ParamLoader& p_loader);

  virtual ~MPCCROS();

  template <typename... Args>
  void log_info(const char* fmt, Args&&... args) {
    RCLCPP_INFO(this->get_logger(), fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_warn(const char* fmt, Args&&... args) {
    RCLCPP_WARN(this->get_logger(), fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_error(const char* fmt, Args&&... args) {
    RCLCPP_ERROR(this->get_logger(), fmt, std::forward<Args>(args)...);
  }

  // TODO: Make this safer
  logger::RLLogger& logger() { return *_logger; }

  bool has_logger() { return _logger != nullptr; }

  void publish_ref_viz(const nav_msgs::msg::Path& msg);
  void publish_mpc_horizon_viz(const nav_msgs::msg::Path& msg);
  void publish_mpc_horizon_traj(
      const trajectory_msgs::msg::JointTrajectory& msg);
  void publish_tube_viz(const visualization_msgs::msg::MarkerArray& msg);

 private:
  // --- Core Logic ---
  void mpcc_ctrl_loop();  // ROS2 timers don't strictly require the event object

  // --- Callbacks ---
  void odomcb(const nav_msgs::msg::Odometry::SharedPtr msg);
  void mapcb(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void goalcb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void trajectorycb(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);

  void publishVel();

  // --- Services ---
  bool toggleBackup(const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                    std::shared_ptr<std_srvs::srv::Empty::Response> res);

  bool can_execute();

 private:
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
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pathPub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _trajPub;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr _startPub;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr
      _horizonPub;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _donePub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      _tubeVizPub;

  // Services & Clients
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr _backup_srv;
  // Note: For _sac_srv, you'll need the specific ROS2 Service Type
  // rclcpp::Client<YourSrvType>::SharedPtr _sac_srv;

  // Timers
  rclcpp::TimerBase::SharedPtr _timer;
  rclcpp::TimerBase::SharedPtr _velPubTimer;

  // --- Parameters & State ---
  // std::unique_ptr<mpcc::MPCCore> _mpc_core;
  std::unique_ptr<logger::RLLogger> _logger;

  // Eigen::VectorXd _odom;
  // trajectory_msgs::msg::JointTrajectory _trajectory;
  nav2_costmap_2d::Costmap2DROS* _local_costmap;

  // std::shared_ptr<NodeConfig> _node_cfg;
  // std::shared_ptr<mpcc::MPCConfig> _mpc_cfg;

  // std::vector<Eigen::Vector3d> poses;
  // std::vector<double> mpc_results;
  //
  // std::map<std::string, double> _mpc_params;
  //
  // double _w_vel, _w_angvel, _w_linvel, _w_angvel_d, _w_linvel_d, _w_etheta,
  //     _max_angvel, _max_linvel, _bound_value, _x_goal, _y_goal, _theta_goal,
  //     _tol, _max_linacc, _max_anga, _w_cte, _w_pos, _w_qc, _w_ql, _w_q_speed;
  //
  // double _cbf_alpha_abv, _cbf_alpha_blw, _cbf_colinear, _cbf_padding;
  //
  // double _prop_gain, _prop_gain_thresh;
  //
  // double _clf_gamma;
  // double _w_ql_lyap;
  // double _w_qc_lyap;
  //
  // double _min_alpha;
  // double _max_alpha;
  // double _min_alpha_dot;
  // double _max_alpha_dot;
  // double _min_h_val;
  // double _max_h_val;
  //
  // double _ref_len;
  // double _true_ref_len;
  // double _mpc_ref_len_sz;
  // double _max_tube_width;
  //
  // double _dt, _curr_vel, _curr_ang_vel, _vel_pub_freq;
  // bool _is_init, _is_goal, _teleop, _use_vicon, _estop, _is_at_goal, _use_cbf,
  //     _use_dynamic_alpha, _reverse_mode;
  //
  // bool _is_traj_set{false};
  // bool _is_eval;
  //
  // int _task_id;
  // int _mpc_steps;
  // int _num_samples;
  // int _tube_degree;
  // int _tube_samples;
  // int _max_path_length;
  // int _mpc_ref_samples;
  //
  // Eigen::VectorXd _prev_rl_state;
  // Eigen::VectorXd _curr_rl_state;
  //
  // std::string _logging_table_name;
  // std::string _logging_topic_name;
  //
  // mpcc::MPCType _mpc_input_type;
  //
  // Eigen::MatrixX4d _poly;
  // geometry_msgs::msg::Twist _vel_msg;
  //
  // std::string _frame_id;

  bool _reverse_mode{false};
  std::thread timer_thread;
  static constexpr double kMAX_ALPHA = 10.0;
};

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "mpcc/ros/mpcc_ros.hpp"

MPCCROS::MPCCROS() : Node("mpcc_controller_node") {
  _is_init      = false;
  _is_goal      = false;
  _reverse_mode = false;
  _is_traj_set  = false;

  _vel_msg.linear.x  = 0;
  _vel_msg.angular.z = 0;

  // --- Declare and Get Parameters ---
  this->declare_parameter("use_vicon", false);
  this->declare_parameter("vel_pub_freq", 20.0);
  this->declare_parameter("controller_frequency", 10.0);
  this->declare_parameter("mpc_steps", 10.0);
  this->declare_parameter("mpc_input_type",
                          static_cast<int>(mpcc::MPCType::kDoubleIntegrator));
  this->declare_parameter("frame_id", "odom");

  // Cost Weights
  this->declare_parameter("w_vel", 1.0);
  this->declare_parameter("w_angvel", 1.0);
  this->declare_parameter("w_linvel", 1.0);
  this->declare_parameter("w_lag_e", 50.0);
  this->declare_parameter("w_contour_e", 0.1);
  this->declare_parameter("w_speed", 0.3);

  // Constraints
  this->declare_parameter("max_angvel", 3.0);
  this->declare_parameter("max_linvel", 2.0);
  this->declare_parameter("max_linacc", 3.0);

  _use_vicon      = this->get_parameter("use_vicon").as_bool();
  _vel_pub_freq   = this->get_parameter("vel_pub_freq").as_double();
  double freq     = this->get_parameter("controller_frequency").as_double();
  _mpc_steps      = this->get_parameter("mpc_steps").as_double();
  _frame_id       = this->get_parameter("frame_id").as_string();
  _mpc_input_type = static_cast<mpcc::MPCType>(
      this->get_parameter("mpc_input_type").as_int());

  _dt = 1.0 / freq;

  // --- Fill MPC Params Map ---
  _mpc_params["DT"]        = _dt;
  _mpc_params["STEPS"]     = _mpc_steps;
  _mpc_params["W_V"]       = this->get_parameter("w_vel").as_double();
  _mpc_params["W_LAG"]     = this->get_parameter("w_lag_e").as_double();
  _mpc_params["W_CONTOUR"] = this->get_parameter("w_contour_e").as_double();
  _mpc_params["LINVEL"]    = this->get_parameter("max_linvel").as_double();
  _mpc_params["ANGVEL"]    = this->get_parameter("max_angvel").as_double();
  _mpc_params["DEBUG"]     = true;

  _mpc_core = std::make_unique<mpcc::MPCCore>(_mpc_input_type);
  RCLCPP_INFO(this->get_logger(), "Loading MPC core params...");
  _mpc_core->load_params(_mpc_params);

  // --- ROS2 Interfaces ---
  _mapSub = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/grid_map", 1, std::bind(&MPCCROS::mapcb, this, std::placeholders::_1));

  _odomSub = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", 1,
      std::bind(&MPCCROS::odomcb, this, std::placeholders::_1));

  _trajSub = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "/reference_trajectory", 1,
      std::bind(&MPCCROS::trajectorycb, this, std::placeholders::_1));

  _velPub  = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  _pathPub = this->create_publisher<nav_msgs::msg::Path>("/spline_path", 10);
  _trajPub = this->create_publisher<nav_msgs::msg::Path>("/mpc_prediction", 10);
  _startPub   = this->create_publisher<std_msgs::msg::Float64>("/progress", 10);
  _horizonPub = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/mpc_horizon", 10);

  // Service
  _backup_srv = this->create_service<std_srvs::srv::Empty>(
      "/mpc_backup", std::bind(&MPCCROS::toggleBackup, this,
                               std::placeholders::_1, std::placeholders::_2));

  // Control Loop Timer
  _timer = this->create_wall_timer(std::chrono::duration<double>(_dt),
                                   std::bind(&MPCCROS::mpcc_ctrl_loop, this));

  // High frequency vel pub thread
  timer_thread = std::thread(&MPCCROS::publishVel, this);
}

MPCCROS::~MPCCROS() {
  if (timer_thread.joinable())
    timer_thread.join();
}

void MPCCROS::mapcb(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  map_util::OccupancyGrid<int8_t>::MapConfig config;
  config.width      = msg->info.width;
  config.height     = msg->info.height;
  config.resolution = msg->info.resolution;
  config.origin = {msg->info.origin.position.x, msg->info.origin.position.y};
  config.occupied_values       = {100};
  config.no_information_values = {-1};

  _mpc_core->set_map<int8_t>(config, msg->data);
}

void MPCCROS::odomcb(const nav_msgs::msg::Odometry::SharedPtr msg) {
  tf2::Quaternion q;
  tf2::fromMsg(msg->pose.pose.orientation, q);
  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  _odom = Eigen::VectorXd(3);
  _odom << msg->pose.pose.position.x, msg->pose.pose.position.y, yaw;

  _mpc_core->set_odom(_odom);

  if (!_is_init) {
    _is_init = true;
    RCLCPP_INFO(this->get_logger(), "Tracker initialized");
  }
}

void MPCCROS::trajectorycb(
    const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Trajectory received!");
  _trajectory = *msg;

  if (msg->points.empty()) {
    _vel_msg.linear.x  = 0;
    _vel_msg.angular.z = 0;
    return;
  }

  int N = msg->points.size();
  Eigen::VectorXd ss(N), xs(N), ys(N);
  for (int i = 0; i < N; ++i) {
    xs[i] = msg->points[i].positions[0];
    ys[i] = msg->points[i].positions[1];
    ss[i] = rclcpp::Duration(msg->points[i].time_from_start).seconds();
  }

  _mpc_core->set_trajectory(xs, ys, ss);
  _is_traj_set = true;
  // visualizeTraj(); // Port this similarly if needed
}

void MPCCROS::mpcc_ctrl_loop() {
  if (!_is_init || !_is_traj_set)
    return;

  const auto& trajectory = _mpc_core->get_trajectory();
  double true_ref_len    = trajectory.get_arclen();
  double len_start       = trajectory.get_closest_s(_odom.head(2));

  auto start_msg = std_msgs::msg::Float64();
  start_msg.data = len_start / true_ref_len;
  _startPub->publish(start_msg);

  if (len_start > true_ref_len - 0.25) {
    RCLCPP_INFO(this->get_logger(), "Goal Reached.");
    _vel_msg.linear.x  = 0;
    _vel_msg.angular.z = 0;
    _trajectory.points.clear();
    return;
  }

  auto now = this->now();

  Eigen::VectorXd state(4);
  if (_mpc_input_type == mpcc::MPCType::kUnicycle)
    state << _odom(0), _odom(1), _odom(2), _vel_msg.linear.x;
  else
    state << _odom(0), _odom(1), _vel_msg.linear.x, _vel_msg.linear.y;

  std::array<double, 2> input = _mpc_core->solve(state);

  if (_mpc_input_type == mpcc::MPCType::kUnicycle) {
    _vel_msg.linear.x  = input[0];
    _vel_msg.angular.z = input[1];
  } else {
    _vel_msg.linear.x = input[0];
    _vel_msg.linear.y = input[1];
  }

  RCLCPP_DEBUG(this->get_logger(), "Solve time: %.3f",
               (this->now() - now).seconds());

  publishReference();
  publishMPCTrajectory();
}

void MPCCROS::publishVel() {
  rclcpp::Rate loop_rate(_vel_pub_freq);
  while (rclcpp::ok()) {
    if (!_trajectory.points.empty()) {
      _velPub->publish(_vel_msg);
    }
    loop_rate.sleep();
  }
}

bool MPCCROS::toggleBackup(
    const std::shared_ptr<std_srvs::srv::Empty::Request> req,
    std::shared_ptr<std_srvs::srv::Empty::Response> res) {
  (void)req;
  (void)res;
  _reverse_mode = !_reverse_mode;
  return true;
}

void MPCCROS::publishReference() {
  if (_trajectory.points.empty())
    return;

  nav_msgs::msg::Path msg;
  msg.header.stamp    = this->now();
  msg.header.frame_id = _frame_id;

  for (const auto& pt : _trajectory.points) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id    = _frame_id;
    pose.header.stamp       = this->now();
    pose.pose.position.x    = pt.positions[0];
    pose.pose.position.y    = pt.positions[1];
    pose.pose.orientation.w = 1.0;
    msg.poses.push_back(pose);
  }
  _pathPub->publish(msg);
}

void MPCCROS::publishMPCTrajectory() {
  mpcc::MPCCore::AnyHorizon horizon = _mpc_core->get_horizon();
  size_t horizon_steps =
      std::visit([](const auto& arg) { return arg.length; }, horizon);
  if (horizon_steps == 0)
    return;

  nav_msgs::msg::Path pathMsg;
  pathMsg.header.frame_id = _frame_id;
  pathMsg.header.stamp    = this->now();

  for (size_t step = 0; step < horizon_steps; ++step) {
    const Eigen::VectorXd& pos = std::visit(
        [&](const auto& arg) { return arg.get_pos_at_step(step); }, horizon);
    geometry_msgs::msg::PoseStamped tmp;
    tmp.header             = pathMsg.header;
    tmp.pose.position.x    = pos(0);
    tmp.pose.position.y    = pos(1);
    tmp.pose.orientation.w = 1.0;
    pathMsg.poses.push_back(tmp);
  }
  _trajPub->publish(pathMsg);
}

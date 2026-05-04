#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <nav_msgs/msg/path.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <mpcc_ros2/mpcc_ros2.h>

MPCCROS::MPCCROS() : Node("mpcc_controller_node") {
  _is_init      = false;
  _is_goal      = false;
  _reverse_mode = false;
  _is_traj_set  = false;

  _vel_msg.linear.x  = 0;
  _vel_msg.angular.z = 0;

  ParamLoader p_loader;
  p_loader.node = this;

  _node_cfg = std::make_shared<NodeConfig>(loadNodeConfig(p_loader));
  _mpc_cfg  = std::make_shared<mpcc::MPCConfig>(loadMPCConfig(p_loader));

  _mpc_core = std::make_unique<mpcc::MPCCore>(_mpc_cfg);
  RCLCPP_INFO(this->get_logger(), "Loading MPC core params...");

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
  _timer = this->create_wall_timer(std::chrono::duration<double>(_mpc_cfg->dt),
                                   std::bind(&MPCCROS::mpcc_ctrl_loop, this));

  // High frequency vel pub thread
  timer_thread = std::thread(&MPCCROS::publishVel, this);
}

MPCCROS::~MPCCROS() {
  if (timer_thread.joinable())
    timer_thread.join();
}

NodeConfig MPCCROS::loadNodeConfig(ParamLoader& p_loader) {
  NodeConfig conf;

  conf.use_vicon    = p_loader.getb("use_vicon", false);
  conf.is_eval      = p_loader.getb("is_eval", false);
  conf.vel_pub_freq = p_loader.getd("vel_pub_freq", 20.0);
  conf.frame_id     = p_loader.gets("frame_id", "odom");

  return conf;
}

mpcc::MPCConfig MPCCROS::loadMPCConfig(ParamLoader& p_loader) {
  mpcc::MPCConfig conf;

  // general MPC params
  conf.steps       = p_loader.geti("mpc_steps", 10);
  conf.dt          = 1.0 / p_loader.getd("controller_frequency", 10.0);
  conf.ref_samples = p_loader.geti("mpc_ref_samples", 10);
  conf.ref_length  = p_loader.getd("ref_length_size", 4.0);
  conf.input_type  = static_cast<mpcc::MPCType>(p_loader.geti(
      "mpc_input_type", static_cast<int>(mpcc::MPCType::kDoubleIntegrator)));

  // Cost weights
  conf.weights.w_vel       = p_loader.getd("w_vel", 1.0);
  conf.weights.w_angvel    = p_loader.getd("w_angvel", 1.0);
  conf.weights.w_linvel    = p_loader.getd("w_linvel", 1.0);
  conf.weights.w_angvel_d  = p_loader.getd("w_angvel_d", 1.0);
  conf.weights.w_linvel_d  = p_loader.getd("w_linvel_d", 0.5);
  conf.weights.w_etheta    = p_loader.getd("w_etheta", 0.5);
  conf.weights.w_cte       = p_loader.getd("w_cte", 1.0);
  conf.weights.w_lag_e     = p_loader.getd("w_lag_e", 50.0);
  conf.weights.w_contour_e = p_loader.getd("w_contour_e", 0.1);
  conf.weights.w_speed     = p_loader.getd("w_speed", 0.3);

  // Constraints
  conf.constraints.max_angvel  = p_loader.getd("max_angvel", 3.0);
  conf.constraints.max_linvel  = p_loader.getd("max_linvel", 2.0);
  conf.constraints.max_linacc  = p_loader.getd("max_linacc", 3.0);
  conf.constraints.max_angacc  = p_loader.getd("max_angacc", 2 * M_PI);
  conf.constraints.bound_value = p_loader.getd("bound_value", 1.0e19);

  // CBF
  conf.cbf.use_cbf       = p_loader.getb("use_cbf", false);
  conf.cbf.alpha_abv     = p_loader.getd("cbf_alpha_abv", 0.5);
  conf.cbf.alpha_blw     = p_loader.getd("cbf_alpha_blw", 0.5);
  conf.cbf.colinear      = p_loader.getd("cbf_colinear", 0.1);
  conf.cbf.padding       = p_loader.getd("cbf_padding", 0.1);
  conf.cbf.dynamic_alpha = p_loader.getb("dynamic_alpha", false);
  conf.cbf.min_alpha     = p_loader.getd("min_alpha", 0.1);
  conf.cbf.max_alpha     = p_loader.getd("max_alpha", 5.0);
  conf.cbf.min_alpha_dot = p_loader.getd("min_alpha_dot", -3.0);
  conf.cbf.max_alpha_dot = p_loader.getd("max_alpha_dot", 3.0);
  conf.cbf.min_h_val     = p_loader.getd("min_h_val", -100.0);
  conf.cbf.max_h_val     = p_loader.getd("max_h_val", 100.0);

  // CLF
  conf.clf.w_lag_e     = p_loader.getd("w_lyap_lag_e", 1.0);
  conf.clf.w_contour_e = p_loader.getd("w_lyap_contour_e", 1.0);
  conf.clf.gamma       = p_loader.getd("clf_gamma", 0.5);

  // Prop controller params
  conf.prop.gain        = p_loader.getd("prop_gain", 0.5);
  conf.prop.gain_thresh = p_loader.getd("prop_gain_thresh", 30. * M_PI / 180.);

  // Tube Generation (for CBF)
  conf.tube.poly_degree = p_loader.geti("tube_poly_degree", 6);
  conf.tube.num_samples = p_loader.geti("tube_num_samples", 50);
  conf.tube.max_width   = p_loader.getd("max_tube_width", 2.0);

  return conf;
}

void MPCCROS::init() {
  if (_mpc_cfg->cbf.use_cbf && _node_cfg->is_eval) {
    RCLCPP_WARN(this->get_logger(), "******************");
    RCLCPP_WARN(this->get_logger(), "LOGGING IS ENABLED");
    RCLCPP_WARN(this->get_logger(), "******************");

    // std::unordered_map<std::string, double> logger_params;
    // logger_params["MIN_ALPHA"]       = _min_alpha;
    // logger_params["MAX_ALPHA"]       = _max_alpha;
    // logger_params["MIN_ALPHA_DOT"]   = _min_alpha_dot;
    // logger_params["MAX_ALPHA_DOT"]   = _max_alpha_dot;
    // logger_params["MIN_H_VAL"]       = _min_h_val;
    // logger_params["MAX_H_VAL"]       = _max_h_val;
    // logger_params["MAX_OBS_DIST"]    = _max_tube_width;
    // logger_params["TASK_ID"]         = _task_id;
    // logger_params["NUM_SAMPLES"]     = _num_samples;
    // logger_params["MAX_PATH_LENGTH"] = _max_path_length;

    // rclcpp::Node actuall inherits from enable_shared_from_this...
    _logger = std::make_unique<logger::RLLogger>(
        this->shared_from_this(), _node_cfg, _mpc_cfg, _is_logging);

  } else if (!_mpc_cfg->cbf.use_cbf) {
    _cbf_alpha_abv = kMAX_ALPHA;
    _cbf_alpha_blw = kMAX_ALPHA;
  }
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
  if (_mpc_cfg->input_type == mpcc::MPCType::kUnicycle)
    state << _odom(0), _odom(1), _odom(2), _vel_msg.linear.x;
  else
    state << _odom(0), _odom(1), _vel_msg.linear.x, _vel_msg.linear.y;

  if (_logger) {
    _logger->request_alpha(*_mpc_core);
  }

  mpcc::MPCResult result = _mpc_core->solve(state);

  if (result.status != mpcc::SolverStatus::kSuccess ||
      result.status != mpcc::SolverStatus::kPresolve) {
    RCLCPP_INFO(this->get_logger(), "MPC solve was not successful!");
  }

  if (_mpc_cfg->input_type == mpcc::MPCType::kUnicycle) {
    _vel_msg.linear.x  = result.command[0];
    _vel_msg.angular.z = result.command[1];
  } else {
    _vel_msg.linear.x = result.command[0];
    _vel_msg.linear.y = result.command[1];
  }

  RCLCPP_DEBUG(this->get_logger(), "Solve time: %.3f",
               (this->now() - now).seconds());

  publishReference();
  publishMPCTrajectory();
}

void MPCCROS::publishVel() {
  rclcpp::Rate loop_rate(_node_cfg->vel_pub_freq);
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
  msg.header.frame_id = _node_cfg->frame_id;

  for (const auto& pt : _trajectory.points) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id    = _node_cfg->frame_id;
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
  pathMsg.header.frame_id = _node_cfg->frame_id;
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

  trajectory_msgs::msg::JointTrajectory traj;
  traj.header.stamp    = this->now();
  traj.header.frame_id = _node_cfg->frame_id;

  double dt = _mpc_cfg->dt;
  for (int step = 0; step < horizon_steps; ++step) {

    const Eigen::VectorXd& pos = std::visit(
        [&](const auto& arg) { return arg.get_pos_at_step(step); }, horizon);

    const Eigen::VectorXd& vel = std::visit(
        [&](const auto& arg) { return arg.get_vel_at_step(step); }, horizon);

    const Eigen::VectorXd& acc = std::visit(
        [&](const auto& arg) { return arg.get_vel_at_step(step); }, horizon);

    // manually compute jerk in x and y directions from acceleration
    /*double jerk_x = 0;*/
    /*double jerk_y = 0;*/
    Eigen::VectorXd jerk;
    if (step < horizon_steps - 1) {
      const Eigen::VectorXd& next_acc = std::visit(
          [&](const auto& arg) { return arg.get_vel_at_step(step + 1); },
          horizon);
      jerk = (next_acc - acc) / dt;
    } else {
      jerk = Eigen::VectorXd::Zero(vel.size());
    }

    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.time_from_start =
        rclcpp::Duration(std::chrono::duration<double>(step * dt));
    pt.positions     = {pos(0), pos(1), 0};
    pt.velocities    = {vel(0), vel(1), 0};
    pt.accelerations = {acc(0), acc(1), 0};
    pt.effort        = {jerk(0), jerk(1), 0};

    traj.points.push_back(pt);
  }

  _horizonPub->publish(traj);
}

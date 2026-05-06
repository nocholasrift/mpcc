#include "mpcc_ros1/mpcc_ros.h"
#include "mpcc/common/mpcc_base.h"
#include "mpcc/common/mpcc_core.h"

#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <tf/tf.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <algorithm>
#include <unordered_map>

#include "nav_msgs/OccupancyGrid.h"
#include "ros/console.h"

MPCCROS::MPCCROS(ros::NodeHandle& nh) : _nh("~") {
  _is_init      = false;
  _is_traj_set  = false;
  _reverse_mode = false;

  _vel_msg.linear.x  = 0;
  _vel_msg.angular.z = 0;

  ParamLoader p_loader;
  p_loader.nh = &_nh;

  _node_cfg = std::make_shared<mpcc_node::NodeConfig>(loadNodeConfig(p_loader));
  _mpc_cfg  = std::make_shared<mpcc::MPCConfig>(loadMPCConfig(p_loader));

  load_params(_mpc_cfg, _node_cfg);
  ROS_INFO("Loading MPC core params...");

  _mapSub  = nh.subscribe("/grid_map", 1, &MPCCROS::mapcb, this);
  _odomSub = nh.subscribe("/odometry/filtered", 1, &MPCCROS::odomcb, this);
  _trajSub =
      nh.subscribe("/reference_trajectory", 1, &MPCCROS::trajectorycb, this);

  _timer = nh.createTimer(ros::Duration(_mpc_cfg->dt), &MPCCROS::mpcc_ctrl_loop,
                          this);

  _startPub       = nh.advertise<std_msgs::Float64>("/progress", 10);
  _pathPub        = nh.advertise<nav_msgs::Path>("/spline_path", 10);
  _velPub         = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
  _trajPub        = nh.advertise<nav_msgs::Path>("/mpc_prediction", 10);
  _solveTimePub   = nh.advertise<std_msgs::Float64>("/mpc_solve_time", 0);
  _goalReachedPub = nh.advertise<std_msgs::Bool>("/mpc_goal_reached", 10);
  _pointPub       = nh.advertise<geometry_msgs::PointStamped>("traj_point", 0);
  _refVizPub  = nh.advertise<visualization_msgs::Marker>("/mpc_reference", 0);
  _tubeVizPub = nh.advertise<visualization_msgs::MarkerArray>("/tube_viz", 0);
  _horizonPub =
      nh.advertise<trajectory_msgs::JointTrajectory>("/mpc_horizon", 0);
  _refPub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>(
      "/current_reference", 10);

  timer_thread = std::thread(&MPCCROS::publishVel, this);

  _backup_srv =
      nh.advertiseService("/mpc_backup", &MPCCROS::toggleBackup, this);
}

MPCCROS::~MPCCROS() {
  if (timer_thread.joinable())
    timer_thread.join();
}

void MPCCROS::init() {
  if (_node_cfg->is_eval && _mpc_cfg->cbf.use_cbf) {
    ROS_WARN("******************");
    ROS_WARN("LOGGING IS ENABLED");
    ROS_WARN("******************");

    /*std::unordered_map<std::string, double> logger_params;*/
    /*logger_params["MIN_ALPHA"]       = _min_alpha;*/
    /*logger_params["MAX_ALPHA"]       = _max_alpha;*/
    /*logger_params["MIN_ALPHA_DOT"]   = _min_alpha_dot;*/
    /*logger_params["MAX_ALPHA_DOT"]   = _max_alpha_dot;*/
    /*logger_params["MIN_H_VAL"]       = _min_h_val;*/
    /*logger_params["MAX_H_VAL"]       = _max_h_val;*/
    /*logger_params["MAX_OBS_DIST"]    = _max_tube_width;*/
    /*logger_params["TASK_ID"]         = _task_id;*/
    /*logger_params["NUM_SAMPLES"]     = _num_samples;*/
    /*logger_params["MAX_PATH_LENGTH"] = _max_path_length;*/

    _logger = std::make_unique<logger::RLLogger>(_nh, _node_cfg, _mpc_cfg);

  } else if (!_mpc_cfg->cbf.use_cbf) {
    _mpc_cfg->cbf.alpha_abv = kMAX_ALPHA;
    _mpc_cfg->cbf.alpha_blw = kMAX_ALPHA;
  }
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

mpcc_node::NodeConfig MPCCROS::loadNodeConfig(ParamLoader& p_loader) {
  mpcc_node::NodeConfig conf;

  conf.use_vicon    = p_loader.getb("use_vicon", false);
  conf.is_eval      = p_loader.getb("is_eval", false);
  conf.vel_pub_freq = p_loader.getd("vel_pub_freq", 20.0);
  conf.frame_id     = p_loader.gets("frame_id", "odom");

  return conf;
}

bool MPCCROS::toggleBackup(std_srvs::Empty::Request& req,
                           std_srvs::Empty::Response& res) {
  _reverse_mode = !_reverse_mode;
  return true;
}

void MPCCROS::publishVel() {
  constexpr double pub_vel_loop_rate_hz = 50;
  const std::chrono::milliseconds pub_loop_period(
      static_cast<int>(1000.0 / pub_vel_loop_rate_hz));

  while (ros::ok()) {
    if (_trajectory.points.size() > 0)
      _velPub.publish(_vel_msg);

    // _velPub.publish(_vel_msg);

    std::this_thread::sleep_for(pub_loop_period);
  }
}

void MPCCROS::mapcb(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
  process_map(*msg);
  /*const nav_msgs::OccupancyGrid& grid = *msg;*/
  /*map_util::OccupancyGrid<int8_t>::MapConfig config;*/
  /*config.width      = grid.info.width;*/
  /*config.height     = grid.info.height;*/
  /*config.resolution = grid.info.resolution;*/
  /*config.origin = {grid.info.origin.position.x, grid.info.origin.position.y};*/
  /*config.occupied_values       = {100};*/
  /*config.no_information_values = {-1};*/
  /**/
  /*_mpc_core->set_map<int8_t>(config, grid.data);*/
}

void MPCCROS::odomcb(const nav_msgs::Odometry::ConstPtr& msg) {
  process_odom(*msg);
  /*tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,*/
  /*                 msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);*/
  /**/
  /*tf::Matrix3x3 m(q);*/
  /*double roll, pitch, yaw;*/
  /*m.getRPY(roll, pitch, yaw);*/
  /**/
  /*_odom = Eigen::VectorXd(3);*/
  /**/
  /*_odom(0) = msg->pose.pose.position.x;*/
  /*_odom(1) = msg->pose.pose.position.y;*/
  /*_odom(2) = yaw;*/
  /**/
  /*_mpc_core->set_odom(_odom);*/
  /**/
  /*if (!_is_init) {*/
  /*  _is_init = true;*/
  /*  ROS_INFO("tracker initialized");*/
  /*}*/
}

/**********************************************************************
 * Function: MPCCROS::trajectorycb(const
 *trajectory_msgs::JointTrajectory::ConstPtr& msg) Description: Callback for
 *trajectory message Parameters:
 * @param msg: trajectory_msgs::JointTrajectory::ConstPtr
 * Returns:
 * N/A
 * Notes:
 * This function sets the reference trajectory for the MPC controller
 * Since the ACADOS MPC requires a hard coded trajectory size, the
 * trajectory is extended if it is less than the required size
 **********************************************************************/
void MPCCROS::trajectorycb(
    const trajectory_msgs::JointTrajectory::ConstPtr& msg) {
  process_trajectory(*msg);

  /*ROS_INFO("Trajectory received!");*/
  /*_trajectory = *msg;*/
  /**/
  /*if (msg->points.size() == 0) {*/
  /*  ROS_WARN("Trajectory is empty, stopping!");*/
  /*  _vel_msg.linear.x  = 0;*/
  /*  _vel_msg.angular.z = 0;*/
  /*  return;*/
  /*}*/
  /**/
  /*int N = msg->points.size();*/
  /**/
  /*Eigen::VectorXd ss(N), xs(N), ys(N);*/
  /*for (int i = 0; i < N; ++i) {*/
  /*  xs[i] = msg->points[i].positions[0];*/
  /*  ys[i] = msg->points[i].positions[1];*/
  /*  ss[i] = msg->points[i].time_from_start.toSec();*/
  /*}*/
  /**/
  /*_mpc_core->set_trajectory(xs, ys, ss);*/
  /*_is_traj_set = true;*/
  /**/
  /*visualizeTraj();*/
  /**/
  /*ROS_INFO("**********************************************************");*/
  /*ROS_INFO("MPC received trajectory! Length: %.2f",*/
  /*         _mpc_core->get_trajectory().get_arclen());*/
  /*ROS_INFO("**********************************************************");*/
}

void MPCCROS::mpcc_ctrl_loop(const ros::TimerEvent& event) {
  control_loop();
}

void MPCCROS::publish_mpc_horizon_viz(const nav_msgs::Path& msg) {
  _trajPub.publish(msg);
}

void MPCCROS::publish_ref_viz(const nav_msgs::Path& msg) {
  _pathPub.publish(msg);
}

void MPCCROS::publish_mpc_horizon_traj(
    const trajectory_msgs::JointTrajectory& msg) {
  _horizonPub.publish(msg);
}

void MPCCROS::publish_tube_viz(const visualization_msgs::MarkerArray& msg) {
  _tubeVizPub.publish(msg);
}

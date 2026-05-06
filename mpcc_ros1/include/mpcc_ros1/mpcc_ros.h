#pragma once

#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <nav_msgs/Path.h>
#include <std_srvs/Empty.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <string>
#include <thread>

#include <mpcc/common/mpcc_core.h>
#include <mpcc/node/node_impl.h>
#include <mpcc_ros1/logger.h>

struct ROS1Traits {
  using OdomMsg        = nav_msgs::Odometry;
  using MapMsg         = nav_msgs::OccupancyGrid;
  using TrajMsg        = trajectory_msgs::JointTrajectory;
  using TrajPointMsg   = trajectory_msgs::JointTrajectoryPoint;
  using PoseStampedMsg = geometry_msgs::PoseStamped;
  using PathMsg        = nav_msgs::Path;
  using TwistMsg       = geometry_msgs::Twist;
  using Marker         = visualization_msgs::Marker;
  using MarkerArray    = visualization_msgs::MarkerArray;
  using Point          = geometry_msgs::Point;
  using ColorRGBA      = std_msgs::ColorRGBA;
  using PointStamped   = geometry_msgs::PointStamped;
  using Header         = std_msgs::Header;

  static double to_seconds(const ros::Duration& d) {
    return d.toSec();
  }

  static ros::Time now() { return ros::Time::now(); }

  static double elapsed(const ros::Time& t) {
    return (ros::Time::now() - t).toSec();
  }

  static ros::Duration duration(double dur) {
    return ros::Duration(dur);
  }

  static Eigen::Vector3d odom_to_state(const nav_msgs::Odometry& msg) {
    tf2::Quaternion q(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);

    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return {msg.pose.pose.position.x, msg.pose.pose.position.y, yaw};
  }

  static Header make_header(const std::string& frame_id) {
    Header h;
    h.frame_id = frame_id;
    h.stamp    = ros::Time::now();
    return h;
  }
};


struct ParamLoader {
  ros::NodeHandle* nh;

  double getd(const char* name, double def) const {
    double ret;
    nh->param(name, ret, def);
    return ret;
  }

  double geti(const char* name, int def) const {
    int ret;
    nh->param(name, ret, def);
    return ret;
  }

  double getb(const char* name, bool def) const {
    bool ret;
    nh->param(name, ret, def);
    return ret;
  }

  std::string gets(const char* name, const std::string& def) const {
    std::string ret;
    nh->param<std::string>(name, ret, def);
    return ret;
  }
};

class MPCCROS : public mpcc_node::MPCCNodeImpl<MPCCROS, ROS1Traits>{
 public:
  MPCCROS(ros::NodeHandle& nh);
  void init();

  mpcc::MPCConfig loadMPCConfig(ParamLoader& p_loader);
  mpcc_node::NodeConfig loadNodeConfig(ParamLoader& p_loader);

  ~MPCCROS();

  void publish_mpc_horizon_viz(const nav_msgs::Path& msg);

  void publish_ref_viz(const nav_msgs::Path& msg);

  void publish_mpc_horizon_traj(const trajectory_msgs::JointTrajectory& msg);

  void publish_tube_viz(const visualization_msgs::MarkerArray& msg);

  const logger::RLLogger& logger() const { return *_logger; }

  logger::RLLogger& logger() { return *_logger; }

  bool has_logger() { return _logger != nullptr; }

  template <typename... Args>
  void log_info(const char* fmt, Args&&... args) {
    ROS_INFO(fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_warn(const char* fmt, Args&&... args) {
    ROS_WARN(fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_error(const char* fmt, Args&&... args) {
    ROS_ERROR(fmt, std::forward<Args>(args)...);
  }


 private:
  /*void publishMPCTrajectory();*/
  /**********************************************************************
     * Function: MPCCROS::publishMPCTrajectory()
     * Description: Publishes the MPC prediction horizon
     * Parameters:
     * N/A
     * Returns:
     * N/A
     * Notes:
     * This function outputs the trajectory in JointTrajectory form so
     * trajectory generators can determine initial pos, vel, acc, etc.
     * for initial seeding.
     **********************************************************************/

  /*void publishReference();*/
  /**********************************************************************
     * Function: MPCCROS::publishReference()
     * Description: Publishes the reference trajectory
     * Parameters:
     * N/A
     * Returns:
     * N/A
     **********************************************************************/

  void mpcc_ctrl_loop(const ros::TimerEvent& event);
  /**********************************************************************
     * Function: MPCCROS::mpcc_ctrl_loop()
     * Description: Main control loop for MPC controller
     * Parameters:
     * Returns:
     * N/A
     * Notes:
     * Main control loop for the MPC, responsible for generating CBF tubes
     * and calling the MPC solver. Also sets up the virtual state s_dot
     **********************************************************************/

  /**********************************************************************
     * Callbacks for CBF alpha parameter, map, goal (not implemented
     * currently), odometry, and trajectory
     **********************************************************************/
  void odomcb(const nav_msgs::Odometry::ConstPtr& msg);
  void mapcb(const nav_msgs::OccupancyGrid::ConstPtr& msg);
  void trajectorycb(const trajectory_msgs::JointTrajectory::ConstPtr& msg);

  void publishVel();
  /**********************************************************************
     * Function: MPCCROS::publishVel()
     * Description: Publishes velocity command
     * Parameters:
     * Returns:
     * N/A
     * Notes:
     * Some vehicles require very high velocity publish rates (BD SPOT),
     * so the publishing of velocity is done in this separate thread at
     * a much higher frequency than the control loop.
     **********************************************************************/

  /*void visualizeTubes();*/
  /**********************************************************************
     * Function: MPCCROS::visualizeTubes()
     * Description: Visualizes the MPC tubes in rviz
     * Parameters:
     * Returns:
     * N/A
     * Notes:
     * Tubes are defined as a polynomial corridor separating the reference
     * trajectory from obstacles.
     **********************************************************************/

  /*void visualizeTraj();*/

  /**********************************************************************
     * Function: MPCCROS::toggleBackup()
     * Description: Toggles backup driving
     * Parameters:
     * Returns:
     * N/A
     **********************************************************************/
  bool toggleBackup(std_srvs::Empty::Request& req,
                    std_srvs::Empty::Response& res);


  /************************
     * Class variables
     ************************/

  std::unique_ptr<mpcc::MPCCore> _mpc_core;
  /**********************************************************************
     * In previous projects this has been the wrapper that can switch
     * between different MPC class implementations, but in this project only
     * one is currently implemented (the MPCC). Will eventually add more.
     **********************************************************************/
  std::unique_ptr<logger::RLLogger> _logger;

  ros::Subscriber _trajSub;
  ros::Subscriber _trajNoResetSub;
  ros::Subscriber _obsSub;
  ros::Subscriber _alphaSub;
  ros::Subscriber _odomSub;
  ros::Subscriber _collisionSub;
  ros::Subscriber _mapSub;

  ros::Publisher _velPub;
  ros::Publisher _trajPub;
  ros::Publisher _pathPub;
  ros::Publisher _pointPub;
  ros::Publisher _odomPub;
  ros::Publisher _refPub;
  ros::Publisher _goalReachedPub;
  ros::Publisher _horizonPub;
  ros::Publisher _solveTimePub;
  ros::Publisher _donePub;
  ros::Publisher _loggingPub;
  ros::Publisher _tubeVizPub;
  ros::Publisher _refVizPub;
  ros::Publisher _startPub;

  ros::ServiceServer _eStop_srv;
  ros::ServiceServer _mode_srv;
  ros::ServiceServer _backup_srv;

  ros::ServiceClient _sac_srv;

  ros::NodeHandle _nh;

  ros::Timer _timer, _velPubTimer;

  Eigen::VectorXd _odom;

  /*trajectory_msgs::JointTrajectory _trajectory;*/

  costmap_2d::Costmap2DROS* _local_costmap;

  /*std::shared_ptr<mpcc_node::NodeConfig> _node_cfg;*/
  /*std::shared_ptr<mpcc::MPCConfig> _mpc_cfg;*/

  /*std::vector<Eigen::Vector3d> poses;*/
  /*std::vector<double> mpc_results;*/
  /**/
  /*std::map<std::string, double> _mpc_params;*/
  /**/
  /*double _mpc_steps, _w_vel, _w_angvel, _w_linvel, _w_angvel_d, _w_linvel_d,*/
  /*    _w_etheta, _max_angvel, _max_linvel, _bound_value, _x_goal, _y_goal,*/
  /*    _theta_goal, _tol, _max_linacc, _max_anga, _w_cte, _w_pos, _w_qc, _w_ql,*/
  /*    _w_q_speed;*/
  /**/
  /*double _cbf_alpha_abv, _cbf_alpha_blw, _cbf_colinear, _cbf_padding;*/
  /**/
  /*double _prop_gain, _prop_angle_thresh;*/
  /**/
  /*double _clf_gamma;*/
  /*double _w_ql_lyap;*/
  /*double _w_qc_lyap;*/
  /**/
  /*double _min_alpha;*/
  /*double _max_alpha;*/
  /*double _min_alpha_dot;*/
  /*double _max_alpha_dot;*/
  /*double _min_h_val;*/
  /*double _max_h_val;*/
  /**/
  /*double _ref_len;*/
  /*double _true_ref_len;*/
  /*double _mpc_ref_len_sz;*/
  /*double _max_tube_width;*/
  /**/
  /*double _dt, _curr_vel, _curr_ang_vel, _vel_pub_freq;*/
  /*bool _is_init, _is_goal, _teleop, _use_vicon, _estop, _is_at_goal, _use_cbf,*/
  /*    _use_dynamic_alpha, _reverse_mode;*/
  /**/
  /*bool _is_traj_set{false};*/
  /*bool _is_logging;*/
  /*bool _is_eval;*/
  /**/
  /*int _task_id;*/
  /*int _num_samples;*/
  /*int _tube_degree;*/
  /*int _tube_samples;*/
  /*int _max_path_length;*/
  /*int _mpc_ref_samples;*/
  /**/
  /*Eigen::MatrixX4d _poly;*/
  /*geometry_msgs::Twist _vel_msg;*/
  /**/
  /*Eigen::VectorXd _prev_rl_state;*/
  /*Eigen::VectorXd _curr_rl_state;*/

  /*std::string _frame_id;*/
  /*std::string _logging_table_name;*/
  /*std::string _logging_topic_name;*/

  /*mpcc::MPCType _mpc_input_type;*/

  bool _reverse_mode{false};
  std::thread timer_thread;

  static constexpr double kMAX_ALPHA = 10.f;
};

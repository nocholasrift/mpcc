#ifndef MPCC_NODE_NODE_IMPL_H
#define MPCC_NODE_NODE_IMPL_H

#include <mpcc/common/mpcc_core.h>

namespace mpcc_node {

// definition
template <typename RosTraits>
class MPCCNodeImpl {
 public:
  using MapMsg         = typename RosTraits::MapMsg;
  using Float64Msg     = typename RosTraits::Float64Msg;
  using TrajMsg        = typename RosTraits::TrajMsg;
  using TrajPointMsg   = typename RosTraits::TrajPointMsg;
  using PathMsg        = typename RosTraits::PathMsg;
  using PoseStampedMsg = typename RosTraits::PoseStampedMsg;

  MPCCNodeImpl()  = default;
  ~MPCCNodeImpl() = default;

  void process_map(MapMsg msg) {
    map_util::OccupancyGrid<int8_t>::MapConfig config;
    config.width      = msg->info.width;
    config.height     = msg->info.height;
    config.resolution = msg->info.resolution;
    config.origin = {msg->info.origin.position.x, msg->info.origin.position.y};
    config.occupied_values       = {100};
    config.no_information_values = {-1};

    _mpc_core->set_map<int8_t>(config, msg->data);
  }

  bool set_trajectory(TrajMsg msg) {
    if (msg->points.empty()) {
      return false;
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
  }

  void control_loop() {
    if (!_is_init || !_is_traj_set) {
      return;
    }

    const auto& trajectory = _mpc_core->get_trajectory();
    double true_ref_len    = trajectory.get_arclen();
    double len_start       = trajectory.get_closst_s(_odom.head(2));

    Eigen::VectorXd state(4);
    if (_mpc_cfg->input_type == mpcc::MPCType::kUnicycle)
      state << _odom(0), _odom(1), _odom(2), _vel_msg.linear.x;
    else
      state << _odom(0), _odom(1), _vel_msg.linear.x, _vel_msg.linear.y;

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
  }

  void get_reference_msg(PathMsg nav_msg, PoseStampedMsg pose_msg) {
    if (_trajectory.points.empty()) {
      return;
    }

    PathMsg msg;
    for (const auto& pt : _trajectory.points) {
      PoseStampedMsg pose;
      pose.pose.position.x    = pt.positions[0];
      pose.pose.position.y    = pt.positions[1];
      pose.pose.orientation.w = 1.0;
      msg.poses.push_back(pose);
    }
  }

  void get_mpc_horizon_msg(PathMsg& path_msg, TrajMsg& traj_msg) {

    mpcc::MPCCore::AnyHorizon horizon = _mpc_core->get_horizon();
    size_t horizon_steps =
        std::visit([](const auto& arg) { return arg.length; }, horizon);
    if (horizon_steps == 0)
      return;

    PathMsg path_msg;
    // path_msg.header.frame_id = _node_cfg->frame_id;
    // path_msg.header.stamp    = this->now();

    for (size_t step = 0; step < horizon_steps; ++step) {
      const Eigen::VectorXd& pos = std::visit(
          [&](const auto& arg) { return arg.get_pos_at_step(step); }, horizon);
      geometry_msgs::msg::PoseStamped tmp;
      tmp.header             = path_msg.header;
      tmp.pose.position.x    = pos(0);
      tmp.pose.position.y    = pos(1);
      tmp.pose.orientation.w = 1.0;
      path_msg.poses.push_back(tmp);
    }
    // _trajPub->publish(path_msg);

    TrajMsg traj_msg;
    // traj.header.stamp    = this->now();
    // traj.header.frame_id = _node_cfg->frame_id;

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

      TrajPointMsg pt;
      pt.time_from_start =
          rclcpp::Duration(std::chrono::duration<double>(step * dt));
      pt.positions     = {pos(0), pos(1), 0};
      pt.velocities    = {vel(0), vel(1), 0};
      pt.accelerations = {acc(0), acc(1), 0};
      pt.effort        = {jerk(0), jerk(1), 0};

      traj_msg.points.push_back(pt);
    }
  }

  const mpcc::MPCCore& get_mpc_core() const { return *_mpc_core; }

 private:
  std::unique_ptr<mpcc::MPCCore> _mpc_core;

  bool _is_traj_set{false};
  bool _is_init{false};
  bool _is_traj_set{false};

  TrajMsg _trajectory;
}

}  // namespace mpcc_node
#endif

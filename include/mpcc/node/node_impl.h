#ifndef MPCC_NODE_NODE_IMPL_H
#define MPCC_NODE_NODE_IMPL_H

#include <mpcc/common/mpcc_core.h>

namespace mpcc_node {

struct NodeConfig {
  bool use_vicon       = false;
  bool is_eval         = false;
  double vel_pub_freq  = 20.0;
  std::string frame_id = "odom";
  // mpcc::MPCConfig mpc;
};

// definition
template <typename Derived, typename RosTraits>
class MPCCNodeImpl {
 public:
  using OdomMsg        = typename RosTraits::OdomMsg;
  using MapMsg         = typename RosTraits::MapMsg;
  using TrajMsg        = typename RosTraits::TrajMsg;
  using TwistMsg       = typename RosTraits::TwistMsg;
  using TrajPointMsg   = typename RosTraits::TrajPointMsg;
  using PathMsg        = typename RosTraits::PathMsg;
  using PoseStampedMsg = typename RosTraits::PoseStampedMsg;
  using Marker         = typename RosTraits::Marker;
  using MarkerArray    = typename RosTraits::MarkerArray;
  using Point          = typename RosTraits::Point;
  using ColorRGBA      = typename RosTraits::ColorRGBA;

  MPCCNodeImpl()  = default;
  ~MPCCNodeImpl() = default;

  void load_params(std::shared_ptr<mpcc::MPCConfig> mpc_cfg,
                   std::shared_ptr<NodeConfig> node_cfg) {
    _mpc_cfg  = mpc_cfg;
    _node_cfg = node_cfg;
    _mpc_core = std::make_unique<mpcc::MPCCore>(_mpc_cfg);
  }

  void process_map(const MapMsg& msg) {
    map_util::OccupancyGrid<int8_t>::MapConfig config;
    config.width      = msg.info.width;
    config.height     = msg.info.height;
    config.resolution = msg.info.resolution;
    config.origin = {msg.info.origin.position.x, msg.info.origin.position.y};
    config.occupied_values       = {100};
    config.no_information_values = {-1};

    _mpc_core->set_map<int8_t>(config, msg.data);
  }

  void process_odom(const OdomMsg& msg) {
    _odom = RosTraits::odom_to_state(msg);
    _mpc_core->set_odom(_odom);
    if (!_is_init) {
      _is_init = true;
      derived().log_info("tracker initialized");
    }
  }

  void process_trajectory(TrajMsg msg) {
    _trajectory = msg;
    if (msg.points.empty()) {
      derived().log_warn("Trajectory is empty, stopping!");
      _vel_msg.linear.x  = 0;
      _vel_msg.angular.z = 0;
      return;
    }

    int N = msg.points.size();
    Eigen::VectorXd ss(N), xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
      xs[i] = msg.points[i].positions[0];
      ys[i] = msg.points[i].positions[1];
      ss[i] = RosTraits::to_seconds(msg.points[i].time_from_start);
    }

    _mpc_core->set_trajectory(xs, ys, ss);
    _is_traj_set = true;

    derived().log_info("MPC received trajectory! Length: %.2f",
                       _mpc_core->get_trajectory().get_arclen());
  }

  void control_loop() {
    if (!_is_init || !_is_traj_set) {
      return;
    }

    const auto& trajectory = _mpc_core->get_trajectory();
    double true_ref_len    = trajectory.get_arclen();
    double len_start       = trajectory.get_closest_s(_odom.head(2));

    if (len_start > true_ref_len - 0.25) {
      derived().log_info("Reached end of traj %.2f / %.2f", len_start,
                         true_ref_len);
      _vel_msg.linear.x  = 0;
      _vel_msg.linear.y  = 0;
      _vel_msg.angular.z = 0;
      _trajectory.points.clear();
      return;
    }

    auto t_start          = RosTraits::now();
    Eigen::VectorXd state = build_state();

    if (derived().has_logger()) {
      derived().logger().request_alpha(*_mpc_core);
    }

    // derived().log_error("alpha_abv: %.2f", _mpc_cfg->cbf.alpha_abv);
    // derived().log_error("alpha_blw: %.2f", _mpc_cfg->cbf.alpha_blw);

    mpcc::MPCResult result = _mpc_core->solve(state);

    if (result.status != mpcc::SolverStatus::kSuccess &&
        result.status != mpcc::SolverStatus::kPresolve) {
      derived().log_error("MPC solve was not successful!");
    }

    apply_input(result);
    derived().log_warn("runtime: %.3f", RosTraits::elapsed(t_start));

    publish_reference();
    publish_mpc_horizon();
    visualize_tubes();
  }

  void publish_reference() {
    if (_trajectory.points.empty()) {
      return;
    }

    PathMsg msg;
    msg.header = RosTraits::make_header(_node_cfg->frame_id);
    msg.poses.reserve(_trajectory.points.size());
    for (const auto& pt : _trajectory.points) {
      PoseStampedMsg& pose    = msg.poses.emplace_back();
      pose.header             = msg.header;
      pose.pose.position.x    = pt.positions[0];
      pose.pose.position.y    = pt.positions[1];
      pose.pose.position.z    = 0;
      pose.pose.orientation.x = 0;
      pose.pose.orientation.y = 0;
      pose.pose.orientation.z = 0;
      pose.pose.orientation.w = 1;
    }

    derived().publish_ref_viz(msg);
  }

  void publish_mpc_horizon() {

    mpcc::MPCCore::AnyHorizon horizon = _mpc_core->get_horizon();
    size_t horizon_steps =
        std::visit([](const auto& arg) { return arg.length; }, horizon);
    if (horizon_steps == 0)
      return;

    PathMsg path_msg;
    path_msg.header = RosTraits::make_header(_node_cfg->frame_id);

    const mpcc::types::Trajectory& reference = _mpc_core->get_trajectory();
    double true_ref_len                      = reference.get_arclen();

    for (size_t step = 0; step < horizon_steps; ++step) {
      double s = std::visit(
          [&](const auto& arg) { return arg.get_arclen_at_step(step); },
          horizon);

      if (s > true_ref_len) {
        break;
      }

      const Eigen::VectorXd& pos = std::visit(
          [&](const auto& arg) { return arg.get_pos_at_step(step); }, horizon);

      PoseStampedMsg& pose    = path_msg.poses.emplace_back();
      pose.header             = path_msg.header;
      pose.pose.position.x    = pos(0);
      pose.pose.position.y    = pos(1);
      pose.pose.position.z    = 0.1;
      pose.pose.orientation.w = 1.0;
    }

    derived().publish_mpc_horizon_viz(path_msg);

    TrajMsg traj_msg;
    traj_msg.header = RosTraits::make_header(_node_cfg->frame_id);

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
      pt.time_from_start = RosTraits::duration(step * dt);
      pt.positions       = {pos(0), pos(1), 0};
      pt.velocities      = {vel(0), vel(1), 0};
      pt.accelerations   = {acc(0), acc(1), 0};
      pt.effort          = {jerk(0), jerk(1), 0};

      traj_msg.points.push_back(pt);
    }

    derived().publish_mpc_horizon_traj(traj_msg);
  }

  void visualize_tubes() {
    mpcc::types::Corridor corridor = _mpc_core->get_corridor(_odom.head(2));
    double horizon                 = corridor.get_trajectory().get_arclen();

    if (horizon < 0.05)
      return;

    Marker tubemsg_a;
    tubemsg_a.header             = RosTraits::make_header(_node_cfg->frame_id);
    tubemsg_a.ns                 = "tube_above";
    tubemsg_a.id                 = 87;
    tubemsg_a.action             = Marker::ADD;
    tubemsg_a.type               = Marker::LINE_STRIP;
    tubemsg_a.scale.x            = 0.075;
    tubemsg_a.pose.orientation.w = 1;

    Marker tubemsg_b = tubemsg_a;
    tubemsg_b.ns     = "tube_below";
    tubemsg_b.id     = 88;

    tubemsg_a.points.reserve(2 * (horizon / 0.05));
    tubemsg_b.points.reserve(2 * (horizon / 0.05));
    tubemsg_a.colors.reserve(2 * (horizon / 0.05));
    tubemsg_b.colors.reserve(2 * (horizon / 0.05));

    for (double s = 0; s < horizon; s += 0.05) {
      auto sample = corridor.get_at(s);

      Point pt_a;
      pt_a.x = sample.above(0);
      pt_a.y = sample.above(1);
      pt_a.z = 1.0;

      Point pt_b;
      pt_b.x = sample.below(0);
      pt_b.y = sample.below(1);
      pt_b.z = 1.0;

      ColorRGBA color_abv;
      color_abv.r = 192.0 / 255.0;
      color_abv.g = 0.0;
      color_abv.b = 0.0;
      color_abv.a = 1.0;

      ColorRGBA color_blw;
      color_blw.r = 251.0 / 255.0;
      color_blw.g = 133.0 / 255.0;
      color_blw.b = 0.0;
      color_blw.a = 1.0;

      tubemsg_a.points.push_back(pt_a);
      tubemsg_b.points.push_back(pt_b);
      tubemsg_a.colors.push_back(color_abv);
      tubemsg_b.colors.push_back(color_blw);
    }

    MarkerArray tube_ma;
    tube_ma.markers.push_back(std::move(tubemsg_a));
    tube_ma.markers.push_back(std::move(tubemsg_b));

    derived().publish_tube_viz(tube_ma);
  }

  const mpcc::MPCCore& get_mpc_core() const { return *_mpc_core; }

 protected:
  std::unique_ptr<mpcc::MPCCore> _mpc_core;

  bool _is_traj_set{false};
  bool _is_init{false};

  TrajMsg _trajectory;

  std::shared_ptr<NodeConfig> _node_cfg;
  std::shared_ptr<mpcc::MPCConfig> _mpc_cfg;

  Eigen::VectorXd _odom;
  TwistMsg _vel_msg;

 private:
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  Eigen::VectorXd build_state() const {
    Eigen::VectorXd state;
    if (_mpc_cfg->input_type == mpcc::MPCType::kUnicycle) {
      state.resize(4);
      state << _odom(0), _odom(1), _odom(2), _vel_msg.linear.x;
    } else if (_mpc_cfg->input_type == mpcc::MPCType::kDoubleIntegrator) {
      state.resize(4);
      state << _odom(0), _odom(1), _vel_msg.linear.x, _vel_msg.linear.y;
    } else {
      throw std::runtime_error(
          "Unknown MPC input type: " +
          std::to_string(static_cast<unsigned int>(_mpc_cfg->input_type)));
    }

    return state;
  }

  void apply_input(const mpcc::MPCResult& input) {
    if (_mpc_cfg->input_type == mpcc::MPCType::kUnicycle) {
      _vel_msg.linear.x  = input.command[0];
      _vel_msg.angular.z = input.command[1];
    } else if (_mpc_cfg->input_type == mpcc::MPCType::kDoubleIntegrator) {
      _vel_msg.linear.x = input.command[0];
      _vel_msg.linear.y = input.command[1];
    }
  }

 private:
};

}  // namespace mpcc_node
#endif

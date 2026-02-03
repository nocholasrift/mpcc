#pragma once

#include <mpcc/QuerySAC.h>
#include <mpcc/QuerySACDI.h>
#include <mpcc/RLState.h>
#include <mpcc/mpcc_core.h>

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <Eigen/Core>

#include <cstdint>

namespace logger {

class RLLogger {
 public:
  RLLogger(ros::NodeHandle& nh,
           const std::unordered_map<std::string, double>& params,
           bool is_logging);

  void load_params(const std::unordered_map<std::string, double>& params);

  ~RLLogger();

  void log_transition(const mpcc::MPCCore& mpc_core, double len_start);
  bool request_alpha(mpcc::MPCCore& mpc_core);

 private:
  void fill_state(const mpcc::MPCCore& mpc_core, mpcc::RLState& state);

  ros::NodeHandle _nh;

  ros::Publisher _done_pub;
  ros::Publisher _logging_pub;
  ros::Publisher _alpha_pub_abv;
  ros::Publisher _alpha_pub_blw;

  ros::ServiceClient _sac_srv;

  mpcc::RLState _prev_rl_state;
  mpcc::RLState _curr_rl_state;

  unsigned int _count;

  std::string _table_name;
  std::string _topic_name;

  int _task_id;
  int _num_samples;
  int _max_path_length;

  double _min_alpha;
  double _max_alpha;
  double _min_alpha_dot;
  double _max_alpha_dot;
  double _min_h_val;
  double _max_h_val;

  double _max_obs_dist;

  double _alpha_dot_abv;
  double _alpha_dot_blw;

  bool _is_done;
  bool _is_logging;
  bool _is_colliding;
  bool _is_first_iter;

  uint8_t _exceeded_bounds;

  std::unordered_map<std::string, double> _params;
};

double normalize(double val, double min, double max);

}  // namespace logger

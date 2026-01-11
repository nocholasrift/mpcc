#pragma once

#include <mpcc/types.h>
#include <mpcc/utils.h>

#include <Eigen/Core>
//
// acados
#include "acados_c/ocp_nlp_interface.h"

#include <cmath>
#include <map>

namespace mpcc {
using TrajectoryView = types::Trajectory::View;
using MPCHorizon     = types::MPCHorizon;

// Interface assumes the use of acados
class MPCBase {

 public:
  virtual ~MPCBase() = default;
  virtual void load_params(const std::map<std::string, double>& params) = 0;

  virtual std::array<double, 2> solve(const Eigen::VectorXd& state,
                                      const TrajectoryView& reference,
                                      bool is_reverse = false) = 0;

  void set_odom(const Eigen::VectorXd& odom) {
    _odom = odom;

    if (!_odom_init) {
      _odom_init = true;
      reset_horizon();
    }
  }
  virtual void reset_horizon() = 0;

  void set_tubes(const std::array<Eigen::VectorXd, 2>& tubes) {
    _tubes = tubes;
  }

  const std::array<Eigen::VectorXd, 2>& get_tubes() const { return _tubes; }

  virtual const Eigen::VectorXd& get_state() const                      = 0;
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits() const = 0;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits() const = 0;

  virtual MPCHorizon get_horizon() const = 0;

  virtual std::array<double, 2> get_command() const { return _cmd; }
  virtual Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control,
                                       bool is_abv) const = 0;

  const bool get_solver_status() const { return _solve_success; }

  double limit(double prev_val, double input, double max_rate,
               double dt) const {
    double ret = input;
    if (fabs(prev_val - input) / dt > max_rate) {
      if (input > prev_val)
        ret = prev_val + max_rate * dt;
      else
        ret = prev_val - max_rate * dt;
    }

    return ret;
  }

 protected:
 protected:
  sim_config* _sim_config;
  sim_in* _sim_in;
  sim_out* _sim_out;
  void* _sim_dims;

  ocp_nlp_in* _nlp_in;
  ocp_nlp_out* _nlp_out;
  ocp_nlp_dims* _nlp_dims;
  ocp_nlp_config* _nlp_config;
  ocp_nlp_solver* _nlp_solver;
  void* _nlp_opts;

  double* _new_time_steps;

  Eigen::VectorXd _state;
  Eigen::VectorXd _odom;

  Eigen::VectorXd _prev_x0;
  Eigen::VectorXd _prev_u0;

  std::map<std::string, double> _params;

  std::array<double, 2> _cmd;
  std::array<Eigen::VectorXd, 2> _tubes;

  int _mpc_steps;
  int _ref_samples;

  double _dt;
  double _ref_len_sz;

  bool _solve_success;
  bool _odom_init;
};
}  // namespace mpcc

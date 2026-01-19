#pragma once

#include <mpcc/mpcc_acados.h>
#include <mpcc/mpcc_di_acados.h>
#include <mpcc/types.h>

#include <map>
#include <variant>

namespace mpcc {
enum class MPCType { kDoubleIntegrator, kUnicycle };

class MPCCore {
 public:
  using AnyHorizon = std::variant<UnicycleMPCC::MPCHorizon, DIMPCC::MPCHorizon>;
  MPCCore();

  MPCCore(const MPCType& mpc_input_type);

  ~MPCCore();

  void load_params(const std::map<std::string, double>& params);
  /**********************************************************************
   * Function: MPCCore::load_params()
   * Description: Loads parameters for the MPC controller
   * Parameters:
   * @param params: const std::map<std::string, double>&
   * Returns:
   * N/A
   * Notes:
   * This function loads parameters for the MPC controller, including
   * the time step, maximum angular acc, linear acc, linear vel, angular
   * vel, and prop controller gains and thresholds
   **********************************************************************/

  std::array<double, 2> solve(const Eigen::VectorXd& state,
                              bool is_reverse = false);
  /**********************************************************************
   * Function: MPCCore::solve()
   * Description: Solves the MPC problem for the current timestep
   * Parameters:
   * @param state: const Eigen::VectorXd&
   * Returns:
   * control input for the next timestep
   * Notes:
   * This function solves the MPC problem and returns the control
   * inputs for the robot. If the robot's heading is not aligned with
   * the reference trajectory, orient_robot generates the control inputs
   **********************************************************************/

  /***********************
   * Setters and Getters
   ***********************/
  void set_odom(const Eigen::Vector3d& odom);
  void set_goal(const Eigen::Vector2d& goal);
  void set_trajectory(const Eigen::VectorXd& x_pts,
                      const Eigen::VectorXd& y_pts,
                      const Eigen::VectorXd& knot_parameters);
  void set_tubes(const std::array<Eigen::VectorXd, 2>& tubes);

  const bool get_solver_status() const;
  const double get_true_ref_len() const;
  const Eigen::VectorXd& get_state() const;
  const std::array<Eigen::VectorXd, 2>& get_tubes() const;
  const std::array<Eigen::VectorXd, 2> get_state_limits() const;
  const std::array<Eigen::VectorXd, 2> get_input_limits() const;
  AnyHorizon get_horizon() const;
  const std::map<std::string, double>& get_params() const;
  Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control,
                               bool is_abv) const;
  const types::Trajectory& get_trajectory() { return _trajectory; }

 private:
  // wizardry to wrap varient visit lambda function for mpc
  // using decltype(auto) to ensure constness is not stripped away
  // thanks to template argument deduction, don't need to use <> syntax
  // when calling call_mpc
  template <typename Callable>
  decltype(auto) call_mpc(Callable&& func) {
    return std::visit([&](auto& impl) -> decltype(auto) { return func(impl); },
                      _mpc);
  }

  template <typename Callable>
  decltype(auto) call_mpc(Callable&& func) const {
    return std::visit(
        [&](const auto& impl) -> decltype(auto) { return func(impl); }, _mpc);
  }

 private:
  double _dt{0.1};
  double _max_anga{2 * M_PI};
  double _max_linacc{2.0};
  double _curr_vel{0.};
  double _curr_angvel{0.};
  double _max_vel{2.0};
  double _max_angvel{M_PI / 2.};
  double _ref_length{0.};
  double _true_ref_length{0.};

  double _prop_gain{1.0};
  double _prop_angle_thresh{0.5};
  double _prev_s{0.};

  bool _is_set{false};
  bool _use_cbf{false};
  bool _traj_reset{false};
  bool _has_run{false};

  types::Trajectory _trajectory;

  std::array<double, 2> _prev_cmd;

  Eigen::Vector3d _odom{0., 0., 0.};
  Eigen::Vector2d _goal{0., 0.};

  std::map<std::string, double> _params;

  // std::unique_ptr<MPCBase> _mpc;
  std::variant<UnicycleMPCC, DIMPCC> _mpc;

  MPCType _mpc_input_type = MPCType::kUnicycle;
};
}  // namespace mpcc

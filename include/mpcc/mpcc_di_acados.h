#pragma once

#include <mpcc/mpcc_base.h>
#include <cstdlib>

#include <Eigen/Dense>
#include <map>
#include <vector>

// acados
#include "acados_sim_solver_double_integrator_mpcc.h"
#include "acados_solver_double_integrator_mpcc.h"

namespace mpcc {
using TrajectoryView = types::Trajectory::View;

class DIMPCC : public MPCBase {
 public:
  DIMPCC();
  virtual ~DIMPCC() override;

  virtual std::array<double, 2> solve(const Eigen::VectorXd& state,
                                      const TrajectoryView& reference,
                                      bool is_reverse = false) override;

  virtual void load_params(
      const std::map<std::string, double>& params) override;

  void reset_horizon() override;

  Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control,
                               bool is_abv) const override;

  virtual const Eigen::VectorXd& get_state() const override { return _state; }
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits()
      const override;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits()
      const override;

  virtual MPCHorizon get_horizon() const override;

  // TOOD: make getter for these
  // Use one vector which stores a state struct...
  std::vector<double> mpc_x;
  std::vector<double> mpc_y;
  std::vector<double> mpc_vx;
  std::vector<double> mpc_vy;
  std::vector<double> mpc_s;
  std::vector<double> mpc_s_dot;

  std::vector<double> mpc_ax;
  std::vector<double> mpc_ay;
  std::vector<double> mpc_s_ddots;

  static constexpr uint16_t kNX = DOUBLE_INTEGRATOR_MPCC_NX;
#ifdef DOUBLE_INTEGRATOR_MPCC_NS
  static constexpr uint16_t kNS = DOUBLE_INTEGRATOR_MPCC_NS;
#endif
  static constexpr uint16_t kNP   = DOUBLE_INTEGRATOR_MPCC_NP;
  static constexpr uint16_t kNU   = DOUBLE_INTEGRATOR_MPCC_NU;
  static constexpr uint16_t kNBX0 = DOUBLE_INTEGRATOR_MPCC_NBX0;

 private:
  Eigen::VectorXd next_state(const Eigen::VectorXd& current_state,
                             const Eigen::VectorXd& control);

  void process_solver_output();
  void warm_start_no_u(double* x_init);
  void warm_start_shifted_u(bool correct_perturb, const Eigen::VectorXd& state);
  bool set_solver_parameters(const TrajectoryView& reference);

  int initialize_acados();

 private:
  static constexpr uint8_t kIndX        = 0;
  static constexpr uint8_t kIndY        = 1;
  static constexpr uint8_t kIndVx       = 2;
  static constexpr uint8_t kIndVy       = 3;
  static constexpr uint8_t kIndS        = 4;
  static constexpr uint8_t kIndSDot     = 5;
  static constexpr uint8_t kIndStateInc = 6;

  static constexpr uint8_t kIndAx       = 0;
  static constexpr uint8_t kIndAy       = 1;
  static constexpr uint8_t kIndSDDot    = 2;
  static constexpr uint8_t kIndInputInc = 3;

  bool _has_run;
  // bool _solve_success;
  bool _is_shift_warm;

  double _s_dot;

  // parameters
  double _max_linvel;
  double _max_linacc;

  double _w_ql;
  double _w_qc;
  double _w_q_speed;
  double _gamma;
  double _w_ql_lyap;
  double _w_qc_lyap;
  double _alpha_abv;
  double _alpha_blw;
  double _colinear;
  double _padding;

  bool _use_cbf;

  double_integrator_mpcc_sim_solver_capsule* _acados_sim_capsule = nullptr;
  double_integrator_mpcc_solver_capsule* _acados_ocp_capsule     = nullptr;
};
}  // namespace mpcc

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

class DIMPCC : public MPCBase<DIMPCC> {
  friend class MPCBase<DIMPCC>;

 public:
  static constexpr uint16_t kNX = DOUBLE_INTEGRATOR_MPCC_NX;
#ifdef DOUBLE_INTEGRATOR_MPCC_NS
  static constexpr uint16_t kNS = DOUBLE_INTEGRATOR_MPCC_NS;
#endif
  static constexpr uint16_t kNP   = DOUBLE_INTEGRATOR_MPCC_NP;
  static constexpr uint16_t kNU   = DOUBLE_INTEGRATOR_MPCC_NU;
  static constexpr uint16_t kNBX0 = DOUBLE_INTEGRATOR_MPCC_NBX0;

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

 public:
  struct StateHorizon : public types::StateHorizon {
   public:
    Eigen::VectorXd vs_x;
    Eigen::VectorXd vs_y;

    Eigen::Matrix<double, kNX, 1> get_state_at_step(unsigned int step) const {
      if (step >= xs.size()) {
        throw std::runtime_error(
            "[MPCHorizon] requested state at step " + std::to_string(step) +
            " for horizon of size " + std::to_string(xs.size()));
      }
      return {xs[step],   ys[step],      vs_x[step],
              vs_y[step], arclens[step], arclens_dot[step]};
    }
  };

  struct InputHorizon : public types::InputHorizon {
   public:
    Eigen::VectorXd accs_x;
    Eigen::VectorXd accs_y;

    Eigen::Matrix<double, kNU, 1> get_input_at_step(unsigned int step) const {
      if (step >= arclens_ddot.size()) {
        throw std::runtime_error(
            "[MPCHorizon] requested input at step " + std::to_string(step) +
            " for horizon of size " + std::to_string(arclens_ddot.size()));
      }
      return {accs_x[step], accs_y[step], arclens_ddot[step]};
    }
  };

  using MPCHorizon = types::MPCHorizon<DIMPCC>;

  DIMPCC();
  virtual ~DIMPCC();

  // virtual void load_params(
  //     const std::map<std::string, double>& params) override;
  void load_params(const std::map<std::string, double>& params);

  void reset_horizon() override;

  Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control,
                               bool is_abv) const override;

  virtual const Eigen::VectorXd& get_state() const override { return _state; }
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits()
      const override;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits()
      const override;

  MPCHorizon get_horizon() const;

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

 private:
  Eigen::VectorXd next_state(const Eigen::VectorXd& current_state,
                             const Eigen::VectorXd& control);

  Eigen::VectorXd prepare_initial_state(const Eigen::VectorXd& state);

  bool run_acados_solver(const Eigen::VectorXd& initial_state);

  std::array<double, 2> compute_mpc_vel_command(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& u);

  // void process_solver_output();
  // void warm_start_no_u(double* x_init);
  // void warm_start_shifted_u(bool correct_perturb, const Eigen::VectorXd& state);

  void acados_capsule_update_params(const std::vector<double>& params,
                                    unsigned int step);

  void map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                 const Eigen::VectorXd& utraj);

  bool is_acados_ready() { return _acados_ocp_capsule != nullptr; }

  int initialize_acados();

 private:
  bool _has_run;
  // bool _solve_success;
  // bool _is_shift_warm;

  double _s_dot;

  // parameters
  double _max_linvel;
  double _max_linacc;

  double_integrator_mpcc_sim_solver_capsule* _acados_sim_capsule = nullptr;
  double_integrator_mpcc_solver_capsule* _acados_ocp_capsule     = nullptr;
};
}  // namespace mpcc

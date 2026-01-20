#pragma once

#include <mpcc/mpcc_base.h>
#include <mpcc/orientable.h>
#include <mpcc/types.h>

#include <cstdlib>
#include <map>
#include <vector>

#include <Eigen/Dense>

// acados
#include "acados_sim_solver_unicycle_model_mpcc.h"
#include "acados_solver_unicycle_model_mpcc.h"

namespace mpcc {
using TrajectoryView = types::Trajectory::View;

class UnicycleMPCC;

// setup acados solver traits for class
template <>
struct types::SolverTraits<UnicycleMPCC> {
  using SolverCapsule = unicycle_model_mpcc_solver_capsule;

  static int create_capsule(double mpc_steps, double*& time_steps,
                            SolverCapsule*& capsule) {
    capsule = unicycle_model_mpcc_acados_create_capsule();
    return unicycle_model_mpcc_acados_create_with_discretization(
        capsule, mpc_steps, time_steps);
  }

  static void free_capsule(SolverCapsule*& capsule) noexcept {
    unicycle_model_mpcc_acados_free(capsule);
  }

  static ocp_nlp_in* get_nlp_in(SolverCapsule* capsule) {
    return unicycle_model_mpcc_acados_get_nlp_in(capsule);
  }

  static ocp_nlp_out* get_nlp_out(SolverCapsule* capsule) {
    return unicycle_model_mpcc_acados_get_nlp_out(capsule);
  }

  static void* get_nlp_opts(SolverCapsule* capsule) {
    return unicycle_model_mpcc_acados_get_nlp_opts(capsule);
  }

  static ocp_nlp_dims* get_nlp_dims(SolverCapsule* capsule) {
    return unicycle_model_mpcc_acados_get_nlp_dims(capsule);
  }

  static ocp_nlp_solver* get_nlp_solver(SolverCapsule* capsule) {
    return unicycle_model_mpcc_acados_get_nlp_solver(capsule);
  }

  static ocp_nlp_config* get_nlp_config(SolverCapsule* capsule) {
    return unicycle_model_mpcc_acados_get_nlp_config(capsule);
  }

  static int solve(SolverCapsule* capsule) {
    return unicycle_model_mpcc_acados_solve(capsule);
  }

  static void set_params(SolverCapsule* capsule, unsigned int step,
                         const std::vector<double>& params) {
    unicycle_model_mpcc_acados_update_params(
        capsule, step, const_cast<double*>(params.data()), params.size());
  }
};

class UnicycleMPCC : public MPCBase<UnicycleMPCC>, public Orientable {
  friend class MPCBase<UnicycleMPCC>;

 public:
  static constexpr uint16_t kNX = UNICYCLE_MODEL_MPCC_NX;
#ifdef UNICYCLE_MODEL_UnicycleMPCC_NS
  static constexpr uint16_t kNS = UNICYCLE_MODEL_MPCC_NS;
#endif
  static constexpr uint16_t kNP   = UNICYCLE_MODEL_MPCC_NP;
  static constexpr uint16_t kNU   = UNICYCLE_MODEL_MPCC_NU;
  static constexpr uint16_t kNBX0 = UNICYCLE_MODEL_MPCC_NBX0;

  static constexpr uint8_t kIndX        = 0;
  static constexpr uint8_t kIndY        = 1;
  static constexpr uint8_t kIndTheta    = 2;
  static constexpr uint8_t kIndV        = 3;
  static constexpr uint8_t kIndS        = 4;
  static constexpr uint8_t kIndSDot     = 5;
  static constexpr uint8_t kIndStateInc = 6;

  static constexpr uint8_t kIndLinAcc   = 0;
  static constexpr uint8_t kIndAngVel   = 1;
  static constexpr uint8_t kIndSDDot    = 2;
  static constexpr uint8_t kIndInputInc = 3;

 public:
  struct StateHorizon : public types::StateHorizon {
   public:
    Eigen::VectorXd thetas;
    Eigen::VectorXd vs;

    Eigen::Matrix<double, kNX, 1> get_state_at_step(unsigned int step) const {
      if (step >= xs.size()) {
        throw std::runtime_error(
            "[MPCHorizon] requested state at step " + std::to_string(step) +
            " for horizon of size " + std::to_string(xs.size()));
      }
      return {xs[step], ys[step],      thetas[step],
              vs[step], arclens[step], arclens_dot[step]};
    }
  };

  struct InputHorizon : public types::InputHorizon {
   public:
    Eigen::VectorXd angvels;
    Eigen::VectorXd linaccs;

    Eigen::Matrix<double, kNU, 1> get_input_at_step(unsigned int step) const {
      if (step >= arclens_ddot.size()) {
        throw std::runtime_error(
            "[MPCHorizon] requested input at step " + std::to_string(step) +
            " for horizon of size " + std::to_string(arclens_ddot.size()));
      }
      // return {angvels[step], linaccs[step], arclens_ddot[step]};
      return {linaccs[step], angvels[step], arclens_ddot[step]};
    }
  };

  using MPCHorizon = types::MPCHorizon<UnicycleMPCC>;

 public:
  UnicycleMPCC();
  virtual ~UnicycleMPCC();

  void load_params(const std::map<std::string, double>& params);
  // virtual void load_params(
  //     const std::map<std::string, double>& params) override;
  /**********************************************************************
   * Function: UnicycleMPCC::load_params()
   * Description: Loads parameters for the MPC controller
   * Parameters:
   * @param params: const std::map<std::string, double>&
   * Returns:
   * N/A
   * Notes:
   * This function loads parameters for the MPC controller, including
   * the time step, maximum angular and body rates, weights for the
   * mpcc cost function, and CBF/CLF parameters
   **********************************************************************/

  /***********************
   * Setters and Getters
   ***********************/
  virtual void reset_horizon() override;

  Eigen::VectorXd get_cbf_data(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control,
                               bool is_abv) const override;

  virtual const Eigen::VectorXd& get_state() const override { return _state; };
  virtual const std::array<Eigen::VectorXd, 2> get_state_limits()
      const override;
  virtual const std::array<Eigen::VectorXd, 2> get_input_limits()
      const override;

  MPCHorizon get_horizon() const;

 public:
  // TOOD: make getter for these
  // Use one vector which stores a state struct...
  std::vector<double> mpc_x;
  std::vector<double> mpc_y;
  std::vector<double> mpc_theta;
  std::vector<double> mpc_linvels;
  std::vector<double> mpc_s;
  std::vector<double> mpc_s_dot;

  std::vector<double> mpc_angvels;
  std::vector<double> mpc_linaccs;
  std::vector<double> mpc_s_ddots;

 private:
  Eigen::VectorXd next_state(const Eigen::VectorXd& current_state,
                             const Eigen::VectorXd& control);
  /**********************************************************************
   * Function: UnicycleMPCC::next_state()
   * Description: Calculates the next state of the robot given current
   * state and control input
   * Parameters:
   * @param current_state: const Eigen::VectorXd&
   * @param control: const Eigen::VectorXd&
   * Returns:
   * Next state of the robot
   **********************************************************************/

  // void warm_start_no_u(double* x_init);
  /**********************************************************************
   * Function: UnicycleMPCC::warm_start_no_u()
   * Description: Warm starts the MPC solver with no control inputs
   * Parameters:
   * @param x_init: double*
   * Returns:
   * N/A
   * Notes:
   * This function sets the initial state for the MPC solver assuming
   * a 0 control input
   **********************************************************************/

  // void warm_start_shifted_u(bool correct_perturb, const Eigen::VectorXd& state);
  /**********************************************************************
   * Function: UnicycleMPCC::warm_start_shifted_u()
   * Description: Warm starts the MPC solver with shifted control inputs
   * Parameters:
   * @param correct_perturb: bool
   * @param state: const Eigen::VectorXd&
   * Returns:
   * N/A
   * Notes:
   * This function sets the initial state for the MPC solver by shifting
   * the control inputs and states from the previous solution.
   * See From linear to nonlinear MPC: bridging the gap via the
   * real-time iteration, Gros et. al. for more details.
   **********************************************************************/

  // void process_solver_output();
  /**********************************************************************
   * Function: UnicycleMPCC::process_solver_output()
   * Description: Processes the output of the MPC solver
   * Parameters:
   * @param s: double
   * Returns:
   * N/A
   **********************************************************************/

  Eigen::VectorXd prepare_initial_state(const Eigen::VectorXd& state);

  std::array<double, 2> compute_mpc_vel_command(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& u);

  void map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                 const Eigen::VectorXd& utraj);

  std::optional<std::array<double, 2>> presolve_hook(
      const Eigen::VectorXd& state, const types::Corridor& corridor) const;

 private:
  Eigen::MatrixXd _dyna_obs;

  double _ds;

  double _bound_value;
  double _max_linvel;
  double _max_angvel;
  double _max_linacc;
  double _max_anga;

  double _traj_alignment_threshold;
  double _alignment_p_gain;

  double _s_dot;

  unsigned int iterations;

  bool _use_eigen;
  bool _use_dyna_obs;
};
}  // namespace mpcc

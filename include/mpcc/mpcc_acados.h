#pragma once

#include <mpcc/mpcc_base.h>
#include <mpcc/types.h>
#include <cstdlib>

#include <Eigen/Dense>
#include <map>
#include <vector>

// acados
#include "acados_sim_solver_unicycle_model_mpcc.h"
#include "acados_solver_unicycle_model_mpcc.h"

namespace mpcc {
using TrajectoryView = types::Trajectory::View;

class UnicycleMPCC : public MPCBase<UnicycleMPCC> {
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

  bool run_acados_solver(const Eigen::VectorXd& initial_state);

  std::array<double, 2> compute_mpc_vel_command(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& u);

  // vs_x and vs_y must be _mpc_steps length prior
  void compute_world_frame_velocities(Eigen::VectorXd& vs_x,
                                      Eigen::VectorXd& vs_y) const;

  // accs_x and accs_y must be _mpc_steps -1 length prior
  void compute_world_frame_accelerations(Eigen::VectorXd& accs_x,
                                         Eigen::VectorXd& accs_y) const;

  void acados_capsule_update_params(const std::vector<double>& params,
                                    unsigned int step);

  void map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                 const Eigen::VectorXd& utraj);

  bool is_acados_ready() { return _acados_ocp_capsule != nullptr; }

 private:
  Eigen::VectorXd _prev_x0;
  Eigen::VectorXd _prev_u0;

  Eigen::MatrixXd _dyna_obs;

  double _ds;

  double _bound_value;
  double _max_linvel;
  double _max_angvel;
  double _max_linacc;
  double _max_anga;

  // double _alpha_abv;
  // double _alpha_blw;
  // double _colinear;
  // double _padding;

  double _s_dot;

  unsigned int iterations;

  bool _use_eigen;
  // bool _is_shift_warm;
  bool _use_dyna_obs;
  // bool _has_run;

  unicycle_model_mpcc_sim_solver_capsule* _acados_sim_capsule;
  unicycle_model_mpcc_solver_capsule* _acados_ocp_capsule;
};
}  // namespace mpcc

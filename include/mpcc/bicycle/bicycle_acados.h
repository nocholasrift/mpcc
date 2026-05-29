#pragma once

#include <mpcc/common/mpcc_base.h>
#include <cstdlib>

#include <Eigen/Dense>
#include <map>
#include <vector>

// acados generated auto-code header
#include "acados_solver_bicycle_mpcc.h"

namespace mpcc {
using TrajectoryView = types::Trajectory::View;

class BicycleMPCC;

template <>
struct types::SolverTraits<BicycleMPCC> {
  using SolverCapsule = bicycle_mpcc_solver_capsule;

  static int create_capsule(double mpc_steps, double*& time_steps,
                            SolverCapsule*& capsule) {
    capsule = bicycle_mpcc_acados_create_capsule();
    return bicycle_mpcc_acados_create_with_discretization(capsule, mpc_steps,
                                                          time_steps);
  }

  static void free_capsule(SolverCapsule*& capsule) noexcept {
    bicycle_mpcc_acados_free(capsule);
  }

  static ocp_nlp_in* get_nlp_in(SolverCapsule* capsule) {
    return bicycle_mpcc_acados_get_nlp_in(capsule);
  }

  static ocp_nlp_out* get_nlp_out(SolverCapsule* capsule) {
    return bicycle_mpcc_acados_get_nlp_out(capsule);
  }

  static void* get_nlp_opts(SolverCapsule* capsule) {
    return bicycle_mpcc_acados_get_nlp_opts(capsule);
  }

  static ocp_nlp_dims* get_nlp_dims(SolverCapsule* capsule) {
    return bicycle_mpcc_acados_get_nlp_dims(capsule);
  }

  static ocp_nlp_solver* get_nlp_solver(SolverCapsule* capsule) {
    return bicycle_mpcc_acados_get_nlp_solver(capsule);
  }

  static ocp_nlp_config* get_nlp_config(SolverCapsule* capsule) {
    return bicycle_mpcc_acados_get_nlp_config(capsule);
  }

  static int solve(SolverCapsule* capsule) {
    return bicycle_mpcc_acados_solve(capsule);
  }

  static void set_params(SolverCapsule* capsule, unsigned int step,
                         const std::vector<double>& params) {
    bicycle_mpcc_acados_update_params(
        capsule, step, const_cast<double*>(params.data()), params.size());
  }
};

class BicycleMPCC : public MPCBase<BicycleMPCC> {
  friend class MPCBase<BicycleMPCC>;

 public:
  static constexpr uint16_t kNX = BICYCLE_MPCC_NX;
#ifdef BICYCLE_MPCC_NS
  static constexpr uint16_t kNS = BICYCLE_MPCC_NS;
#endif
  static constexpr uint16_t kNP   = BICYCLE_MPCC_NP;
  static constexpr uint16_t kNU   = BICYCLE_MPCC_NU;
  static constexpr uint16_t kNBX0 = BICYCLE_MPCC_NBX0;

  // State Array Layout: [x, y, theta, v, s, sdot]
  static constexpr uint8_t kIndX        = 0;
  static constexpr uint8_t kIndY        = 1;
  static constexpr uint8_t kIndTheta    = 2;
  static constexpr uint8_t kIndV        = 3;
  static constexpr uint8_t kIndS        = 4;
  static constexpr uint8_t kIndSDot     = 5;
  static constexpr uint8_t kIndStateInc = 6;

  // Input Array Layout: [a, delta, sddot]
  static constexpr uint8_t kIndA        = 0;
  static constexpr uint8_t kIndDelta    = 1;
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
      Eigen::Matrix<double, kNX, 1> ret;
      ret << xs[step], ys[step], thetas[step], vs[step], arclens[step],
          arclens_dot[step];
      return ret;
    }
  };

  struct InputHorizon : public types::InputHorizon {
   public:
    Eigen::VectorXd accs;
    Eigen::VectorXd deltas;

    Eigen::Matrix<double, kNU, 1> get_input_at_step(unsigned int step) const {
      if (step >= arclens_ddot.size()) {
        throw std::runtime_error(
            "[MPCHorizon] requested input at step " + std::to_string(step) +
            " for horizon of size " + std::to_string(arclens_ddot.size()));
      }
      Eigen::Matrix<double, kNU, 1> ret;
      ret << accs[step], deltas[step], arclens_ddot[step];
      return ret;
    }
  };

  struct MPCHorizon : public types::MPCHorizon<BicycleMPCC> {
    Eigen::Vector2d get_pos(unsigned int step) const {
      return {states.xs[step], states.ys[step]};
    }

    Eigen::Vector2d get_vel(unsigned int step) const {
      return {states.vs[step] * cos(states.thetas[step]),
              states.vs[step] * sin(states.thetas[step])};
    }

    Eigen::Vector2d get_acc(unsigned int step) const {
      return {inputs.accs[step] * cos(states.thetas[step]),
              inputs.accs[step] * sin(states.thetas[step])};
    }

    // double get_heading(unsigned int step) const { return states.thetas[step]; }
    //
    // double get_speed(unsigned int step) const { return states.vs[step]; }
    //
    // double get_steer(unsigned int step) const { return inputs.deltas[step]; }
  };

 public:
  BicycleMPCC() = default;
  BicycleMPCC(std::shared_ptr<MPCConfig> cfg);
  virtual ~BicycleMPCC();

  void load_params(std::shared_ptr<MPCConfig> cfg);
  void reset_horizon();

  Eigen::VectorXd get_cbf_data(const types::Corridor& corridor,
                               size_t horizon_idx) const;

  const std::array<Eigen::VectorXd, 2> get_state_limits() const;
  const std::array<Eigen::VectorXd, 2> get_input_limits() const;

  MPCHorizon get_horizon() const;

 private:
  Eigen::VectorXd next_state(const Eigen::VectorXd& current_state,
                             const Eigen::VectorXd& control);

  Eigen::VectorXd prepare_initial_state(const Eigen::VectorXd& state,
                                        const types::Corridor& corridor);

  std::array<double, 2> compute_mpc_vel_command(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& u);

  void map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                 const Eigen::VectorXd& utraj);

  bool set_solver_parameters(const types::Corridor& corridor);

 private:
  std::vector<double> mpc_x;
  std::vector<double> mpc_y;
  std::vector<double> mpc_theta;
  std::vector<double> mpc_v;
  std::vector<double> mpc_s;
  std::vector<double> mpc_s_dot;

  std::vector<double> mpc_a;
  std::vector<double> mpc_delta;
  std::vector<double> mpc_s_ddots;

  bool _has_run;
  double _s_dot;
  double _body_len;

  double _max_linvel;
  double _max_linacc;
};
}  // namespace mpcc

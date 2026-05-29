#include <mpcc/bicycle/bicycle_acados.h>
#include <mpcc/bicycle/casadi_bicycle_interface.h>
#include "mpcc/common/types.h"

#include <bicycle_mpcc_param_indices.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace mpcc;

const uint16_t BicycleMPCC::kNX;
const uint16_t BicycleMPCC::kNS;
const uint16_t BicycleMPCC::kNP;
const uint16_t BicycleMPCC::kNU;
const uint16_t BicycleMPCC::kNBX0;

BicycleMPCC::BicycleMPCC(std::shared_ptr<MPCConfig> cfg) {
  _state = Eigen::VectorXd::Zero(kNX);
  _odom  = Eigen::VectorXd::Zero(3);

  _has_run       = false;
  _solve_success = false;
  _is_shift_warm = false;
  _odom_init     = false;

  load_params(cfg);
}

BicycleMPCC::~BicycleMPCC() {}

void BicycleMPCC::load_params(std::shared_ptr<MPCConfig> cfg) {
  _mpc_cfg = cfg;

  double mpc_steps = _mpc_cfg->steps;
  std::cout << "Bicycle MPC steps: " << mpc_steps << "\n";
  int status = _acados_solver.initialize(mpc_steps);
  if (status) {
    throw std::runtime_error("Acados initialization failed with status " +
                             std::to_string(status) + "!");
  }

  mpc_x.resize(mpc_steps + 1);
  mpc_y.resize(mpc_steps + 1);
  mpc_theta.resize(mpc_steps + 1);
  mpc_v.resize(mpc_steps + 1);
  mpc_s.resize(mpc_steps + 1);
  mpc_s_dot.resize(mpc_steps + 1);

  mpc_a.resize(mpc_steps);
  mpc_delta.resize(mpc_steps);
  mpc_s_ddots.resize(mpc_steps);

  if (_prev_x0.size() != (mpc_steps + 1) * kNX) {
    std::cout
        << "x0, u0 size differs from mpc_steps size, resizing and clearing.\n";
    _prev_x0 = Eigen::VectorXd::Zero((mpc_steps + 1) * kNX);
    _prev_u0 = Eigen::VectorXd::Zero(mpc_steps * kNU);
  }

  _acados_solver.initialize(mpc_steps);

  std::cout << "!! MPC Obj parameters updated !! " << std::endl;
  std::cout << "!! ACADOS bicycle model instantiated !! " << std::endl;
}

bool BicycleMPCC::set_solver_parameters(const types::Corridor& corridor) {
  using Side      = types::Corridor::Side;
  namespace Param = mpcc::bicycle_param;

  std::vector<double> params;
  params.resize(Param::kNP);

  const TrajectoryView& traj_view = corridor.get_trajectory().view();
  const auto& ctrls_x             = traj_view.xs;
  const auto& ctrls_y             = traj_view.ys;

  if (corridor.get_above_poly().get_degree() !=
      corridor.get_below_poly().get_degree()) {
    std::cerr << "[MPCC] Tube degrees do not match for above and below!"
              << std::endl;
    return false;
  }

  int N_tube_coeffs = corridor.get_above_poly().get_coeffs().size();
  int provided_params =
      ctrls_x.size() + ctrls_y.size() + 2 * N_tube_coeffs + Param::kNP_Scalar;

  if (provided_params != Param::kNP) {
    std::cerr << "[MPCC] Provided param count " << provided_params
              << " does not match acados dimension parameter size "
              << Param::kNP << std::endl;
    return false;
  }

  params[Param::k_Q_c]         = _mpc_cfg->weights.w_contour_e;
  params[Param::k_Q_l]         = _mpc_cfg->weights.w_lag_e;
  params[Param::k_Q_s]         = _mpc_cfg->weights.w_speed;
  params[Param::k_alpha_abv]   = _mpc_cfg->cbf.alpha_abv;
  params[Param::k_alpha_blw]   = _mpc_cfg->cbf.alpha_blw;
  params[Param::k_Ql_c]        = _mpc_cfg->clf.w_contour_e;
  params[Param::k_Ql_l]        = _mpc_cfg->clf.w_lag_e;
  params[Param::k_gamma]       = _mpc_cfg->clf.gamma;
  params[Param::k_body_length] = _mpc_cfg->body_length;
  params[Param::k_L_path]      = traj_view.arclen;

  int N_ctrls = ctrls_x.size();
  for (int i = 0; i < N_ctrls; ++i) {
    params[i]           = ctrls_x[i];
    params[i + N_ctrls] = ctrls_y[i];
  }

  const auto& above_coeffs = corridor.get_tube_coeffs(Side::kAbove);
  const auto& below_coeffs = corridor.get_tube_coeffs(Side::kBelow);

  for (int i = 0; i < N_tube_coeffs; ++i) {
    params[i + 2 * N_ctrls]                 = above_coeffs(i);
    params[i + 2 * N_ctrls + N_tube_coeffs] = below_coeffs(i);
  }

  double mpc_steps = _mpc_cfg->steps;
  for (int step = 0; step < mpc_steps + 1; ++step) {
    _acados_solver.update_params(step, params);
  }

  return true;
}

void BicycleMPCC::reset_horizon() {
  double mpc_steps = _mpc_cfg->steps;
  for (int i = 0; i < mpc_steps; ++i) {
    mpc_x[i]     = _state(kIndX);
    mpc_y[i]     = _state(kIndY);
    mpc_theta[i] = _state(kIndTheta);
    mpc_v[i]     = 0.0;
    mpc_s[i]     = 1e-2;
    mpc_s_dot[i] = 0.0;
  }

  for (int i = 0; i < mpc_steps - 1; ++i) {
    mpc_a[i]       = 0.0;
    mpc_delta[i]   = 0.0;
    mpc_s_ddots[i] = 0.0;
  }
}

Eigen::VectorXd BicycleMPCC::get_cbf_data(const types::Corridor& corridor,
                                          size_t horizon_idx) const {
  Eigen::VectorXd state = _prev_x0.segment(horizon_idx * kNX, kNX);
  Eigen::VectorXd input = _prev_u0.segment(horizon_idx * kNU, kNU);

  CasadiBicycleInterface::Params params;
  params.qc_lyap    = _mpc_cfg->clf.w_contour_e;
  params.ql_lyap    = _mpc_cfg->clf.w_lag_e;
  params.gamma_lyap = _mpc_cfg->clf.gamma;

  CasadiBicycleInterface casadi_interface;
  double h_abv = casadi_interface.get_h_abv(state, input, corridor, params);
  double hdot_abv =
      casadi_interface.get_h_dot_abv(state, input, corridor, params);

  double h_blw = casadi_interface.get_h_blw(state, input, corridor, params);
  double hdot_blw =
      casadi_interface.get_h_dot_blw(state, input, corridor, params);

  return Eigen::Vector4d(h_abv, hdot_abv, h_blw, hdot_blw);
}

Eigen::VectorXd BicycleMPCC::next_state(const Eigen::VectorXd& current_state,
                                        const Eigen::VectorXd& control) {
  Eigen::VectorXd ret(kNX);

  double x1    = current_state(kIndX);
  double y1    = current_state(kIndY);
  double theta = current_state(kIndTheta);
  double v     = current_state(kIndV);
  double s1    = current_state(kIndS);
  double sdot1 = current_state(kIndSDot);

  double a     = control(kIndA);
  double delta = control(kIndDelta);
  double sddot = control(kIndSDDot);

  double dt         = _mpc_cfg->dt;
  double L          = _mpc_cfg->body_length;
  double max_linvel = _mpc_cfg->constraints.max_linvel;

  // Kinematic bicycle vehicle footprint constants
  // double lf   = 0.16;
  // double lr   = 0.15;
  // double beta = std::atan((lr / (lf + lr)) * std::tan(delta));

  ret(kIndX) = x1 + v * std::cos(theta) * dt;
  ret(kIndY) = y1 + v * std::sin(theta) * dt;
  // ret(kIndTheta) = theta + (v / lr) * std::sin(beta) * dt;
  ret(kIndTheta) = theta + (v * tan(delta) / L) * dt;
  ret(kIndV)     = std::max(std::min(v + a * dt, max_linvel),
                            0.0);  // Velocity should be non-negative
  ret(kIndS)     = s1 + sdot1 * dt;
  ret(kIndSDot)  = std::max(std::min(sdot1 + sddot * dt, max_linvel), 0.0);

  return ret;
}

Eigen::VectorXd BicycleMPCC::prepare_initial_state(
    const Eigen::VectorXd& state, const types::Corridor& corridor) {
  Eigen::VectorXd x0 = state;

  if (_has_run) {
    Eigen::VectorXd prev_x0 = _prev_x0.head(kNX);
    double etheta           = x0(kIndTheta) - prev_x0(kIndTheta);
    if (etheta > M_PI)
      x0(kIndTheta) -= 2 * M_PI;
    if (etheta < -M_PI)
      x0(kIndTheta) += 2 * M_PI;
  }

  double min_sdot = 0.1;
  x0(kIndSDot)    = std::max(x0(kIndV), min_sdot);

  return x0;
}

std::array<double, 2> BicycleMPCC::compute_mpc_vel_command(
    const Eigen::VectorXd& state, const Eigen::VectorXd& u) {

  double dt         = _mpc_cfg->dt;
  double max_linacc = _mpc_cfg->constraints.max_linacc;

  std::cout << "mpc U is: " << u[kIndA] << ", " << u[kIndDelta] << "\n";

  // Calculate integrated target tracking velocity
  double new_w = (state[kIndV] * u[kIndDelta] / _mpc_cfg->body_length) * dt;
  double new_v =
      limit(state[kIndV], state[kIndV] + u[kIndA] * dt, max_linacc, dt);
  double max_linvel = _mpc_cfg->constraints.max_linvel;
  new_v             = std::max(std::min(new_v, max_linvel), 0.0);

  // Return commanded linear speed along with the localized steering control angle
  return {new_v, new_w};
}

void BicycleMPCC::map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                            const Eigen::VectorXd& utraj) {
  double mpc_steps = _mpc_cfg->steps;
  for (int i = 0; i <= mpc_steps; ++i) {
    mpc_x[i]     = xtraj[kIndX + i * kIndStateInc];
    mpc_y[i]     = xtraj[kIndY + i * kIndStateInc];
    mpc_theta[i] = xtraj[kIndTheta + i * kIndStateInc];
    mpc_v[i]     = xtraj[kIndV + i * kIndStateInc];
    mpc_s[i]     = xtraj[kIndS + i * kIndStateInc];
    mpc_s_dot[i] = xtraj[kIndSDot + i * kIndStateInc];
  }

  for (int i = 0; i < mpc_steps; ++i) {
    mpc_a[i]       = utraj[kIndA + i * kIndInputInc];
    mpc_delta[i]   = utraj[kIndDelta + i * kIndInputInc];
    mpc_s_ddots[i] = utraj[kIndSDDot + i * kIndInputInc];
  }
}

const std::array<Eigen::VectorXd, 2> BicycleMPCC::get_state_limits() const {
  Eigen::VectorXd xmin(kNX), xmax(kNX);
  double max_linvel = _mpc_cfg->constraints.max_linvel;

  // Limits layout: [x, y, theta, v, s, sdot]
  xmin << -1e3, -1e3, -2.0 * M_PI, 0.0, 0.0, 0.0;
  xmax << 1e3, 1e3, 2.0 * M_PI, max_linvel, _mpc_cfg->ref_length, max_linvel;

  return {xmin, xmax};
}

const std::array<Eigen::VectorXd, 2> BicycleMPCC::get_input_limits() const {
  Eigen::VectorXd umin(kNU), umax(kNU);
  double max_linacc = _mpc_cfg->constraints.max_linacc;
  double max_steer  = tan(0.52);  // ~30 degrees max steer angle bounds

  // Limits layout: [a, delta, sddot]
  umin << -max_linacc, -max_steer, -max_linacc;
  umax << max_linacc, max_steer, max_linacc;

  return {umin, umax};
}

BicycleMPCC::MPCHorizon BicycleMPCC::get_horizon() const {
  double mpc_steps = _mpc_cfg->steps;
  BicycleMPCC::MPCHorizon horizon;

  horizon.states.xs          = utils::vector_to_eigen(mpc_x);
  horizon.states.ys          = utils::vector_to_eigen(mpc_y);
  horizon.states.thetas      = utils::vector_to_eigen(mpc_theta);
  horizon.states.vs          = utils::vector_to_eigen(mpc_v);
  horizon.states.arclens     = utils::vector_to_eigen(mpc_s);
  horizon.states.arclens_dot = utils::vector_to_eigen(mpc_s_dot);

  horizon.inputs.accs         = utils::vector_to_eigen(mpc_a);
  horizon.inputs.deltas       = utils::vector_to_eigen(mpc_delta);
  horizon.inputs.arclens_ddot = utils::vector_to_eigen(mpc_s_ddots);
  horizon.length              = mpc_steps + 1;

  const auto N = mpc_steps + 1;

  assert(horizon.states.xs.size() == N);
  assert(horizon.states.ys.size() == N);
  assert(horizon.states.thetas.size() == N);
  assert(horizon.states.vs.size() == N);
  assert(horizon.states.arclens.size() == N);
  assert(horizon.states.arclens_dot.size() == N);

  assert(horizon.inputs.accs.size() == N - 1);
  assert(horizon.inputs.deltas.size() == N - 1);
  assert(horizon.inputs.arclens_ddot.size() == N - 1);

  return horizon;
}

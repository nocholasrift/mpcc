#include <mpcc/double_integrator/casadi_double_integrator_interface.h>
#include <mpcc/double_integrator/mpcc_di_acados.h>
#include "mpcc/common/types.h"

#include <double_integrator_mpcc_param_indices.h>

#include <cmath>
#include <stdexcept>

using namespace mpcc;

// #ifdef FOUND_PYBIND11
// the declarations in the header cannot be referenced by pybind...
// need to define them
const uint16_t DIMPCC::kNX;
const uint16_t DIMPCC::kNS;
const uint16_t DIMPCC::kNP;
const uint16_t DIMPCC::kNU;
const uint16_t DIMPCC::kNBX0;
// #endif

DIMPCC::DIMPCC(std::shared_ptr<MPCConfig> cfg) {
  _state = Eigen::VectorXd::Zero(kNX);
  _odom  = Eigen::VectorXd(3);

  _has_run       = false;
  _solve_success = false;
  _is_shift_warm = false;
  _odom_init     = false;

  load_params(cfg);
}

DIMPCC::~DIMPCC() {}

// void DIMPCC::load_params(const std::map<std::string, double>& params) {
void DIMPCC::load_params(std::shared_ptr<MPCConfig> cfg) {

  _mpc_cfg = cfg;

  double mpc_steps = _mpc_cfg->steps;
  std::cout << "mpc steps: " << mpc_steps << "\n";
  int status = _acados_solver.initialize(mpc_steps);
  if (status) {
    throw std::runtime_error("Acados initialization failed with status + " +
                             std::to_string(status) + "!");
  }

  mpc_x.resize(mpc_steps + 1);
  mpc_y.resize(mpc_steps + 1);
  mpc_vx.resize(mpc_steps + 1);
  mpc_vy.resize(mpc_steps + 1);
  mpc_s.resize(mpc_steps + 1);
  mpc_s_dot.resize(mpc_steps + 1);

  mpc_ax.resize(mpc_steps);
  mpc_ay.resize(mpc_steps);
  mpc_s_ddots.resize(mpc_steps);

  if (_prev_x0.size() != (mpc_steps + 1) * kNX) {
    std::cout << termcolor::yellow
              << "x0, u0 size differs from mpc_steps size, resizing and "
                 "zeroing out\n";
    _prev_x0 = Eigen::VectorXd::Zero((mpc_steps + 1) * kNX);
    _prev_u0 = Eigen::VectorXd::Zero(mpc_steps * kNU);
  }

  _acados_solver.initialize(mpc_steps);

  std::cout << "!! MPC Obj parameters updated !! " << std::endl;
  std::cout << "!! ACADOS model instantiated !! " << std::endl;
}

bool DIMPCC::set_solver_parameters(const types::Corridor& corridor) {
  using Side      = types::Corridor::Side;
  namespace Param = mpcc::double_integrator_param;

  std::vector<double> params;
  params.resize(Param::kNP);

  const TrajectoryView& traj_view = corridor.get_trajectory().view();
  const auto& ctrls_x             = traj_view.xs;
  const auto& ctrls_y             = traj_view.ys;

  // this shoudl never happen... but just in case.
  if (corridor.get_above_poly().get_degree() !=
      corridor.get_below_poly().get_degree()) {
    std::cerr << termcolor::yellow
              << "[MPCC] tube degrees do not match for above and below!"
              << Param::kNP << termcolor::reset << std::endl;
    return false;
  }

  int N_tube_coeffs = corridor.get_above_poly().get_coeffs().size();
  int provided_params =
      ctrls_x.size() + ctrls_y.size() + 2 * N_tube_coeffs + Param::kNP_Scalar;

  if (provided_params != Param::kNP) {
    std::cerr << termcolor::yellow << "[MPCC] provided param count "
              << provided_params << " does not match acados parameter size "
              << Param::kNP << termcolor::reset << std::endl;

    return false;
  }

  params[Param::k_Q_c]       = _mpc_cfg->weights.w_contour_e;
  params[Param::k_Q_l]       = _mpc_cfg->weights.w_lag_e;
  params[Param::k_Q_s]       = _mpc_cfg->weights.w_speed;
  params[Param::k_alpha_abv] = _mpc_cfg->cbf.alpha_abv;
  params[Param::k_alpha_blw] = _mpc_cfg->cbf.alpha_blw;
  params[Param::k_Ql_c]      = _mpc_cfg->clf.w_contour_e;
  params[Param::k_Ql_l]      = _mpc_cfg->clf.w_lag_e;
  params[Param::k_gamma]     = _mpc_cfg->clf.gamma;
  params[Param::k_L_path]    = traj_view.arclen;
  params[Param::k_V_max]     = _mpc_cfg->constraints.max_linvel;
  params[Param::k_A_max]     = _mpc_cfg->constraints.max_linacc;

  // params[Param::k_s_start]   = corridor.get_offset();

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

void DIMPCC::reset_horizon() {
  double mpc_steps = _mpc_cfg->steps;
  for (int i = 0; i < mpc_steps; ++i) {
    mpc_x[i]     = _state(kIndX);
    mpc_y[i]     = _state(kIndY);
    mpc_vx[i]    = 0;
    mpc_vy[i]    = 0;
    mpc_s[i]     = 1e-2;
    mpc_s_dot[i] = 0;
  }

  for (int i = 0; i < mpc_steps - 1; ++i) {
    mpc_ax[i]      = 0;
    mpc_ay[i]      = 0;
    mpc_s_ddots[i] = 0;
  }
}

Eigen::VectorXd DIMPCC::get_cbf_data(const types::Corridor& corridor,
                                     size_t horizon_idx) const {
  Eigen::VectorXd state = _prev_x0.segment(horizon_idx * kNX, kNX);
  Eigen::VectorXd input = _prev_u0.segment(horizon_idx * kNU, kNU);

  CasadiDoubleIntegratorInterface::Params params;
  params.qc_lyap    = _mpc_cfg->clf.w_contour_e;
  params.ql_lyap    = _mpc_cfg->clf.w_lag_e;
  params.gamma_lyap = _mpc_cfg->clf.gamma;

  CasadiDoubleIntegratorInterface casadi_interface;
  double h_abv = casadi_interface.get_h_abv(state, input, corridor, params);
  double hdot_abv =
      casadi_interface.get_h_dot_abv(state, input, corridor, params);

  double h_blw = casadi_interface.get_h_blw(state, input, corridor, params);
  double hdot_blw =
      casadi_interface.get_h_dot_blw(state, input, corridor, params);

  // std::cout << "h_abv is: " << h_abv << "\n";
  // std::cout << "h_dot_abv is: " << hdot_abv << "\n";
  // std::cout << "h_blw is: " << h_blw << "\n";
  // std::cout << "h_dot_blw is: " << hdot_blw << "\n";

  return Eigen::Vector4d(h_abv, hdot_abv, h_blw, hdot_blw);
}

Eigen::VectorXd DIMPCC::next_state(const Eigen::VectorXd& current_state,
                                   const Eigen::VectorXd& control) {
  Eigen::VectorXd ret(kNX);

  double x1    = current_state(kIndX);
  double y1    = current_state(kIndY);
  double vx1   = current_state(kIndVx);
  double vy1   = current_state(kIndVy);
  double s1    = current_state(kIndS);
  double sdot1 = current_state(kIndSDot);

  double ax    = control(kIndAx);
  double ay    = control(kIndAy);
  double sddot = control(kIndSDDot);

  double dt         = _mpc_cfg->dt;
  double max_linvel = _mpc_cfg->constraints.max_linvel;

  ret(kIndX)  = x1 + vx1 * dt;
  ret(kIndY)  = y1 + vy1 * dt;
  ret(kIndVx) = std::max(std::min(vx1 + ax * dt, max_linvel), -max_linvel);
  ret(kIndVy) = std::max(std::min(vy1 + ay * dt, max_linvel), -max_linvel);
  ret(kIndS)  = s1 + sdot1 * dt;
  ret(kIndSDot) =
      std::max(std::min(sdot1 + sddot * dt, max_linvel), -max_linvel);

  return ret;
}

Eigen::VectorXd DIMPCC::prepare_initial_state(const Eigen::VectorXd& state,
                                              const types::Corridor& corridor) {
  double eps         = 1e-6;
  Eigen::VectorXd x0 = state;
  // x0(kIndS)          = 0.;
  if (x0.segment(kIndVx, 2).norm() < eps) {
    const types::Trajectory ref = corridor.get_trajectory();
    Eigen::Vector2d tangent     = ref(eps, types::Trajectory::kFirstOrder);
    double theta                = atan2(tangent(1), tangent(0));
    Eigen::Vector2d unit_head(cos(theta), sin(theta));
    x0(kIndVx) = eps * unit_head(0);
    x0(kIndVy) = eps * unit_head(1);
  }

  double min_sdot = 0.1;
  x0(kIndSDot)    = std::max(x0.segment(kIndVx, 2).norm(), min_sdot);

  return x0;
}

std::array<double, 2> DIMPCC::compute_mpc_vel_command(
    const Eigen::VectorXd& state, const Eigen::VectorXd& u) {

  double dt         = _mpc_cfg->dt;
  double max_linacc = _mpc_cfg->constraints.max_linacc;
  double new_velx =
      limit(state[kIndVx], state[kIndVx] + u[kIndAx] * dt, max_linacc, dt);
  double new_vely =
      limit(state[kIndVy], state[kIndVy] + u[kIndAy] * dt, max_linacc, dt);

  // ensure velx and y are within input bounds
  double max_linvel = _mpc_cfg->constraints.max_linvel;
  new_velx          = std::max(std::min(new_velx, max_linvel), -max_linvel);
  new_vely          = std::max(std::min(new_vely, max_linvel), -max_linvel);

  return {new_velx, new_vely};
}

void DIMPCC::map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                       const Eigen::VectorXd& utraj) {
  double mpc_steps = _mpc_cfg->steps;
  for (int i = 0; i <= mpc_steps; ++i) {
    mpc_x[i]     = xtraj[kIndX + i * kIndStateInc];
    mpc_y[i]     = xtraj[kIndY + i * kIndStateInc];
    mpc_vx[i]    = xtraj[kIndVx + i * kIndStateInc];
    mpc_vy[i]    = xtraj[kIndVy + i * kIndStateInc];
    mpc_s[i]     = xtraj[kIndS + i * kIndStateInc];
    mpc_s_dot[i] = xtraj[kIndSDot + i * kIndStateInc];
  }

  for (int i = 0; i < mpc_steps; ++i) {
    mpc_ax[i]      = utraj[kIndAx + i * kIndInputInc];
    mpc_ay[i]      = utraj[kIndAy + i * kIndInputInc];
    mpc_s_ddots[i] = utraj[kIndSDDot + i * kIndInputInc];
  }
}

const std::array<Eigen::VectorXd, 2> DIMPCC::get_state_limits() const {
  Eigen::VectorXd xmin(kNX), xmax(kNX);
  double max_linvel = _mpc_cfg->constraints.max_linvel;
  xmin << -1e3, -1e3, -max_linvel, -max_linvel, 0, -max_linvel;
  xmax << 1e3, 1e3, max_linvel, max_linvel, _mpc_cfg->ref_length, max_linvel;

  return {xmin, xmax};
}

const std::array<Eigen::VectorXd, 2> DIMPCC::get_input_limits() const {
  Eigen::VectorXd umin(kNU), umax(kNU);
  double max_linacc = _mpc_cfg->constraints.max_linacc;
  umin << -max_linacc, -max_linacc, -max_linacc;
  umax << max_linacc, max_linacc, max_linacc;

  return {umin, umax};
}

DIMPCC::MPCHorizon DIMPCC::get_horizon() const {

  // sadly c++17 does not support designated initializers 😭
  double mpc_steps = _mpc_cfg->steps;
  DIMPCC::MPCHorizon horizon;
  horizon.states.xs          = utils::vector_to_eigen(mpc_x);
  horizon.states.ys          = utils::vector_to_eigen(mpc_y);
  horizon.states.vs_x        = utils::vector_to_eigen(mpc_vx);
  horizon.states.vs_y        = utils::vector_to_eigen(mpc_vy);
  horizon.states.arclens     = utils::vector_to_eigen(mpc_s);
  horizon.states.arclens_dot = utils::vector_to_eigen(mpc_s_dot);

  horizon.inputs.accs_x       = utils::vector_to_eigen(mpc_ax);
  horizon.inputs.accs_y       = utils::vector_to_eigen(mpc_ay);
  horizon.inputs.arclens_ddot = utils::vector_to_eigen(mpc_s_ddots);
  horizon.length              = mpc_steps + 1;

  const auto N = mpc_steps + 1;

  // these should all hold true by construction, mostly here for
  // future refactoring in case I screw something up down the line
  assert(horizon.states.xs.size() == N);
  assert(horizon.states.ys.size() == N);
  assert(horizon.states.vs_x.size() == N);
  assert(horizon.states.vs_y.size() == N);
  assert(horizon.states.arclens.size() == N);
  assert(horizon.states.arclens_dot.size() == N);

  assert(horizon.inputs.accs_x.size() == N - 1);
  assert(horizon.inputs.accs_y.size() == N - 1);
  assert(horizon.inputs.arclens_ddot.size() == N - 1);

  return horizon;
}

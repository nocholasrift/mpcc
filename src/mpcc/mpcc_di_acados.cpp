#include <mpcc/mpcc_di_acados.h>

#include <cmath>
#include <stdexcept>

using namespace mpcc;

#ifdef FOUND_PYBIND11
// the declarations in the header cannot be referenced by pybind...
// need to define them
const uint16_t DIMPCC::kNX;
const uint16_t DIMPCC::kNS;
const uint16_t DIMPCC::kNP;
const uint16_t DIMPCC::kNU;
const uint16_t DIMPCC::kNBX0;
#endif

DIMPCC::DIMPCC() {
  _dt         = -1;
  _max_linvel = 2.;
  _max_linacc = 2.;
  _mpc_steps  = 20;

  _w_ql      = 50.0;
  _w_qc      = .1;
  _w_q_speed = .3;
  _w_ql_lyap = 1;
  _w_qc_lyap = 1;

  _gamma     = .5;
  _use_cbf   = false;
  _alpha_abv = 1.0;
  _alpha_blw = 1.0;
  _colinear  = 0.01;
  _padding   = .05;

  _s_dot       = 0;
  _ref_samples = 10;
  _ref_len_sz  = 4.0;

  _has_run       = false;
  _solve_success = false;
  _is_shift_warm = false;
  _odom_init     = false;

  _new_time_steps = nullptr;

  _state = Eigen::VectorXd::Zero(kNX);

  _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
  _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);

  mpc_x.resize(_mpc_steps + 1);
  mpc_y.resize(_mpc_steps + 1);
  mpc_vx.resize(_mpc_steps + 1);
  mpc_vy.resize(_mpc_steps + 1);
  mpc_s.resize(_mpc_steps + 1);
  mpc_s_dot.resize(_mpc_steps + 1);

  mpc_ax.resize(_mpc_steps);
  mpc_ay.resize(_mpc_steps);
  mpc_s_ddots.resize(_mpc_steps);

  _acados_solver.initialize(_mpc_steps);
}

DIMPCC::~DIMPCC() {}

void DIMPCC::load_params(const std::map<std::string, double>& params) {
  _params = params;

  // Init parameters for MPC object
  _max_linvel = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL")
                                                        : _max_linvel;
  _max_linacc = _params.find("MAX_LINACC") != _params.end()
                    ? _params.at("MAX_LINACC")
                    : _max_linacc;

  _w_ql = params.find("W_LAG") != params.end() ? params.at("W_LAG") : _w_ql;
  _w_qc =
      params.find("W_CONTOUR") != params.end() ? params.at("W_CONTOUR") : _w_qc;
  _w_q_speed = params.find("W_SPEED") != params.end() ? params.at("W_SPEED")
                                                      : _w_q_speed;

  _ref_len_sz  = params.find("REF_LENGTH") != params.end()
                     ? params.at("REF_LENGTH")
                     : _ref_len_sz;
  _ref_samples = params.find("REF_SAMPLES") != params.end()
                     ? params.at("REF_SAMPLES")
                     : _ref_samples;

  _gamma     = params.find("CLF_GAMMA") != params.end() ? params.at("CLF_GAMMA")
                                                        : _gamma;
  _w_ql_lyap = params.find("CLF_W_LAG") != params.end() ? params.at("CLF_W_LAG")
                                                        : _w_ql_lyap;
  _w_qc_lyap = params.find("CLF_W_CONTOUR") != params.end()
                   ? params.at("CLF_W_CONTOUR")
                   : _w_ql_lyap;

  _use_cbf =
      params.find("USE_CBF") != params.end() ? params.at("USE_CBF") : _use_cbf;
  _alpha_abv = params.find("CBF_ALPHA_ABV") != params.end()
                   ? params.at("CBF_ALPHA_ABV")
                   : _alpha_abv;
  _alpha_blw = params.find("CBF_ALPHA_BLW") != params.end()
                   ? params.at("CBF_ALPHA_BLW")
                   : _alpha_blw;
  _colinear  = params.find("CBF_COLINEAR") != params.end()
                   ? params.at("CBF_COLINEAR")
                   : _colinear;
  _padding   = params.find("CBF_PADDING") != params.end()
                   ? params.at("CBF_PADDING")
                   : _padding;

  double new_dt = params.find("DT") != params.end() ? params.at("DT") : _dt;
  double new_steps =
      _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;

  _dt        = new_dt;
  _mpc_steps = new_steps;

  int status = _acados_solver.initialize(_mpc_steps);
  if (status) {
    throw std::runtime_error("Acados initialization failed with status + " +
                             std::to_string(status) + "!");
  }

  mpc_x.resize(_mpc_steps + 1);
  mpc_y.resize(_mpc_steps + 1);
  mpc_vx.resize(_mpc_steps + 1);
  mpc_vy.resize(_mpc_steps + 1);
  mpc_s.resize(_mpc_steps + 1);
  mpc_s_dot.resize(_mpc_steps + 1);

  mpc_ax.resize(_mpc_steps);
  mpc_ay.resize(_mpc_steps);
  mpc_s_ddots.resize(_mpc_steps);

  if (_prev_x0.size() != (_mpc_steps + 1) * kNX) {
    std::cout << termcolor::yellow
              << "x0, u0 size differs from mpc_steps size, resizing and "
                 "zeroing out\n";
    _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
    _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);
  }
}

void DIMPCC::reset_horizon() {
  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_x[i]     = _odom(0);
    mpc_y[i]     = _odom(1);
    mpc_vx[i]    = 0;
    mpc_vy[i]    = 0;
    mpc_s[i]     = 0;
    mpc_s_dot[i] = 0;
  }

  for (int i = 0; i < _mpc_steps - 1; ++i) {
    mpc_ax[i]      = 0;
    mpc_ay[i]      = 0;
    mpc_s_ddots[i] = 0;
  }
}

Eigen::VectorXd DIMPCC::get_cbf_data(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control,
                                     bool is_abv) const {
  return Eigen::Vector3d(0., 0., 0.);
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

  ret(kIndX)  = x1 + vx1 * _dt;
  ret(kIndY)  = y1 + vy1 * _dt;
  ret(kIndVx) = std::max(std::min(vx1 + ax * _dt, _max_linvel), -_max_linvel);
  ret(kIndVy) = std::max(std::min(vy1 + ay * _dt, _max_linvel), -_max_linvel);
  ret(kIndS)  = s1 + sdot1 * _dt;
  ret(kIndSDot) =
      std::max(std::min(sdot1 + sddot * _dt, _max_linvel), -_max_linvel);

  return ret;
}

Eigen::VectorXd DIMPCC::prepare_initial_state(const Eigen::VectorXd& state) {
  Eigen::VectorXd x0 = state;
  // x0(kIndS)          = 0.;
  if (x0.segment(2, 2).norm() < 1e-6) {
    x0(kIndVx) = 1e-6;
  }

  return x0;
}

std::array<double, 2> DIMPCC::compute_mpc_vel_command(
    const Eigen::VectorXd& state, const Eigen::VectorXd& u) {

  double new_velx =
      limit(state[kIndVx], state[kIndVx] + u[kIndAx] * _dt, _max_linacc, _dt);
  double new_vely =
      limit(state[kIndVy], state[kIndVy] + u[kIndAy] * _dt, _max_linacc, _dt);

  // ensure velx and y are within input bounds
  new_velx = std::max(std::min(new_velx, _max_linvel), -_max_linvel);
  new_vely = std::max(std::min(new_vely, _max_linvel), -_max_linvel);

  return {new_velx, new_vely};
}

void DIMPCC::map_trajectory_to_buffers(const Eigen::VectorXd& xtraj,
                                       const Eigen::VectorXd& utraj) {
  for (int i = 0; i <= _mpc_steps; ++i) {
    mpc_x[i]     = xtraj[kIndX + i * kIndStateInc];
    mpc_y[i]     = xtraj[kIndY + i * kIndStateInc];
    mpc_vx[i]    = xtraj[kIndVx + i * kIndStateInc];
    mpc_vy[i]    = xtraj[kIndVy + i * kIndStateInc];
    mpc_s[i]     = xtraj[kIndS + i * kIndStateInc];
    mpc_s_dot[i] = xtraj[kIndSDot + i * kIndStateInc];
  }

  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_ax[i]      = utraj[kIndAx + i * kIndInputInc];
    mpc_ay[i]      = utraj[kIndAy + i * kIndInputInc];
    mpc_s_ddots[i] = utraj[kIndSDDot + i * kIndInputInc];
  }
}

const std::array<Eigen::VectorXd, 2> DIMPCC::get_state_limits() const {
  Eigen::VectorXd xmin(kNX), xmax(kNX);
  xmin << -1e3, -1e3, -_max_linvel, -_max_linvel, 0, -_max_linvel;
  xmax << 1e3, 1e3, _max_linvel, _max_linvel, _ref_len_sz, _max_linvel;

  return {xmin, xmax};
}

const std::array<Eigen::VectorXd, 2> DIMPCC::get_input_limits() const {
  Eigen::VectorXd umin(kNU), umax(kNU);
  umin << -_max_linacc, -_max_linacc, -_max_linacc;
  umax << _max_linacc, _max_linacc, _max_linacc;

  return {umin, umax};
}

DIMPCC::MPCHorizon DIMPCC::get_horizon() const {

  // sadly c++17 does not support designated initializers 😭
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
  horizon.length              = _mpc_steps + 1;

  const auto N = _mpc_steps + 1;

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

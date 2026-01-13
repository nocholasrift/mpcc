#include <mpcc/mpcc_acados.h>

#include <array>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mpcc/termcolor.hpp>
#include <stdexcept>

using namespace mpcc;

#ifdef FOUND_PYBIND11
// the declarations in the header cannot be referenced by pybind...
// need to define them
const uint16_t MPCC::kNX;
const uint16_t MPCC::kNS;
const uint16_t MPCC::kNP;
const uint16_t MPCC::kNU;
const uint16_t MPCC::kNBX0;
#endif

MPCC::MPCC() {
  // Set default value
  _dt          = .1;
  _mpc_steps   = 10;
  _max_angvel  = 3.0;  // Maximal angvel radian (~30 deg)
  _max_linvel  = 2.0;  // Maximal linvel accel
  _max_linacc  = 4.0;  // Maximal linacc accel
  _max_anga    = 2 * M_PI;
  _bound_value = 1.0e3;  // Bound value for other variables

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

  _use_eigen   = false;
  _ref_samples = 10;
  _ref_len_sz  = 4.0;

  _has_run   = false;
  _odom_init = false;

  _acados_ocp_capsule = nullptr;
  _new_time_steps     = nullptr;

  _s_dot = 0;

  _state = Eigen::VectorXd::Zero(kNX);

  _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
  _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);

  mpc_x.resize(_mpc_steps);
  mpc_y.resize(_mpc_steps);
  mpc_theta.resize(_mpc_steps);
  mpc_linvels.resize(_mpc_steps);
  mpc_s.resize(_mpc_steps);
  mpc_s_dot.resize(_mpc_steps);

  mpc_angvels.resize(_mpc_steps - 1);
  mpc_linaccs.resize(_mpc_steps - 1);
  mpc_s_ddots.resize(_mpc_steps - 1);

  _use_dyna_obs  = false;
  _is_shift_warm = false;
  _solve_success = false;

  _odom = Eigen::VectorXd(3);

  // cpg_update_A_mat(0, -1.1);
}

MPCC::~MPCC() {
  if (_acados_ocp_capsule)
    delete _acados_ocp_capsule;

  if (_new_time_steps)
    delete _new_time_steps;
}

void MPCC::load_params(const std::map<std::string, double>& params) {
  _params = params;

  // Init parameters for MPC object
  _dt = params.find("DT") != params.end() ? params.at("DT") : _dt;
  _mpc_steps =
      _params.find("STEPS") != _params.end() ? _params.at("STEPS") : _mpc_steps;
  _max_angvel = _params.find("ANGVEL") != _params.end() ? _params.at("ANGVEL")
                                                        : _max_angvel;
  _max_linvel = _params.find("LINVEL") != _params.end() ? _params.at("LINVEL")
                                                        : _max_linvel;
  _max_linacc = _params.find("MAX_LINACC") != _params.end()
                    ? _params.at("MAX_LINACC")
                    : _max_linacc;
  _max_anga = _params.find("MAX_ANGA") != _params.end() ? _params.at("MAX_ANGA")
                                                        : _max_anga;

  _bound_value = _params.find("BOUND") != _params.end() ? _params.at("BOUND")
                                                        : _bound_value;

  _w_angvel   = params.find("W_ANGVEL") != params.end() ? params.at("W_ANGVEL")
                                                        : _w_angvel;
  _w_angvel_d = params.find("W_DANGVEL") != params.end()
                    ? params.at("W_DANGVEL")
                    : _w_angvel_d;
  _w_linvel_d =
      params.find("W_DA") != params.end() ? params.at("W_DA") : _w_linvel_d;

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

  _acados_ocp_capsule = unicycle_model_mpcc_acados_create_capsule();

  if (_new_time_steps)
    delete[] _new_time_steps;

  _acados_sim_capsule = unicycle_model_mpcc_acados_sim_solver_create_capsule();
  int status = unicycle_model_mpcc_acados_sim_create(_acados_sim_capsule);

  if (status) {
    throw std::runtime_error("acados_create() returned status " +
                             std::to_string(status) + ". Exiting.");
  }

  // acados sim
  _sim_in     = unicycle_model_mpcc_acados_get_sim_in(_acados_sim_capsule);
  _sim_out    = unicycle_model_mpcc_acados_get_sim_out(_acados_sim_capsule);
  _sim_dims   = unicycle_model_mpcc_acados_get_sim_dims(_acados_sim_capsule);
  _sim_config = unicycle_model_mpcc_acados_get_sim_config(_acados_sim_capsule);

  status = unicycle_model_mpcc_acados_create_with_discretization(
      _acados_ocp_capsule, _mpc_steps, _new_time_steps);

  if (status) {
    throw std::runtime_error(
        "unicycle_model_mpcc_acados_create() returned status " +
        std::to_string(status) + ". Exiting.");
  }

  // acados solver
  _nlp_in     = unicycle_model_mpcc_acados_get_nlp_in(_acados_ocp_capsule);
  _nlp_out    = unicycle_model_mpcc_acados_get_nlp_out(_acados_ocp_capsule);
  _nlp_opts   = unicycle_model_mpcc_acados_get_nlp_opts(_acados_ocp_capsule);
  _nlp_dims   = unicycle_model_mpcc_acados_get_nlp_dims(_acados_ocp_capsule);
  _nlp_solver = unicycle_model_mpcc_acados_get_nlp_solver(_acados_ocp_capsule);
  _nlp_config = unicycle_model_mpcc_acados_get_nlp_config(_acados_ocp_capsule);

  mpc_x.resize(_mpc_steps);
  mpc_y.resize(_mpc_steps);
  mpc_theta.resize(_mpc_steps);
  mpc_linvels.resize(_mpc_steps);
  mpc_s.resize(_mpc_steps);
  mpc_s_dot.resize(_mpc_steps);

  mpc_angvels.resize(_mpc_steps - 1);
  mpc_linaccs.resize(_mpc_steps - 1);
  mpc_s_ddots.resize(_mpc_steps - 1);

  _prev_x0 = Eigen::VectorXd::Zero((_mpc_steps + 1) * kNX);
  _prev_u0 = Eigen::VectorXd::Zero(_mpc_steps * kNU);

  std::cout << "!! MPC Obj parameters updated !! " << std::endl;
  std::cout << "!! ACADOS model instantiated !! " << std::endl;
}

Eigen::VectorXd MPCC::next_state(const Eigen::VectorXd& current_state,
                                 const Eigen::VectorXd& control) {
  Eigen::VectorXd ret(kNX);

  // Extracting current state values
  double x1     = current_state(0);
  double y1     = current_state(1);
  double theta1 = current_state(2);
  double v1     = current_state(3);
  double s1     = current_state(4);
  double sdot1  = current_state(5);

  // Extracting control inputs
  double a     = control(0);
  double w     = control(1);
  double sddot = control(2);

  // Dynamics equations
  ret(0) = x1 + v1 * cos(theta1) * _dt;
  ret(1) = y1 + v1 * sin(theta1) * _dt;
  ret(2) = theta1 + w * _dt;
  ret(3) = std::max(std::min(v1 + a * _dt, _max_linvel), -_max_linvel);
  ret(4) = s1 + sdot1 * _dt;
  ret(5) = std::max(std::min(sdot1 + sddot * _dt, _max_linvel), -_max_linvel);

  return ret;
}

void MPCC::warm_start_no_u(double* x_init) {
  double u_init[kNU];
  u_init[0] = 0.0;
  u_init[1] = 0.0;
  u_init[2] = 0.0;

  x_init[5] = x_init[3];

  for (int i = 0; i < _mpc_steps; ++i) {
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", x_init);
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u", u_init);
  }

  ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", x_init);
}

// From linear to nonlinear MPC: bridging the gap via the real-time iteration,
// Gros et. al.
void MPCC::warm_start_shifted_u(bool correct_perturb,
                                const Eigen::VectorXd& state) {
  double starting_s = _prev_x0[1 * kNX + 4];
  if (correct_perturb) {
    std::cout << termcolor::red << "[MPCC] Guess pos. too far, correcting"
              << termcolor::reset << std::endl;

    Eigen::VectorXd curr = state;

    // project forward previous control inputs, starting from true current
    // state
    for (int i = 0; i < _mpc_steps - 1; ++i) {
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "x", &curr[0]);
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i, "u",
                      &_prev_u0[(i + 1) * kNU]);
      curr = next_state(curr, _prev_u0.segment((i + 1) * kNU, kNU));
      // std::cout << curr.transpose() << std::endl;
    }

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x",
                    &curr[0]);
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                    &_prev_u0[(_mpc_steps - 1) * kNU]);

    curr = next_state(curr, _prev_u0.tail(kNU));
    // std::cout << curr.transpose() << std::endl;

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x",
                    &curr[0]);
    // exit(0);
  } else {
    for (int i = 1; i < _mpc_steps; ++i) {
      Eigen::VectorXd warm_state = _prev_x0.segment(i * kNX, kNX);
      warm_state(4) -= starting_s;

      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "x",
                      &warm_state[0]);
      ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, i - 1, "u",
                      &_prev_u0[i * kNU]);
    }

    Eigen::VectorXd xN_prev = _prev_x0.tail(kNX);
    xN_prev(4) -= starting_s;

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "x",
                    &xN_prev[0]);
    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps - 1, "u",
                    &_prev_u0[(_mpc_steps - 1) * kNU]);

    Eigen::VectorXd uN_prev = _prev_u0.tail(kNU);
    Eigen::VectorXd xN      = next_state(xN_prev, uN_prev);

    ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x", &xN[0]);
  }
}

bool MPCC::set_solver_parameters(
    const mpcc::types::Trajectory::View& reference) {
  double params[kNP];
  auto& ctrls_x = reference.xs;
  auto& ctrls_y = reference.ys;

  int num_params =
      ctrls_x.size() + ctrls_y.size() + _tubes[0].size() + _tubes[1].size() + 8;
  if (num_params != kNP) {
    std::cout << "[MPCC] reference size " << num_params
              << " does not match acados parameter size " << kNP << std::endl;
    return false;
  }

  params[kNP - 8] = _w_qc;
  params[kNP - 7] = _w_ql;
  params[kNP - 6] = _w_q_speed;
  params[kNP - 5] = _alpha_abv;
  params[kNP - 4] = _alpha_blw;
  params[kNP - 3] = _w_qc_lyap;
  params[kNP - 2] = _w_ql_lyap;
  params[kNP - 1] = _gamma;

  for (int i = 0; i < ctrls_x.size(); ++i) {
    params[i]                  = ctrls_x[i];
    params[i + ctrls_x.size()] = ctrls_y[i];
  }

  for (int i = 0; i < _tubes[0].size(); ++i) {
    params[i + 2 * ctrls_x.size()]                    = _tubes[0](i);
    params[i + 2 * ctrls_x.size() + _tubes[0].size()] = _tubes[1](i);
  }

  for (int i = 0; i < _mpc_steps + 1; ++i) {
    unicycle_model_mpcc_acados_update_params(_acados_ocp_capsule, i, params,
                                             kNP);
  }

  return true;
}

void MPCC::process_solver_output() {
  // stored as x0, y0,..., x1, y1, ..., xN, yN, ...
  Eigen::VectorXd xtraj((_mpc_steps + 1) * kNX);
  Eigen::VectorXd utraj(_mpc_steps * kNU);
  for (int i = 0; i < _mpc_steps; ++i) {
    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "x", &xtraj[i * kNX]);
    ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, i, "u", &utraj[i * kNU]);
  }

  ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, _mpc_steps, "x",
                  &xtraj[_mpc_steps * kNX]);

#ifdef UNICYCLE_MODEL_MPCC_NS
  Eigen::VectorXd slacks(kNS);
  // std::cout << "getting slacks " << NS << std::endl;
  ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, 1, "sl", &slacks[0]);

  std::cout << "[MPCC] Slack values are: " << slacks.transpose() << std::endl;
#endif

  _prev_x0 = xtraj;
  _prev_u0 = utraj;

  for (int i = 0; i <= _mpc_steps; ++i) {
    mpc_x[i]       = xtraj[kIndX + i * kIndStateInc];
    mpc_y[i]       = xtraj[kIndY + i * kIndStateInc];
    mpc_theta[i]   = xtraj[kIndTheta + i * kIndStateInc];
    mpc_linvels[i] = xtraj[kIndV + i * kIndStateInc];
    mpc_s[i]       = xtraj[kIndS + i * kIndStateInc];
    mpc_s_dot[i]   = xtraj[kIndSDot + i * kIndStateInc];
  }

  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_angvels[i] = utraj[kIndAngVel + i * kIndInputInc];
    mpc_linaccs[i] = utraj[kIndLinAcc + i * kIndInputInc];
    mpc_s_ddots[i] = utraj[kIndSDDot + i * kIndInputInc];
  }
}

void MPCC::reset_horizon() {
  for (int i = 0; i < _mpc_steps; ++i) {
    mpc_x[i]       = _odom(0);
    mpc_y[i]       = _odom(1);
    mpc_theta[i]   = 0;
    mpc_linvels[i] = 0;
    mpc_s[i]       = 0;
    mpc_s_dot[i]   = 0;
  }

  for (int i = 0; i < _mpc_steps - 1; ++i) {
    mpc_angvels[i] = 0;
    mpc_linaccs[i] = 0;
    mpc_s_ddots[i] = 0;
  }
}

std::array<double, 2> MPCC::solve(
    const Eigen::VectorXd& state,
    const mpcc::types::Trajectory::View& reference, bool is_reverse) {
  _solve_success = false;

  if (_tubes.size() == 0) {
    std::cout << "[MPCC] tubes are not set yet, mpc cannot run" << std::endl;
    return {0, 0};
  }

  /*************************************
  ********** INITIAL CONDITION *********
  **************************************/

  // _s_dot = std::min(std::max((s - _state(5)) / _dt, 0.), _max_linvel);
  if (state.size() != kNBX0) {
    std::cout << termcolor::yellow << "[MPCC] state sized passed has size "
              << state.size() << " but should be " << kNBX0 << termcolor::reset
              << std::endl;
    return {0, 0};
  }

  double lbx0[kNBX0];
  double ubx0[kNBX0];

  // Eigen::VectorXd x0(kNBX0);
  // x0 << state(0), state(1), state(2), state(3), 0, _s_dot;
  Eigen::VectorXd x0 = state;
  if (_has_run) {
    Eigen::VectorXd prev_x0 = _prev_x0.head(kNX);
    double etheta           = x0(2) - prev_x0(2);
    if (etheta > M_PI)
      x0(2) -= 2 * M_PI;
    if (etheta < -M_PI)
      x0(2) += 2 * M_PI;
  }

  memcpy(lbx0, &x0[0], kNBX0 * sizeof(double));
  memcpy(ubx0, &x0[0], kNBX0 * sizeof(double));

  ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "lbx",
                                lbx0);
  ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, 0, "ubx",
                                ubx0);

  /*************************************
  ********* INITIALIZE SOLUTION ********
  **************************************/

  // in our case kNX = kNBX0
  double x_init[kNX];
  memcpy(x_init, lbx0, kNX * sizeof(double));

  double u_init[kNU];
  u_init[kIndAngVel] = 0.0;
  u_init[kIndLinAcc] = 0.0;
  u_init[kIndSDDot]  = 0.0;

  // Eigen::Vector2d prev_pos = _prev_x0.head(2);
  Eigen::Vector2d prev_pos = _prev_x0.segment(kNX, 2);
  Eigen::Vector2d curr_pos = state.head(2);

  if (!_is_shift_warm)
    warm_start_no_u(x_init);
  else {
    // warm_start_shifted_u(false, x0);
    warm_start_shifted_u((prev_pos - curr_pos).norm() > 5e-2, x0);
  }

  /*************************************
  ********* SET REFERENCE PARAMS *******
  **************************************/

  if (!set_solver_parameters(reference))
    return {0, 0};

  /*************************************
  ************* RUN SOLVER *************
  **************************************/

  // double elapsed_time = 0.0;
  double timer;

  // run at most 2 times, if first fails, try with simple initialization
  for (int i = 0; i < 2; ++i) {
    // timer for acados using chrono
    auto start = std::chrono::high_resolution_clock::now();
    int status = unicycle_model_mpcc_acados_solve(_acados_ocp_capsule);
    auto end   = std::chrono::high_resolution_clock::now();
    // for some reason this causes problems in docker container, commenting
    // out for now
    // ocp_nlp_get(_nlp_config, _nlp_solver, "time_tot", &timer);
    // elapsed_time += timer;

    if (status == ACADOS_SUCCESS) {
      std::cout << "[MPCC] unicycle_model_mpcc_acados_solve(): SUCCESS! "
                << std::chrono::duration<double>(end - start).count() << "s"
                << std::endl;
      //          << elapsed_time * 1000 << std::endl;
      _is_shift_warm = true;
      _solve_success = true;
      break;
    } else {
      _is_shift_warm = false;
      std::cout << "[MPCC] unicycle_model_mpcc_acados_solve() failed with "
                   "status "
                << status << std::endl;
      std::cout << "[MPCC] using simple warm start procedure" << std::endl;
      std::cout << "[MPCC] xinit is: " << x0.transpose() << std::endl;
      warm_start_no_u(x_init);
    }
  }

  /*************************************
  *********** PROCESS OUTPUT ***********
  **************************************/

  double prev_angvel = _prev_u0[kIndAngVel];
  process_solver_output();
  std::cout << "mpc x[0] is " << _prev_x0.head(kNX).transpose() << std::endl;
  std::cout << "true x[0] is " << x0.transpose() << std::endl;
  std::cout << "mpc x[1] is " << _prev_x0.segment(kNX, kNX).transpose()
            << std::endl;

  _has_run = true;

  _state << _prev_x0[kIndX], _prev_x0[kIndY], _prev_x0[kIndTheta],
      _prev_x0[kIndV], 0, _prev_x0[kIndSDot];

  // std::array<double, 2> input = _cmd.getCommand();

  double curr_angvel = limit(prev_angvel, _prev_u0[kIndAngVel], _max_anga, _dt);
  double new_vel =
      limit(state[kIndV], state[kIndV] + _prev_u0[kIndLinAcc] * _dt,
            _max_linacc, _dt);

  _cmd = {new_vel, curr_angvel};

  std::cout << "[MPCC] commanded input is: " << curr_angvel << " " << new_vel
            << std::endl;

  std::cout << "[MPCC] commanded accel is: " << _prev_u0[kIndLinAcc] << "\n";

  if (!_solve_success) {
    std::cout << "[MPCC] SOLVER STATUS WAS INFEASIBLE!\n";
  }

  return _cmd;
}

const std::array<Eigen::VectorXd, 2> MPCC::get_state_limits() const {
  Eigen::VectorXd xmin(kNX), xmax(kNX);
  xmin << -_bound_value, -_bound_value, -M_PI, -_max_linvel, 0, -_max_linvel;
  xmax << _bound_value, _bound_value, M_PI, _max_linvel, _ref_len_sz,
      _max_linvel;

  return {xmin, xmax};
}

const std::array<Eigen::VectorXd, 2> MPCC::get_input_limits() const {
  Eigen::VectorXd umin(kNU), umax(kNU);
  umin << -_max_angvel, -_max_linacc, -_max_linacc;
  umax << _max_angvel, _max_linacc, _max_linacc;

  return {umin, umax};
}

Eigen::VectorXd MPCC::get_cbf_data(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& control,
                                   bool is_abv) const {
  return Eigen::Vector3d(0., 0., 0.);
}

void MPCC::compute_world_frame_velocities(Eigen::VectorXd& vs_x,
                                          Eigen::VectorXd& vs_y) const {
  assert(vs_x.size() == _mpc_steps);
  assert(vs_y.size() == _mpc_steps);

  for (int i = 0; i < _mpc_steps; ++i) {
    vs_x[i] = mpc_linvels[i] * cos(mpc_theta[i]);
    vs_y[i] = mpc_linvels[i] * sin(mpc_theta[i]);
  }
}

void MPCC::compute_world_frame_accelerations(Eigen::VectorXd& accs_x,
                                             Eigen::VectorXd& accs_y) const {
  assert(accs_x.size() == _mpc_steps - 1);
  assert(accs_y.size() == _mpc_steps - 1);

  for (int i = 0; i < _mpc_steps - 1; ++i) {
    accs_x[i] = mpc_linaccs[i] * cos(mpc_theta[i]);
    accs_y[i] = mpc_linaccs[i] * sin(mpc_theta[i]);
  }
}

MPCHorizon MPCC::get_horizon() const {

  MPCHorizon horizon;
  horizon.states.vs_x = Eigen::VectorXd(_mpc_steps);
  horizon.states.vs_y = Eigen::VectorXd(_mpc_steps);
  compute_world_frame_velocities(horizon.states.vs_x, horizon.states.vs_y);

  // ax ,ay have -1 size because they are inputs
  horizon.inputs.accs_x = Eigen::VectorXd(_mpc_steps - 1);
  horizon.inputs.accs_y = Eigen::VectorXd(_mpc_steps - 1);
  compute_world_frame_accelerations(horizon.inputs.accs_x,
                                    horizon.inputs.accs_y);

  horizon.states.xs          = utils::vector_to_eigen(mpc_x);
  horizon.states.ys          = utils::vector_to_eigen(mpc_y);
  horizon.states.arclens     = utils::vector_to_eigen(mpc_s);
  horizon.states.arclens_dot = utils::vector_to_eigen(mpc_s_dot);

  horizon.inputs.arclens_ddot = utils::vector_to_eigen(mpc_s_ddots);
  horizon.length              = _mpc_steps;

  const auto N = _mpc_steps;
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

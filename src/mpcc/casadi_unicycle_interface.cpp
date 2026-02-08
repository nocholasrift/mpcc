#include <mpcc/casadi_unicycle_interface.h>

namespace mpcc {

std::ostream& operator<<(std::ostream& os, const Eigen::VectorXd& vector) {
  os << "[";
  for (int i = 0; i < vector.size(); ++i) {
    os << vector[i];
    if (i < vector.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Eigen::RowVectorXd& vector) {
  os << "[";
  for (int i = 0; i < vector.size(); ++i) {
    os << vector[i];
    if (i < vector.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

double CasadiUnicycleInterface::get_h_abv(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiUnicycleInterface::Params& params) {
  const types::Trajectory& trajectory = corridor.get_trajectory();
  double traj_length                  = trajectory.get_arclen();

  const double* arg[kH_ABV_SZ_ARG];
  fill_arg_list(state, input, corridor, params, &traj_length, arg);

  double result = 0;
  double* res[kH_ABV_SZ_RES];
  res[0] = &result;

  std::vector<casadi_int> iw(kH_ABV_SZ_IW);
  std::vector<double> w(kH_ABV_SZ_W);

  unicycle_model_h_abv(arg, res, iw.data(), w.data(), 0);

  return result;
}

double CasadiUnicycleInterface::get_h_blw(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiUnicycleInterface::Params& params) {
  const types::Trajectory& trajectory = corridor.get_trajectory();
  double traj_length                  = trajectory.get_arclen();

  const double* arg[kH_BLW_SZ_ARG];
  fill_arg_list(state, input, corridor, params, &traj_length, arg);

  double result = 0;
  double* res[kH_BLW_SZ_RES];
  res[0] = &result;

  std::vector<casadi_int> iw(kH_BLW_SZ_IW);
  std::vector<double> w(kH_BLW_SZ_W);

  unicycle_model_h_blw(arg, res, iw.data(), w.data(), 0);

  return result;
}

double CasadiUnicycleInterface::get_h_dot_abv(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiUnicycleInterface::Params& params) {
  const types::Trajectory& trajectory = corridor.get_trajectory();
  double traj_length                  = trajectory.get_arclen();

  // LFH
  double lfh_abv = 0;
  {
    const double* arg[kLFH_ABV_SZ_ARG];
    fill_arg_list(state, input, corridor, params, &traj_length, arg);

    double* res[kLFH_ABV_SZ_RES];
    res[0] = &lfh_abv;

    std::vector<casadi_int> iw(kLFH_ABV_SZ_IW);
    std::vector<double> w(kLFH_ABV_SZ_W);

    unicycle_model_Lfh_abv(arg, res, iw.data(), w.data(), 0);
  }

  // LGH*u
  double lghu_abv = 0;
  {
    const double* arg[kLGHU_ABV_SZ_ARG];
    fill_arg_list(state, input, corridor, params, &traj_length, arg);

    double* res[kLGHU_ABV_SZ_RES];
    res[0] = &lghu_abv;

    std::vector<casadi_int> iw(kLGHU_ABV_SZ_IW);
    std::vector<double> w(kLGHU_ABV_SZ_W);

    unicycle_model_Lghu_abv(arg, res, iw.data(), w.data(), 0);
  }

  return lfh_abv + lghu_abv;
}

double CasadiUnicycleInterface::get_h_dot_blw(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiUnicycleInterface::Params& params) {
  const types::Trajectory& trajectory = corridor.get_trajectory();
  double traj_length                  = trajectory.get_arclen();

  // LFH
  double lfh_blw = 0;
  {
    const double* arg[kLFH_BLW_SZ_ARG];
    fill_arg_list(state, input, corridor, params, &traj_length, arg);

    double* res[kLFH_BLW_SZ_RES];
    res[0] = &lfh_blw;

    std::vector<casadi_int> iw(kLFH_BLW_SZ_IW);
    std::vector<double> w(kLFH_BLW_SZ_W);

    unicycle_model_Lfh_blw(arg, res, iw.data(), w.data(), 0);
  }

  // LGH*u
  double lghu_blw = 0;
  {
    const double* arg[kLGHU_BLW_SZ_ARG];
    fill_arg_list(state, input, corridor, params, &traj_length, arg);

    double* res[kLGHU_BLW_SZ_RES];
    res[0] = &lghu_blw;

    std::vector<casadi_int> iw(kLGHU_BLW_SZ_IW);
    std::vector<double> w(kLGHU_BLW_SZ_W);

    unicycle_model_Lghu_blw(arg, res, iw.data(), w.data(), 0);
  }

  return lfh_blw + lghu_blw;
}

void CasadiUnicycleInterface::fill_arg_list(const Eigen::VectorXd& state,
                                            const Eigen::VectorXd& input,
                                            const types::Corridor& corridor,
                                            const Params& params,
                                            const double* traj_length_ptr,
                                            const double** arg_list) {

  const types::Trajectory& trajectory = corridor.get_trajectory();

  arg_list[0] = state.data();
  arg_list[1] = input.data();
  arg_list[2] = trajectory.get_ctrls_x().data();
  arg_list[3] = trajectory.get_ctrls_y().data();
  arg_list[4] = corridor.get_tube_coeffs(types::Corridor::Side::kAbove).data();
  arg_list[5] = corridor.get_tube_coeffs(types::Corridor::Side::kBelow).data();
  arg_list[6] = &params.ql_lyap;
  arg_list[7] = &params.qc_lyap;
  arg_list[8] = &params.gamma_lyap;
  arg_list[9] = traj_length_ptr;

  /*Eigen::VectorXd state_t = state.transpose();*/
  /*Eigen::VectorXd input_t = input.transpose();*/
  /*std::cout << "state= " << state_t << "\n";*/
  /*std::cout << "input= " << input_t << "\n";*/
  /*std::cout << "xs=" << trajectory.get_ctrls_x() << "\n";*/
  /*std::cout << "ys=" << trajectory.get_ctrls_y() << "\n";*/
  /**/
  /*std::cout << "cabvs="*/
  /*          << corridor.get_tube_coeffs(types::Corridor::Side::kAbove) << "\n";*/
  /*std::cout << "cblw="*/
  /*          << corridor.get_tube_coeffs(types::Corridor::Side::kBelow) << "\n";*/
  /**/
  /*std::cout << "qcl= " << params.qc_lyap << "\n";*/
  /*std::cout << "qll= " << params.ql_lyap << "\n";*/
  /*std::cout << "gammal= " << params.gamma_lyap << "\n";*/
  /*std::cout << "traj_len= " << *traj_length_ptr << "\n";*/
}

}  // namespace mpcc

#include <mpcc/bicycle/casadi_bicycle_interface.h>
#include <vector>

namespace mpcc {

double CasadiBicycleInterface::get_h_abv(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiBicycleInterface::Params& params) {
  const types::Trajectory& trajectory = corridor.get_trajectory();
  double traj_length                  = trajectory.get_arclen();

  const double* arg[kH_ABV_SZ_ARG];
  fill_arg_list(state, input, corridor, params, &traj_length, arg);

  double result = 0;
  double* res[kH_ABV_SZ_RES];
  res[0] = &result;

  std::vector<casadi_int> iw(kH_ABV_SZ_IW);
  std::vector<double> w(kH_ABV_SZ_W);

  bicycle_h_abv(arg, res, iw.data(), w.data(), 0);

  return result;
}

double CasadiBicycleInterface::get_h_blw(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiBicycleInterface::Params& params) {
  const types::Trajectory& trajectory = corridor.get_trajectory();
  double traj_length                  = trajectory.get_arclen();

  const double* arg[kH_BLW_SZ_ARG];
  fill_arg_list(state, input, corridor, params, &traj_length, arg);

  double result = 0;
  double* res[kH_BLW_SZ_RES];
  res[0] = &result;

  std::vector<casadi_int> iw(kH_BLW_SZ_IW);
  std::vector<double> w(kH_BLW_SZ_W);

  bicycle_h_blw(arg, res, iw.data(), w.data(), 0);

  return result;
}

double CasadiBicycleInterface::get_h_dot_abv(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiBicycleInterface::Params& params) {
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

    bicycle_Lfh_abv(arg, res, iw.data(), w.data(), 0);
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

    bicycle_Lghu_abv(arg, res, iw.data(), w.data(), 0);
  }

  return lfh_abv + lghu_abv;
}

double CasadiBicycleInterface::get_h_dot_blw(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const types::Corridor& corridor,
    const CasadiBicycleInterface::Params& params) {
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

    bicycle_Lfh_blw(arg, res, iw.data(), w.data(), 0);
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

    bicycle_Lghu_blw(arg, res, iw.data(), w.data(), 0);
  }

  return lfh_blw + lghu_blw;
}

void CasadiBicycleInterface::fill_arg_list(const Eigen::VectorXd& state,
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
}

}  // namespace mpcc

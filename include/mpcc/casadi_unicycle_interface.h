#ifndef MPCC_CASADI_UNICYCLE_INTERFACE_H
#define MPCC_CASADI_UNICYCLE_INTERFACE_H

#include <mpcc/types.h>

#include <Eigen/Core>

// header generated from casadi automatically at compile time
#include "casadi_unicycle_model_mpcc_internals.h"

namespace mpcc{

class CasadiUnicycleInterface{

public:
  struct Params{
    double qc_lyap{0};
    double ql_lyap{0};
    double gamma_lyap{0};
  };

  CasadiUnicycleInterface() = default;

  double get_h_abv(const Eigen::VectorXd& state, const Eigen::VectorXd& input, const types::Corridor& corridor, const Params& params);
  double get_h_blw(const Eigen::VectorXd& state, const Eigen::VectorXd& input, const types::Corridor& corridor, const Params& params);

  double get_h_dot_abv(const Eigen::VectorXd& state, const Eigen::VectorXd& input, const types::Corridor& corridor, const Params& params);
  double get_h_dot_blw(const Eigen::VectorXd& state, const Eigen::VectorXd& input, const types::Corridor& corridor, const Params& params);

private:
  void fill_arg_list(const Eigen::VectorXd& state, const Eigen::VectorXd& input, const types::Corridor& corridor, const Params& params, const double* traj_length_ptr, const double** arg_list);

  static constexpr unsigned int kH_ABV_SZ_ARG = unicycle_model_h_abv_SZ_ARG;
  static constexpr unsigned int kH_ABV_SZ_RES = unicycle_model_h_abv_SZ_RES;
  static constexpr unsigned int kH_ABV_SZ_IW = unicycle_model_h_abv_SZ_IW;
  static constexpr unsigned int kH_ABV_SZ_W = unicycle_model_h_abv_SZ_W;

  static constexpr unsigned int kH_BLW_SZ_ARG = unicycle_model_h_blw_SZ_ARG;
  static constexpr unsigned int kH_BLW_SZ_RES = unicycle_model_h_blw_SZ_RES;
  static constexpr unsigned int kH_BLW_SZ_IW = unicycle_model_h_blw_SZ_IW;
  static constexpr unsigned int kH_BLW_SZ_W = unicycle_model_h_blw_SZ_W;

  static constexpr unsigned int kLFH_ABV_SZ_ARG = unicycle_model_Lfh_abv_SZ_ARG;
  static constexpr unsigned int kLFH_ABV_SZ_RES = unicycle_model_Lfh_abv_SZ_RES;
  static constexpr unsigned int kLFH_ABV_SZ_IW = unicycle_model_Lfh_abv_SZ_IW;
  static constexpr unsigned int kLFH_ABV_SZ_W = unicycle_model_Lfh_abv_SZ_W;

  static constexpr unsigned int kLFH_BLW_SZ_ARG = unicycle_model_Lfh_blw_SZ_ARG;
  static constexpr unsigned int kLFH_BLW_SZ_RES = unicycle_model_Lfh_blw_SZ_RES;
  static constexpr unsigned int kLFH_BLW_SZ_IW = unicycle_model_Lfh_blw_SZ_IW;
  static constexpr unsigned int kLFH_BLW_SZ_W = unicycle_model_Lfh_blw_SZ_W;

  static constexpr unsigned int kLGHU_ABV_SZ_ARG = unicycle_model_Lghu_abv_SZ_ARG;
  static constexpr unsigned int kLGHU_ABV_SZ_RES = unicycle_model_Lghu_abv_SZ_RES;
  static constexpr unsigned int kLGHU_ABV_SZ_IW = unicycle_model_Lghu_abv_SZ_IW;
  static constexpr unsigned int kLGHU_ABV_SZ_W = unicycle_model_Lghu_abv_SZ_W;

  static constexpr unsigned int kLGHU_BLW_SZ_ARG = unicycle_model_Lghu_blw_SZ_ARG;
  static constexpr unsigned int kLGHU_BLW_SZ_RES = unicycle_model_Lghu_blw_SZ_RES;
  static constexpr unsigned int kLGHU_BLW_SZ_IW = unicycle_model_Lghu_blw_SZ_IW;
  static constexpr unsigned int kLGHU_BLW_SZ_W = unicycle_model_Lghu_blw_SZ_W;
};
}

#endif

#ifndef MPCC_COMMON_MPCC_CONFIG_HPP
#define MPCC_COMMON_MPCC_CONFIG_HPP

#include <cmath>
#include <string>

#include "mpcc/common/types.h"

namespace mpcc {

struct CostWeights {
  double w_vel       = 1.0;
  double w_angvel    = 1.0;
  double w_linvel    = 1.0;
  double w_angvel_d  = 1.0;
  double w_linvel_d  = 0.5;
  double w_etheta    = 0.5;
  double w_cte       = 1.0;
  double w_lag_e     = 50.0;
  double w_contour_e = 0.1;
  double w_speed     = 0.3;
};

struct Constraints {
  double max_angvel  = 3.0;
  double max_linvel  = 2.0;
  double max_linacc  = 3.0;
  double max_angacc  = 2 * M_PI;
  double bound_value = 1.0e19;
};

struct CBFConfig {
  bool use_cbf         = false;
  double alpha_abv     = 0.5;
  double alpha_blw     = 0.5;
  double colinear      = 0.1;
  double padding       = 0.1;
  bool dynamic_alpha   = false;
  double min_alpha     = 0.1;
  double max_alpha     = 5.0;
  double min_alpha_dot = -3.0;
  double max_alpha_dot = 3.0;
  double min_h_val     = -100.0;
  double max_h_val     = 100.0;
};

struct CLFConfig {
  double w_lag_e     = 1.0;
  double w_contour_e = 1.0;
  double gamma       = 0.5;
};

struct TubeConfig {
  int poly_degree  = 6;
  int num_samples  = 50;
  double max_width = 2.0;
};

struct PropControllerConfig {
  double gain        = 0.5;
  double gain_thresh = 30.0 * M_PI / 180.0;
};

struct MPCConfig {
  int steps          = 10;
  double dt          = 0.1;
  int ref_samples    = 10;
  double ref_length  = 4.0;
  MPCType input_type = MPCType::kDoubleIntegrator;

  CostWeights weights;
  Constraints constraints;
  CBFConfig cbf;
  CLFConfig clf;
  TubeConfig tube;
  PropControllerConfig prop;
};

}  // namespace mpcc

#endif

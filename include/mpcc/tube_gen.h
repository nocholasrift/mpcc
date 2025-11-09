#include <mpcc/types.h>

#include <Eigen/Core>
#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_cv/grid_map_cv.hpp>

#include <iostream>

extern "C" {
#include <cpg_solve.h>
#include <cpg_workspace.h>
}

#include "Highs.h"

namespace tube_utils {
/**********************************************************************
 * Function: raycast_grid
 * Description: Casts a ray from a start point in a direction and
 * stores the distance to the first obstacle in the grid map
 * Parameters:
 * @param start: Eigen::Vector2d
 * @param dir: Eigen::Vector2d
 * @param grid_map: const grid_map::GridMap&
 * @param max_dist: double
 * @param actual_dist: double&
 * Returns:
 * bool - true if successful, false otherwise
 * Notes:
 * Will return false if start or end indices not in map
 **********************************************************************/
inline bool raycast_grid(const Eigen::Vector2d& start,
                         const Eigen::Vector2d& dir,
                         const grid_map::GridMap& grid_map, double max_dist,
                         double& actual_dist) {
  // get indices in map
  grid_map::Index start_ind;
  if (!grid_map.getIndex(start, start_ind))
    return false;

  // raycast several times with small purturbations in direction to ensure
  // thin obstacles are detected
  double min_dist  = 1e6;
  double theta_dir = atan2(dir(1), dir(0));

  for (int i = 0; i < 1; ++i) {
    // purturb by degrees each time
    double theta = theta_dir + i * M_PI / 180;
    Eigen::Vector2d end =
        start + max_dist * Eigen::Vector2d(cos(theta), sin(theta));

    grid_map::Index end_ind;
    if (!grid_map.getIndex(end, end_ind))
      return false;

    // raycast
    Eigen::Vector2d ray_end = end;
    for (grid_map::LineIterator iterator(grid_map, start_ind, end_ind);
         !iterator.isPastEnd(); ++iterator) {
      Eigen::Vector2d tmp;
      if (grid_map.at("layer", *iterator) > .5) {
        // I'm pretty sure this isn't possible
        if (!grid_map.getPosition(*iterator, ray_end))
          return false;

        break;
      }
    }

    double dist = (start - ray_end).norm();
    if (dist < min_dist)
      min_dist = dist;
  }

  actual_dist = min_dist;
  return true;
}

/**********************************************************************
 * Function: setup_lp
 * Description: Sets up the linear program for the CPG solver
 * Parameters:
 * @param d: int
 * @param N: int
 * @param len_start: double
 * @param traj_arc_len: double
 * @param min_dist: double
 * @param dist_vec: const std::vector<double>&
 * Returns:
 * N/A
 * Notes:
 * This function sets up the linear program for the CPG solver, for
 * more details, see Differentiable Collision-Free Parametric Corridors
 * by J. Arrizabalaga, et al.
 **********************************************************************/
inline void setup_lp(int d, int N, double len_start, double traj_arc_len,
                     double min_dist, const std::vector<double>& dist_vec) {
  // set arc length domain for LP
  /*cpg_update_Domain(0, 0);*/
  /*cpg_update_Domain(1, traj_arc_len);*/

  double ds = traj_arc_len / (N - 1);

  for (int i = 0; i < N; ++i) {
    double s   = i * ds;
    double s_k = 1;
    for (int j = 0; j <= d; ++j) {
      // matrices are in column-major order for cpg
      // index for -polynomial <= -min_dist constraint
      size_t ind_upper = i + j * 2 * N;
      // index for polynomial <= obs_dist constraint
      size_t ind_lower = i + N + j * 2 * N;

      cpg_update_A_mat(ind_upper, -s_k);
      cpg_update_A_mat(ind_lower, s_k);
      s_k *= s;
    }

    // index for -polynomial <= -min_dist constraint
    cpg_update_b_vec(i, -min_dist);
    // cpg_update_b_vec(i, 0);
    // index for polynomial <= obs_dist constraint
    cpg_update_b_vec(i + N, dist_vec[i]);
  }
}

inline bool get_distances(const std::array<Spline1D, 2>& traj, int N,
                          double max_dist, double len_start, double horizon,
                          const grid_map::GridMap& grid_map,
                          double& min_dist_abv, double& min_dist_blw,
                          std::vector<double>& ds_above,
                          std::vector<double>& ds_below) {

  ds_above.resize(N);
  ds_below.resize(N);

  double ds    = horizon / (N - 1);
  min_dist_abv = 1e6;
  min_dist_blw = 1e6;

  for (int i = 0; i < N; ++i) {
    double s  = len_start + i * ds;
    double px = traj[0](s).coeff(0);
    double py = traj[1](s).coeff(0);

    double tx = traj[0].derivatives(s, 1).coeff(1);
    double ty = traj[1].derivatives(s, 1).coeff(1);

    // normals are not stable in Eigen, calculate manually
    double curvature, nx, ny;
    if (i < N - 1) {
      double sp     = s + 1e-1;
      double txp    = traj[0].derivatives(sp, 1).coeff(1);
      double typ    = traj[1].derivatives(sp, 1).coeff(1);
      double theta1 = atan2(ty, tx);
      double theta2 = atan2(typ, txp);
      curvature     = (theta2 - theta1) / 1e-1;

      if (theta2 - theta1 > 0) {
        nx = -ty;
        ny = tx;
      } else {
        nx = ty;
        ny = -tx;
      }
    } else {
      double sp     = s - 1e-1;
      double txp    = traj[0].derivatives(sp, 1).coeff(1);
      double typ    = traj[1].derivatives(sp, 1).coeff(1);
      double theta1 = atan2(ty, tx);
      double theta2 = atan2(typ, txp);
      curvature     = (theta1 - theta2) / 1e-1;

      if (theta1 - theta2 > 0) {
        nx = -ty;
        ny = tx;
      } else {
        nx = ty;
        ny = -tx;
      }
    }

    // double nx = traj[0].derivatives(s, 2).coeff(2);
    // double ny = traj[1].derivatives(s, 2).coeff(2);

    Eigen::Vector2d point(px, py);
    Eigen::Vector2d normal(-ty, tx);
    normal.normalize();

    // if (Eigen::Vector2d(-ty, tx).dot(normal) < 0) normal *= -1;
    // Eigen::Vector2d normal(-ty, tx);
    // normal.normalize();

    // double den       = tx * tx + ty * ty;
    // double curvature = fabs(tx * ny - ty * nx) / (den * sqrt(den));  // normal.norm();

    // raycast in direction of normal to find obs dist
    // std::cout << "RAYCASTING ABOVE!" << std::endl;
    double dist_above;
    if (!raycast_grid(point, normal, grid_map, max_dist, dist_above))
      return false;

    /*std::cout << "raycast distance for " << point.transpose() << " is " << dist_above*/
    /*          << std::endl;*/

    // std::cout << "RAYCASTING BELOW!" << std::endl;
    double dist_below;
    if (!raycast_grid(point, -1 * normal, grid_map, max_dist, dist_below))
      return false;

    if (curvature > 1e-1 && curvature > 1 / (2 * max_dist)) {
      Eigen::Vector2d n_vec(nx, ny);
      Eigen::Vector2d abv_n_vec(-ty, tx);

      if (n_vec.dot(abv_n_vec) > 0)
        dist_above = std::min(dist_above, 1 / (2 * curvature));
      else
        dist_below = std::min(dist_below, 1 / (2 * curvature));
    }

    if (dist_above < min_dist_abv)
      min_dist_abv = dist_above;

    if (dist_below < min_dist_blw)
      min_dist_blw = dist_below;

    // check if odom above or below traj, ensure odom
    // is within tube
    /*if (i == 0) {*/
    /*  Eigen::Vector2d pos = odom.head(2);*/
    /*  Eigen::Vector2d dp  = pos - point;*/
    /*  double dot_prod     = dp.dot(normal);*/
    /*  double odom_dist    = dp.norm();*/
    /**/
    /*  if (dot_prod > 0 && dist_above < odom_dist) {*/
    /*    std::cout << "forcefully setting dist_above to be " << dp.norm()*/
    /*              << std::endl;*/
    /*    dist_above = dp.norm();*/
    /*  } else if (dot_prod < 0 && dist_below < odom_dist) {*/
    /*    std::cout << "forcefully setting dist_below to be " << dp.norm()*/
    /*              << std::endl;*/
    /*    dist_below = dp.norm();*/
    /*  }*/
    /*}*/

    ds_above[i] = dist_above;
    ds_below[i] = dist_below;
  }

  return true;
}

/**********************************************************************
 * Function: get_tubes
 * Description: Generates the upper and lower tubes for a trajectory
 * Parameters:
 * @param d: int
 * @param N: int
 * @param max_dist: double
 * @param traj: const std::array<Spline1D, 2>&
 * @param traj_arc_len: double
 * @param len_start: double
 * @param horizon: double
 * @param grid_map: const grid_map::GridMap&
 * @param tubes: std::array<Eigen::VectorXd, 2>&
 * Returns:
 * bool - true if successful, false otherwise
 * Notes:
 * This function generates the upper and lower tubes for a trajectory
 * using the CPG solver. For more details, see Differentiable Collision-Free
 * Parametric Corridors by J. Arrizabalaga, et al. Each tube is
 * represented as a polynomial of degree d, parameterized by arc len.
 **********************************************************************/
inline bool get_tubes(int d, int N, double max_dist,
                      const std::array<Spline1D, 2>& traj, double traj_arc_len,
                      double len_start, double horizon,
                      const Eigen::VectorXd& odom,
                      const grid_map::GridMap& grid_map,
                      std::array<Eigen::VectorXd, 2>& tubes) {
  /*************************************
    ********* Get Traj Distances *********
    **************************************/
  // double ds = traj_arc_len / (N-1);
  double min_dist_abv;
  double min_dist_blw;
  std::vector<double> ds_above;
  std::vector<double> ds_below;
  bool status = get_distances(traj, N, max_dist, len_start, horizon, grid_map,
                              min_dist_abv, min_dist_blw, ds_above, ds_below);

  if (!status) {
    std::cerr << "[Tube Gen] Failed to get map distances\n";
  }

  /*************************************
    ********** Setup & Solve ***********
    **************************************/

  setup_lp(d, N, len_start, horizon, min_dist_abv / 1.1, ds_above);
  // setup_lp(d, N, horizon, 0, ds_above);
  cpg_solve();

  /*std::cout << "solved!" << std::endl;*/

  std::string solved_str = "solved";
  // std::string status = CPG_Info.status;
  // if(strcmp(CPG_Info.status, solved_str.c_str()) != 0)
  // if (status.find(solved_str) == std::string::npos)
  if (CPG_Info.status != 0) {
    std::cerr << "[Tube Gen] LP Above Tube Failed: " << CPG_Info.status << "\n";
    tubes[0] = Eigen::VectorXd(d);
    tubes[1] = Eigen::VectorXd(d);
    return false;
  }

  std::cout << "abv coeffs:\n";
  for (int i = 0; i < d; ++i)
    std::cout << CPG_Result.prim->coeffs[i] << ", ";
  std::cout << "\n";

  Eigen::VectorXd upper_coeffs;
  upper_coeffs.resize(d);
  bool is_straight = true;
  for (int i = 0; i < d; ++i) {
    double val = CPG_Result.prim->coeffs[i];
    if (i == 0 && fabs(1 - val) > 1e-1)
      is_straight = false;
    else if (fabs(val) > 1e-1)
      is_straight = false;

    upper_coeffs[i] = CPG_Result.prim->coeffs[i];
  }

  /*************************************
    ********* Setup & Solve Down *********
    **************************************/

  /*std::cout << "min_dist_blw is: " << min_dist_blw << std::endl;*/
  setup_lp(d, N, len_start, horizon, min_dist_blw / 1.1, ds_below);
  // setup_lp(d, N, horizon, 0, ds_below);
  cpg_solve();

  // if(strcmp(CPG_Info.status, solved_str.c_str()) != 0)
  // if (status.find(solved_str) == std::string::npos)
  if (CPG_Info.status != 0) {
    std::cerr << "[Tube Gen] LP Below Tube Failed: " << CPG_Info.status << "\n";
    tubes[0] = Eigen::VectorXd(d);
    tubes[1] = Eigen::VectorXd(d);
    return false;
  }

  /*for (int i = 0; i < d; ++i)*/
  /*  std::cout << CPG_Result.prim->coeffs[i] << ", ";*/
  /*std::cout << std::endl;*/

  /*std::cout << "dist_below= [";*/
  /*for (int i = 0; i < N; ++i)*/
  /*  std::cout << ds_below[i] << ", ";*/
  /*std::cout << "];" << std::endl;*/

  Eigen::VectorXd lower_coeffs;
  lower_coeffs.resize(d);
  for (int i = 0; i < d; ++i)
    lower_coeffs[i] = CPG_Result.prim->coeffs[i];

  tubes[0] = upper_coeffs;
  tubes[1] = -1 * lower_coeffs;

  // ecos_workspace = 0;

  return true;
}

inline void setup_highs_model(HighsModel& model, int d, int N,
                              double traj_arc_len, double horizon,
                              double min_dist,
                              const std::vector<double>& dist_vec) {
  if (dist_vec.size() != N) {
    std::cerr << "[Tube Gen] Distance vector does not equal sample size N\n";
    return;
  }

  // setup cost function
  std::vector<double> cost_coeffs;
  cost_coeffs.resize(d + 1);

  double h_pow = 1.0;
  for (int i = 0; i < d + 1; ++i) {
    cost_coeffs[i] = -h_pow / (i + 1);
    h_pow *= horizon;
  }

  model.lp_.num_col_  = d + 1;
  model.lp_.num_row_  = N;
  model.lp_.sense_    = ObjSense::kMinimize;
  model.lp_.offset_   = 0.;
  model.lp_.col_cost_ = cost_coeffs;

  model.lp_.col_lower_.resize(d + 1);
  model.lp_.col_upper_.resize(d + 1);
  for (int i = 0; i <= d; ++i) {
    model.lp_.col_lower_[i] = -1.0e30;
    model.lp_.col_upper_[i] = 1.0e30;
  }

  model.lp_.row_lower_.resize(N);
  model.lp_.row_upper_.resize(N);
  for (int i = 0; i < N; ++i) {
    model.lp_.row_lower_[i] = min_dist;
    model.lp_.row_upper_[i] = dist_vec[i];
  }

  // slower but for now will just follow highs tutorial
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model.lp_.a_matrix_.start_.resize(d + 2);
  model.lp_.a_matrix_.index_.resize(N * (d + 1) - d);
  model.lp_.a_matrix_.value_.resize(N * (d + 1) - d);

  double ds = traj_arc_len / (N - 1);

  size_t idx                    = 0;
  model.lp_.a_matrix_.start_[0] = 0;
  for (int j = 0; j <= d; ++j) {
    model.lp_.a_matrix_.start_[j + 1] = (j + 1) * N - j;

    for (int i = 0; i < N; ++i) {

      if (j == 0)
        model.lp_.a_matrix_.value_[idx] = 1;

      if (j > 0 && i == 0)
        continue;

      model.lp_.a_matrix_.index_[idx] = i;

      if (j > 0)
        model.lp_.a_matrix_.value_[idx] =
            model.lp_.a_matrix_.value_[idx - N] * i * ds;

      ++idx;
    }
  }
}

inline bool get_coeffs(int d, int N, double traj_arc_len, double horizon,
                       double min_dist, const std::vector<double>& dists,
                       Eigen::VectorXd& coeffs) {

  HighsModel model;
  setup_highs_model(model, d, N, traj_arc_len, horizon, min_dist, dists);

  /*std::cout << "highs\n";*/
  Highs highs;
  HighsStatus return_status = highs.passModel(model);
  if (return_status != HighsStatus::kOk)
    return false;

  /*std::cout << "get lp\n";*/
  const HighsLp& lp = highs.getLp();

  return_status = highs.run();

  if (return_status != HighsStatus::kOk) {
    std::cerr << "[Tube Gen] Highs solver could not find tubes\n";
    return false;
  }

  /*std::cout << "model status\n";*/
  const HighsModelStatus& model_status = highs.getModelStatus();
  if (model_status != HighsModelStatus::kOptimal) {
    std::cerr << "[Tube Gen] Warning: model status was not optimal"
              << highs.modelStatusToString(model_status) << "\n";
  }

  std::cout << "solution:\n";
  coeffs.resize(lp.num_col_);
  const HighsSolution& solution = highs.getSolution();
  for (int col = 0; col < lp.num_col_; ++col) {
    coeffs[col] = solution.col_value[col];
    std::cout << solution.col_value[col] << ", ";
  }
  std::cout << "\n";
  /*highs.resetGlobalScheduler(true);*/

  /*std::cout << "done getting coeffs\n";*/

  return true;
}

inline bool get_tubes2(int d, int N, double max_dist,
                       const std::array<Spline1D, 2>& traj, double traj_arc_len,
                       double len_start, double horizon,
                       const Eigen::VectorXd& odom,
                       const grid_map::GridMap& grid_map,
                       std::array<Eigen::VectorXd, 2>& tubes) {

  // get distances
  double min_dist_abv;
  double min_dist_blw;
  std::vector<double> ds_above;
  std::vector<double> ds_below;
  bool status = get_distances(traj, N, max_dist, len_start, horizon, grid_map,
                              min_dist_abv, min_dist_blw, ds_above, ds_below);

  if (!get_coeffs(d, N, traj_arc_len, horizon, min_dist_abv, ds_above,
                  tubes[0])) {
    return false;
  }

  if (!get_coeffs(d, N, traj_arc_len, horizon, min_dist_blw, ds_below,
                  tubes[1])) {
    return false;
  }

  tubes[1] *= -1;

  return true;
}

}  // namespace tube_utils

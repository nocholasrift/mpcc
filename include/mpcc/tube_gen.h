#include <mpcc/types.h>
#include <mpcc/utils.h>

#include <Eigen/Core>
#ifdef FOUND_CATKIN
#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#else
#include <mpcc/map_util.h>
#endif

#include <iostream>

#include "Highs.h"

namespace tube_utils {

struct Distances {
  Eigen::VectorXd d_parallel_p;
  Eigen::VectorXd perp_dists_p;
  Eigen::VectorXd d_parallel_n;
  Eigen::VectorXd perp_dists_n;
};
typedef struct Distances distances_t;

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
#ifdef FOUND_CATKIN
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

      if (grid_map.at("layer", *iterator) > 90) {
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

  /*actual_dist = std::max(min_dist - 0.25, 0.);*/
  actual_dist = min_dist;
  return true;
}
#endif

#ifdef FOUND_CATKIN
inline bool get_distances(const std::array<Spline1D, 2>& traj, int N,
                          double max_dist, double len_start, double horizon,
                          const grid_map::GridMap& grid_map,
                          double& min_dist_abv, double& min_dist_blw,
                          std::vector<double>& ds_above,
                          std::vector<double>& ds_below) {
#else
inline bool get_distances(const std::array<Spline1D, 2>& traj, int N,
                          double max_dist, double len_start, double horizon,
                          const map_util::OccupancyGrid& grid_map,
                          double& min_dist_abv, double& min_dist_blw,
                          std::vector<double>& ds_above,
                          std::vector<double>& ds_below) {

#endif

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
    double dist_above;
#ifdef FOUND_CATKIN
    if (!raycast_grid(point, normal, grid_map, max_dist, dist_above))
      return false;
#else
    {
      Eigen::Vector2d end;
      Eigen::Vector2d max_end = point + normal * max_dist;
      grid_map.raycast(point, max_end, end, "inflated");
      dist_above = std::min((point - end).norm(), max_dist);
    }
#endif

    // std::cout << "RAYCASTING BELOW!" << std::endl;
    double dist_below;
#ifdef FOUND_CATKIN
    if (!raycast_grid(point, -1 * normal, grid_map, max_dist, dist_below))
      return false;
#else
    {
      Eigen::Vector2d end;
      Eigen::Vector2d max_end = point - normal * max_dist;
      grid_map.raycast(point, max_end, end, "inflated");
      dist_below = std::min((point - end).norm(), max_dist);
    }
#endif

    /*if (curvature > 1e-1 && curvature > 1 / (2 * max_dist)) {*/
    /*  Eigen::Vector2d n_vec(nx, ny);*/
    /*  Eigen::Vector2d abv_n_vec(-ty, tx);*/
    /**/
    /*  if (n_vec.dot(abv_n_vec) > 0)*/
    /*    dist_above = std::min(dist_above, 1 / (2 * curvature));*/
    /*  else*/
    /*    dist_below = std::min(dist_below, 1 / (2 * curvature));*/
    /*}*/

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

inline void setup_highs_model(HighsModel& model, int d, int N,
                              double traj_arc_len, double horizon,
                              double min_dist, double max_dist,
                              const std::vector<double>& dist_vec) {
  if (dist_vec.size() != N) {
    std::cerr << "[Tube Gen] Distance vector does not equal sample size N\n";
    return;
  }

  if (horizon < 1e-1) {
    std::cerr << "[Tube Gen] Horizon too small! " << horizon << "\n";
    return;
  }
  // figure out how many samples more we need to reach traj_arc_len
  double ds             = horizon / (N - 1);
  int num_extra_samples = (traj_arc_len - horizon) / ds;

  // setup cost function
  std::vector<double> cost_coeffs;
  cost_coeffs.resize(d + 1);

  double h_pow = 1.0;
  for (int i = 0; i < d + 1; ++i) {
    cost_coeffs[i] = -h_pow / (i + 1);
    h_pow *= horizon;
  }

  model.lp_.num_col_  = d + 1;
  model.lp_.num_row_  = N + num_extra_samples;
  model.lp_.sense_    = ObjSense::kMinimize;
  model.lp_.offset_   = 0.;
  model.lp_.col_cost_ = cost_coeffs;

  model.lp_.col_lower_.resize(d + 1);
  model.lp_.col_upper_.resize(d + 1);
  for (int i = 0; i <= d; ++i) {
    model.lp_.col_lower_[i] = -1.0e30;
    model.lp_.col_upper_[i] = 1.0e30;
  }

  model.lp_.row_lower_.resize(N + num_extra_samples);
  model.lp_.row_upper_.resize(N + num_extra_samples);
  for (int i = 0; i < N; ++i) {
    model.lp_.row_lower_[i] = min_dist;
    model.lp_.row_upper_[i] = dist_vec[i];
  }
  for (int i = N; i < N + num_extra_samples; ++i) {
    model.lp_.row_lower_[i] = min_dist;
    model.lp_.row_upper_[i] = max_dist;
  }

  // at this point now we can just increment N
  N += num_extra_samples;

  // slower but for now will just follow highs tutorial
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model.lp_.a_matrix_.start_.resize(d + 2);
  model.lp_.a_matrix_.index_.resize(N * (d + 1) - d);
  model.lp_.a_matrix_.value_.resize(N * (d + 1) - d);

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
                       double min_dist, double max_dist,
                       const std::vector<double>& dists,
                       Eigen::VectorXd& coeffs) {

  HighsModel model;
  setup_highs_model(model, d, N, traj_arc_len, horizon, min_dist, max_dist,
                    dists);

  /*std::cout << "highs\n";*/
  Highs highs;
  highs.setOptionValue("output_flag", false);
  HighsStatus return_status = highs.passModel(model);
  if (return_status != HighsStatus::kOk) {
    std::cerr << "[Tube Gen] Highs solver could not be properly setup!\n";
    return false;
  }

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

  // std::cout << "solution:\n";
  coeffs.resize(lp.num_col_);
  const HighsSolution& solution = highs.getSolution();
  for (int col = 0; col < lp.num_col_; ++col) {
    coeffs[col] = solution.col_value[col];
    // std::cout << solution.col_value[col] << ", ";
  }
  // std::cout << "\n";
  /*highs.resetGlobalScheduler(true);*/

  /*std::cout << "done getting coeffs\n";*/

  return true;
}

#ifdef FOUND_CATKIN
inline bool get_tubes2(int d, int N, double max_dist,
                       const std::array<Spline1D, 2>& traj, double traj_arc_len,
                       double len_start, double horizon,
                       const grid_map::GridMap& grid_map,
                       std::array<Eigen::VectorXd, 2>& tubes) {
#else
inline bool get_tubes2(int d, int N, double max_dist,
                       const Eigen::VectorXd& x_pts,
                       const Eigen::VectorXd& y_pts, int degree,
                       const Eigen::VectorXd& knot_parameters,
                       double traj_arc_len, double len_start, double horizon,
                       const map_util::OccupancyGrid& grid_map,
                       std::vector<Eigen::VectorXd>& tubes) {

  Spline1D splineX(utils::Interp(x_pts, degree, knot_parameters));
  Spline1D splineY(utils::Interp(y_pts, degree, knot_parameters));

  tubes.resize(2);
  std::array<Spline1D, 2> traj{splineX, splineY};
#endif

  // get distances
  double min_dist_abv;
  double min_dist_blw;
  std::vector<double> ds_above;
  std::vector<double> ds_below;

  bool status = get_distances(traj, N, max_dist, len_start, horizon, grid_map,
                              min_dist_abv, min_dist_blw, ds_above, ds_below);

  if (!get_coeffs(d, N, traj_arc_len, horizon, min_dist_abv, max_dist, ds_above,
                  tubes[0])) {
    std::cout << "[TUBE_GEN] get_coeffs above returned with some error\n";
    return false;
  }

  if (!get_coeffs(d, N, traj_arc_len, horizon, min_dist_blw, max_dist, ds_below,
                  tubes[1])) {
    std::cout << "[TUBE_GEN] get_coeffs below returned with some error\n";
    return false;
  }

  tubes[1] *= -1;

  return true;
}

}  // namespace tube_utils

#include <mpcc/types.h>
#include <mpcc/utils.h>

#include <mpcc/map_util.h>
#include <Eigen/Core>

#include <iostream>

#include "Highs.h"

namespace mpcc::tube {

using Trajectory = mpcc::types::Trajectory;

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

inline bool get_distances(const Trajectory& traj, int N, double max_dist,
                          double len_start, double horizon,
                          const map_util::OccupancyGrid& grid_map,
                          double& min_dist_abv, double& min_dist_blw,
                          std::vector<double>& ds_above,
                          std::vector<double>& ds_below) {

  ds_above.resize(N);
  ds_below.resize(N);

  double ds    = horizon / (N - 1);
  min_dist_abv = 1e6;
  min_dist_blw = 1e6;

  for (int i = 0; i < N; ++i) {
    double s = len_start + i * ds;

    Eigen::VectorXd point = traj(s);
    double px             = point(0);
    double py             = point(1);

    Eigen::VectorXd tangent = traj(s, Trajectory::kFirstOrder);
    double tx               = tangent(0);
    double ty               = tangent(1);

    // normals are not stable in Eigen, calculate manually
    double curvature, nx, ny;
    if (i < N - 1) {
      double sp = s + 1e-1;

      Eigen::VectorXd tangent_plus_eps = traj(sp, Trajectory::kFirstOrder);
      double txp                       = tangent_plus_eps(0);
      double typ                       = tangent_plus_eps(1);

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
      double sp = s - 1e-1;

      Eigen::VectorXd tangent_plus_eps = traj(sp, Trajectory::kFirstOrder);
      double txp                       = tangent_plus_eps(0);
      double typ                       = tangent_plus_eps(1);

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

    Eigen::Vector2d normal(-ty, tx);
    normal.normalize();

    double dist_above;
    {
      Eigen::Vector2d end;
      Eigen::Vector2d max_end = point + normal * max_dist;
      grid_map.raycast(point, max_end, end, "inflated");
      dist_above = std::min((point - end).norm(), max_dist);
    }

    // std::cout << "RAYCASTING BELOW!" << std::endl;
    double dist_below;
    {
      Eigen::Vector2d end;
      Eigen::Vector2d max_end = point - normal * max_dist;
      grid_map.raycast(point, max_end, end, "inflated");
      dist_below = std::min((point - end).norm(), max_dist);
    }

    if (dist_above < min_dist_abv)
      min_dist_abv = dist_above;

    if (dist_below < min_dist_blw)
      min_dist_blw = dist_below;

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

inline bool get_tube_coeffs(int d, int N, double traj_arc_len, double horizon,
                            double min_dist, double max_dist,
                            const std::vector<double>& dists,
                            Eigen::VectorXd& coeffs) {

  HighsModel model;
  setup_highs_model(model, d, N, traj_arc_len, horizon, min_dist, max_dist,
                    dists);

  Highs highs;
  highs.setOptionValue("output_flag", false);
  HighsStatus return_status = highs.passModel(model);
  if (return_status != HighsStatus::kOk) {
    // for (int i = 0; i < dists.size(); ++i) {
    //   std::cout << dists[i] << " ";
    // }
    //
    // std::cout << "\n";
    std::cerr << "[Tube Gen] Highs solver could not be properly setup!\n";
    return false;
  }

  const HighsLp& lp = highs.getLp();

  return_status = highs.run();

  if (return_status != HighsStatus::kOk) {
    std::cerr << "[Tube Gen] Highs solver could not find tubes\n";
    return false;
  }

  const HighsModelStatus& model_status = highs.getModelStatus();
  if (model_status != HighsModelStatus::kOptimal) {
    std::cerr << "[Tube Gen] Warning: model status was not optimal"
              << highs.modelStatusToString(model_status) << "\n";
  }

  // std::cout << "coeffs: ";
  coeffs.resize(lp.num_col_);
  const HighsSolution& solution = highs.getSolution();
  for (int col = 0; col < lp.num_col_; ++col) {
    coeffs[col] = solution.col_value[col];
    // std::cout << coeffs[col] << " ";
  }
  // std::cout << "\n";
  /*highs.resetGlobalScheduler(true);*/

  /*std::cout << "done getting coeffs\n";*/

  return true;
}

inline bool construct_tubes(int d, int N, double max_dist,
                            const Trajectory& traj, double len_start,
                            double horizon,
                            const map_util::OccupancyGrid& grid_map,
                            std::array<types::Polynomial, 2>& tubes) {

  // get distances
  double min_dist_abv;
  double min_dist_blw;
  std::vector<double> ds_above;
  std::vector<double> ds_below;

  bool status = get_distances(traj, N, max_dist, len_start, horizon, grid_map,
                              min_dist_abv, min_dist_blw, ds_above, ds_below);

  double traj_arc_len = traj.get_extended_length();
  Eigen::VectorXd abv_coeffs, blw_coeffs;
  if (!get_tube_coeffs(d, N, traj_arc_len, horizon, min_dist_abv, max_dist,
                       ds_above, abv_coeffs)) {
    std::cout << "[TUBE_GEN] get_tube_coeffs above returned with some error\n";
    return false;
  }

  if (!get_tube_coeffs(d, N, traj_arc_len, horizon, min_dist_blw, max_dist,
                       ds_below, blw_coeffs)) {
    std::cout << "[TUBE_GEN] get_tube_coeffs below returned with some error\n";
    return false;
  }

  // std::cout << "above coeffs: " << abv_coeffs.transpose() << "\n";
  // std::cout << "below coeffs: " << blw_coeffs.transpose() << "\n";
  tubes[0].set_coeffs(abv_coeffs);
  tubes[1].set_coeffs(-1 * blw_coeffs);

  return true;
}

}  // namespace mpcc::tube

#ifndef MPCC_TUBE_GEN_H
#define MPCC_TUBE_GEN_H

#include <mpcc/types.h>
#include <mpcc/utils.h>

#include <mpcc/map_util.h>
#include <Eigen/Core>

#include <iostream>
#include <optional>
#include <stdexcept>

#include "HConst.h"
#include "Highs.h"

namespace mpcc::tube {

using Trajectory = mpcc::types::Trajectory;

class HighsSolver {
 public:
  HighsSolver()  = default;
  ~HighsSolver() = default;

  void set_verbose(bool is_verbose) {
    highs_instance_.setOptionValue("output_flag", false);
  }

  void set_num_cols(unsigned int n_cols) {
    model_.lp_.num_col_ = n_cols;
    model_.lp_.col_lower_.resize(n_cols);
    model_.lp_.col_upper_.resize(n_cols);
  }

  void set_num_rows(unsigned int n_rows) {
    model_.lp_.num_row_ = n_rows;
    model_.lp_.row_lower_.resize(n_rows);
    model_.lp_.row_upper_.resize(n_rows);
  }

  void set_cost_coeffs(const std::vector<double>& coeffs,
                       const ObjSense& objective_type, double offset = 0) {
    model_.lp_.col_cost_ = coeffs;
    model_.lp_.sense_    = objective_type;
    model_.lp_.offset_   = offset;
  }

  void set_coeff_bounds(const std::vector<double> mins,
                        const std::vector<double>& maxs) {
    unsigned int n_cols = model_.lp_.num_col_;
    if (n_cols != mins.size()) {
      throw std::runtime_error(
          "[HighsSolver] min vector size " + std::to_string(mins.size()) +
          " does not match number of columns " + std::to_string(n_cols));
    }

    if (n_cols != maxs.size()) {
      throw std::runtime_error(
          "[HighsSolver] max vector size " + std::to_string(mins.size()) +
          " does not match number of columns " + std::to_string(n_cols));
    }

    model_.lp_.col_lower_.resize(n_cols);
    model_.lp_.col_upper_.resize(n_cols);

    for (int i = 0; i < n_cols; ++i) {
      model_.lp_.col_lower_[i] = mins[i];
      model_.lp_.col_upper_[i] = maxs[i];
    }
  }

  void set_constraint_bounds(unsigned int start_ind, unsigned int end_ind,
                             double min_val,
                             const std::vector<double>& max_vals,
                             double offset = 0) {

    for (unsigned int i = start_ind; i < end_ind; ++i) {
      model_.lp_.row_lower_[i] = min_val + offset;
      model_.lp_.row_upper_[i] = max_vals[i] + offset;
    }
  }

  void set_constraint_bounds(unsigned int start_ind, unsigned int end_ind,
                             double min_val, double max_val,
                             double offset = 0) {

    for (unsigned int i = start_ind; i < end_ind; ++i) {
      model_.lp_.row_lower_[i] = min_val + offset;
      model_.lp_.row_upper_[i] = max_val + offset;
    }
  }

  void set_matrices(double ds) {

    DenseMatrix dense_mat = compute_dense_matrix(ds);
    unsigned int n_cols   = model_.lp_.num_col_;
    unsigned int n_rows   = model_.lp_.num_row_;

    model_.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model_.lp_.a_matrix_.start_.resize(n_cols + 1);
    model_.lp_.a_matrix_.index_.resize(dense_mat.num_non_zeros);
    model_.lp_.a_matrix_.value_.resize(dense_mat.num_non_zeros);

    unsigned int idx               = 0;
    model_.lp_.a_matrix_.start_[0] = 0;
    for (int j = 0; j < n_cols; ++j) {

      unsigned int non_zeros = 0;
      for (int i = 0; i < n_rows; ++i) {

        if (std::abs(dense_mat[i][j]) > kHighsZero) {
          model_.lp_.a_matrix_.value_[idx]   = dense_mat[i][j];
          model_.lp_.a_matrix_.index_[idx++] = i;
          ++non_zeros;
        }
        // if (j == 0)
        //   model_.lp_.a_matrix_.value_[N * j + i] = (i + 1) * ds;
        // else
        //   model_.lp_.a_matrix_.value_[N * j + i] =
        //       model_.lp_.a_matrix_.value_[N * (j - 1) + i] * (i + 1) * ds;
      }

      model_.lp_.a_matrix_.start_[j + 1] =
          model_.lp_.a_matrix_.start_[j] + non_zeros;
    }
  }

  // void set_matrices(unsigned int d, unsigned int N, double ds) {
  //   model_.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  //   model_.lp_.a_matrix_.start_.resize(d + 2);
  //   model_.lp_.a_matrix_.index_.resize(N * (d + 1) - d);
  //   model_.lp_.a_matrix_.value_.resize(N * (d + 1) - d);
  //
  //   size_t idx                     = 0;
  //   model_.lp_.a_matrix_.start_[0] = 0;
  //   for (int j = 0; j <= d; ++j) {
  //     model_.lp_.a_matrix_.start_[j + 1] = (j + 1) * N - j;
  //
  //     for (int i = 0; i < N; ++i) {
  //
  //       if (j == 0)
  //         model_.lp_.a_matrix_.value_[idx] = 1;
  //
  //       if (j > 0 && i == 0)
  //         continue;
  //
  //       model_.lp_.a_matrix_.index_[idx] = i;
  //
  //       if (j > 0)
  //         model_.lp_.a_matrix_.value_[idx] =
  //             model_.lp_.a_matrix_.value_[idx - N] * i * ds;
  //
  //       ++idx;
  //     }
  //   }
  // }

  [[nodiscard]] HighsStatus solve(std::vector<double>& solution) {
    HighsStatus status = highs_instance_.passModel(model_);
    if (status != HighsStatus::kOk) {
      std::cerr << "[Tube Gen] Highs solver could not be properly setup!\n";
      return status;
    }

    const HighsLp& lp = highs_instance_.getLp();
    status            = highs_instance_.run();
    if (status != HighsStatus::kOk) {
      std::cerr << "[Tube Gen] Highs solver could not find tubes\n";
      return status;
    }

    const HighsModelStatus& model_status = highs_instance_.getModelStatus();
    if (model_status != HighsModelStatus::kOptimal) {
      std::cerr << "[Tube Gen] Warning: model status was not optimal"
                << highs_instance_.modelStatusToString(model_status) << "\n";
    }

    solution.resize(lp.num_col_);
    const HighsSolution& sol = highs_instance_.getSolution();
    for (int col = 0; col < lp.num_col_; ++col) {
      solution[col] = sol.col_value[col];
    }

    return HighsStatus::kOk;
  }

 public:
  static constexpr double kINF = 1e30;

 private:
  struct DenseMatrix {
    std::vector<std::vector<double>> full_matrix;
    unsigned int num_non_zeros{0};
    unsigned int rows{0};
    unsigned int cols{0};

    unsigned int size() const { return rows * cols; }

    std::vector<double>& operator[](unsigned int ind) {
      return full_matrix[ind];
    }
  };

  // TODO: rework iteration to be more cache friendly
  DenseMatrix compute_dense_matrix(double ds) {
    unsigned int non_zeros = 0;
    unsigned int n_rows    = model_.lp_.num_row_;
    unsigned int n_cols    = model_.lp_.num_col_;
    std::vector<std::vector<double>> full_matrix(n_rows,
                                                 std::vector<double>(n_cols));
    for (int c = 0; c < n_cols; ++c) {
      for (int r = 0; r < n_rows; ++r) {
        if (c == 0)
          full_matrix[r][c] = 1;
        else if (r == 0)
          full_matrix[r][c] = 0;
        else
          full_matrix[r][c] = full_matrix[r][c - 1] * r * ds;

        if (std::abs(full_matrix[r][c]) < kHighsZero) {
          full_matrix[r][c] = 0.0;
        } else {
          ++non_zeros;
        }
      }
    }

    DenseMatrix dense_mat;
    dense_mat.full_matrix   = std::move(full_matrix);
    dense_mat.rows          = n_rows;
    dense_mat.cols          = n_cols;
    dense_mat.num_non_zeros = non_zeros;

    return dense_mat;
  }

 private:
  static constexpr double kHighsZero = 1e-9;

  HighsModel model_;
  Highs highs_instance_;
};

class TubeGenerator {
 public:
  enum class Side { kAbove, kBelow };

  struct Settings {
    unsigned int degree{0};
    unsigned int num_samples{0};
    double max_distance{0};
  };

  TubeGenerator() = default;
  TubeGenerator(const Settings& settings)
      : degree_(settings.degree),
        num_samples_(settings.num_samples),
        max_distance_(settings.max_distance) {}

  void update_settings(const Settings& settings) {
    degree_       = settings.degree;
    num_samples_  = settings.num_samples;
    max_distance_ = settings.max_distance;

    setup_solver_dimensions();
  }

  ~TubeGenerator() = default;

  [[nodiscard]] bool generate(
      const map_util::IGrid& grid_map, const Trajectory& traj, double len_start,
      double horizon, const std::optional<Settings>& settings = std::nullopt) {

    if (settings.has_value()) {
      update_settings(settings.value());
    }

    double min_d_abv{-1}, min_d_blw{-1};
    std::vector<double> d_blws, d_abvs;
    bool dist_status = get_distances(grid_map, traj, len_start, horizon,
                                     min_d_abv, min_d_blw, d_abvs, d_blws);

    if (!dist_status) {
      return false;
    }

    // these are all shared per solve (abv and blw)
    set_solver_cost();
    set_coeff_bounds();
    setup_solver_matrices();

    // only constraints change per solve
    // set_constraint_bounds(settings, traj, min_d_abv, d_abvs, -d_abvs[0]);
    set_constraint_bounds(min_d_abv, d_abvs);

    std::vector<double> coeffs_abv;
    HighsStatus status = solver_.solve(coeffs_abv);
    if (status != HighsStatus::kOk) {
      return false;
    }

    // only constraints change per solve
    // set_constraint_bounds(settings, traj, min_d_blw, d_blws, -d_blws[0]);
    set_constraint_bounds(min_d_blw, d_blws);

    std::vector<double> coeffs_blw;
    status = solver_.solve(coeffs_blw);
    if (status != HighsStatus::kOk) {
      return false;
    }

    // coeffs_abv.insert(coeffs_abv.begin(), d_abvs[0]);
    // coeffs_blw.insert(coeffs_blw.begin(), d_blws[0]);

    Eigen::VectorXd horizon_scale(degree_ + 1);

    // horizon_scale[0] = 1;
    // for (int i = 1; i < horizon_scale.size(); ++i) {
    //   horizon_scale[i] = horizon_scale[i - 1] * traj.get_extended_length();
    // }

    Eigen::VectorXd eigen_coeffs_abv =
        Eigen::Map<Eigen::VectorXd>(coeffs_abv.data(), coeffs_abv.size());

    // eigen_coeffs_abv = eigen_coeffs_abv.cwiseQuotient(horizon_scale);

    Eigen::VectorXd negative_coeffs_blw =
        -1 * Eigen::Map<Eigen::VectorXd>(coeffs_blw.data(), coeffs_blw.size());

    // negative_coeffs_blw = negative_coeffs_blw.cwiseQuotient(horizon_scale);

    // dont love using two different constructors here but dont hate it enough to
    // change coeffs_abv call...
    abv_ = types::Polynomial(eigen_coeffs_abv);
    blw_ = types::Polynomial(negative_coeffs_blw);

    // Eigen::VectorXd cof = Eigen::VectorXd::Zero(degree_ + 1);
    // cof(0)              = 100;
    // abv_                = types::Polynomial(cof);
    // blw_                = types::Polynomial(-1 * cof);

    shifted_abv_ = abv_;
    shifted_blw_ = blw_;

    return true;
  }

  const types::Polynomial& get_side_poly(Side side) const {
    switch (side) {
      case Side::kAbove:
        return shifted_abv_;

      // Side can only be one of the two...
      case Side::kBelow:
      default:
        return shifted_blw_;
    }
  }

  // std::array<types::Polynomial, 2> get_boundary() const { return {abv_, blw_}; }
  std::array<types::Polynomial, 2> get_boundary() const {
    return {shifted_abv_, shifted_blw_};
  }

  void set_tubes(const types::Polynomial& abv, const types::Polynomial& blw) {
    abv_ = abv;
    blw_ = blw;
  }

  void set_verbose(bool is_verbose) { solver_.set_verbose(is_verbose); }

  /*void shift_tube_domain(double len_start, double horizon,*/
  /*                       types::Polynomial& abv, types::Polynomial& blw) {*/
  /*  Eigen::VectorXd domain = Eigen::VectorXd::LinSpaced(num_samples_, len_start,*/
  /*                                                      len_start + horizon);*/
  /*  abv = types::Polynomial::polyfit(domain.array() - len_start, abv_(domain),*/
  /*                                   degree_);*/
  /*  blw = types::Polynomial::polyfit(domain.array() - len_start, blw_(domain),*/
  /*                                   degree_);*/
  /**/
  /*  shifted_abv_ = abv;*/
  /*  shifted_blw_ = blw;*/
  /*}*/

  [[nodiscard]] bool get_distances(const map_util::IGrid& grid_map,
                                   const Trajectory& traj, double start,
                                   double horizon, double& min_dist_abv,
                                   double& min_dist_blw,
                                   std::vector<double>& dists_abv,
                                   std::vector<double>& dists_blw) const {
    using TrajSide = types::Trajectory::Side;

    // get distances above and below
    double end = start + horizon;

    bool dist_status =
        grid_map.get_distances(traj, num_samples_, start, end, max_distance_,
                               TrajSide::kAbove, min_dist_abv, dists_abv);

    if (!dist_status) {
      return false;
    }

    // std::cout << "d_abv:\n";
    for (auto& dist : dists_abv) {
      if (dist < 0) {
        dist = max_distance_;
      }
      // std::cout << dist << ", ";
    }
    // std::cout << "\n";

    dist_status =
        grid_map.get_distances(traj, num_samples_, start, end, max_distance_,
                               TrajSide::kBelow, min_dist_blw, dists_blw);
    if (!dist_status) {
      return false;
    }

    // std::cout << "d_blw:\n";
    for (auto& dist : dists_blw) {
      if (dist < 0) {
        dist = max_distance_;
      }
      // std::cout << dist << ", ";
    }
    // std::cout << "\n";

    return dist_status;
  }

 private:
  void setup_solver_dimensions() {

    // wasteful to resize the solver vectors when typically these params
    // stay constant across solver iterations.
    solver_.set_num_cols(degree_ + 1);
    solver_.set_num_rows(num_samples_);
  }

  void set_solver_cost(double offset = 0) {
    // degree + 1 because there are d+1 coeffs in d degree poly.
    std::vector<double> cost_coeffs;
    cost_coeffs.resize(degree_ + 1);

    double discount_factor = 0.25;
    double multiplier      = 1.0;
    for (int i = 0; i <= degree_; ++i) {
      cost_coeffs[i] = multiplier * -1.0 / (i + 1);
      multiplier *= discount_factor;
    }
    cost_coeffs[0] *= 3;

    solver_.set_cost_coeffs(cost_coeffs, ObjSense::kMinimize, offset);
  }

  void set_coeff_bounds() {
    std::vector<double> mins(degree_ + 1, -HighsSolver::kINF);
    std::vector<double> maxs(degree_ + 1, HighsSolver::kINF);
    // std::vector<double> mins(settings.degree, -HighsSolver::kINF);
    // std::vector<double> maxs(settings.degree, HighsSolver::kINF);

    solver_.set_coeff_bounds(mins, maxs);
  }

  void set_constraint_bounds(double min_d, const std::vector<double>& dists,
                             double offset = 0) {

    unsigned int start_ind = 0;
    solver_.set_constraint_bounds(start_ind, num_samples_, min_d, dists,
                                  offset);
  }

  void setup_solver_matrices() {

    // unsigned int num_extra_samples =
    //     (traj.get_extended_length() - settings.horizon) / ds;

    // solver_.set_matrices(ds);
    solver_.set_matrices(1.0 / (num_samples_ - 1));
  }

 private:
  unsigned int degree_{0};
  unsigned int num_samples_{0};
  double max_distance_{0};

  HighsSolver solver_;
  types::Polynomial abv_, blw_;
  types::Polynomial shifted_abv_, shifted_blw_;
};

}  // namespace mpcc::tube

#endif

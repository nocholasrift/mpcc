#pragma once

#include <mpcc/spline.h>
#include <iostream>

#include <stdexcept>
#include <unsupported/Eigen/Splines>

namespace mpcc {
namespace types {

class SplineWrapper {
 public:
  tk::spline spline;  // Expose the tk::spline
};

using StateHorizon = struct StateHorizon {
  Eigen::VectorXd xs;
  Eigen::VectorXd ys;
  // Eigen::VectorXd vs_x;
  // Eigen::VectorXd vs_y;
  Eigen::VectorXd arclens;
  Eigen::VectorXd arclens_dot;
};

using InputHorizon = struct InputHorizon {
  // Eigen::VectorXd accs_x;
  // Eigen::VectorXd accs_y;
  Eigen::VectorXd arclens_ddot;
};

template <typename Derived>
struct MPCHorizon {
  typename Derived::StateHorizon states;
  typename Derived::InputHorizon inputs;
  unsigned int length{0};

  Eigen::VectorXd get_state_at_step(unsigned int step) const {
    return states.get_state_at_step(step);
  }

  Eigen::VectorXd get_input_at_step(unsigned int step) const {
    return inputs.get_input_at_step(step);
  }
};

using Spline1D = Eigen::Spline<double, 1, 3>;

// polynomial class takes in coefficients in order of ascending degree!
// c0 + c1 * t + c2 * t^2 ...
class Polynomial {
 public:
  Polynomial() = default;

  Polynomial(const Eigen::VectorXd& coeffs)
      : coeffs_(coeffs), degree_(coeffs.size() - 1) {}

  ~Polynomial() = default;

  double pos(double t) const {
    Eigen::VectorXd basis = get_basis(t);
    return coeffs_.dot(basis);
  }

  double derivative(double t, unsigned int order) const {
    if (order == 0)
      return pos(t);
    if (order >= coeffs_.size())
      return 0.0;

    double result           = 0.0;
    Eigen::VectorXd t_basis = get_basis(t);

    const unsigned int max_i = coeffs_.size() - order;
    for (unsigned int i = 0; i < max_i; ++i) {
      const unsigned int k = i + order;
      result += t_basis[i] * coeffs_[k] * deriv_coeff(k, order);
    }

    return result;
  }

 private:
  Eigen::VectorXd coeffs_{Eigen::Vector3d::Zero()};
  unsigned int degree_{0};

 private:
  const Eigen::VectorXd get_basis(double t) const {

    double pow{1.};
    Eigen::VectorXd basis{Eigen::VectorXd::Zero(degree_ + 1)};

    for (unsigned int i{0}; i <= basis.size(); ++i) {
      basis[i] = pow;
      pow *= t;
    }

    return basis;
  }

  const double deriv_coeff(unsigned int ind, unsigned int order) const {
    if (order > ind)
      return 0.;

    double c = 1.;
    for (unsigned int j = 0; j < order; ++j) {
      c *= static_cast<double>(ind - j);
    }

    return c;
  }
};

class Spline {
 public:
  static constexpr unsigned int kDegree = 3;

  using Spline1D = Eigen::Spline<double, 1, kDegree>;
  using Point    = Eigen::Vector2d;

  Spline() = default;

  Spline(const Eigen::RowVectorXd& knots, const Eigen::RowVectorXd& xs)
      : knots_(knots), xs_(xs) {

    const auto fitX = interp(xs, kDegree, knots);
    spline_         = Spline1D(fitX);
  }

  ~Spline() = default;

  double operator()(double t) const { return spline_(t).coeff(0); }

  double operator()(double t, unsigned int order) const {
    return spline_.derivatives(t, order).coeff(order);
  }

  double pos(double t) const { return spline_(t).coeff(0); }

  double derivative(double t, unsigned int order) const {
    if (order == 0) {
      return pos(t);
    }

    return spline_.derivatives(t, order).coeff(order);
  }

  const Eigen::RowVectorXd& get_knots() const { return knots_; }
  const Eigen::RowVectorXd& get_ctrls() const { return xs_; }

 private:
  Spline1D interp(const Eigen::RowVectorXd& pts, Eigen::DenseIndex degree,
                  const Eigen::RowVectorXd& knot_parameters) {
    using namespace Eigen;

    typedef typename Spline1D::KnotVectorType::Scalar Scalar;
    typedef typename Spline1D::ControlPointVectorType ControlPointVectorType;

    typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

    Eigen::RowVectorXd knots;
    knots.resize(knot_parameters.size() + degree + 1);

    // not-a-knot condition setup
    knots.segment(0, degree + 1) =
        knot_parameters(0) * Eigen::RowVectorXd::Ones(degree + 1);
    knots.segment(degree + 1, knot_parameters.size() - 4) =
        knot_parameters.segment(2, knot_parameters.size() - 4);
    knots.segment(knots.size() - degree - 1, degree + 1) =
        knot_parameters(knot_parameters.size() - 1) *
        Eigen::RowVectorXd::Ones(degree + 1);

    DenseIndex n = pts.cols();
    MatrixType A = MatrixType::Zero(n, n);
    for (DenseIndex i = 1; i < n - 1; ++i) {
      const DenseIndex span = Spline1D::Span(knot_parameters[i], degree, knots);

      // The segment call should somehow be told the spline order at compile
      // time.
      A.row(i).segment(span - degree, degree + 1) =
          Spline1D::BasisFunctions(knot_parameters[i], degree, knots);
    }
    A(0, 0)         = 1.0;
    A(n - 1, n - 1) = 1.0;

    HouseholderQR<MatrixType> qr(A);

    // Here, we are creating a temporary due to an Eigen issue.
    ControlPointVectorType ctrls =
        qr.solve(MatrixType(pts.transpose())).transpose();

    return Spline1D(knots, ctrls);
  }

  Spline1D spline_;

  Eigen::RowVectorXd xs_;
  Eigen::RowVectorXd knots_;
};

class Trajectory {
 public:
  using Point = Eigen::Vector2d;
  using Row   = Eigen::RowVectorXd;

  struct View {
    Row knots;
    Row xs;
    Row ys;
    double arclen{0};
  };

  Trajectory() = default;

  Trajectory(const Row& knots, const Row& xs, const Row& ys) {
    spline_x_         = Spline(knots, xs);
    spline_y_         = Spline(knots, ys);
    true_arc_len_     = knots(Eigen::indexing::last);
    extended_arc_len_ = true_arc_len_;
  }

  Trajectory(const Spline& x, const Spline& y)
      : spline_x_(x),
        spline_y_(y),
        true_arc_len_(y.get_knots()(Eigen::indexing::last)),
        extended_arc_len_(true_arc_len_) {}

  // extension length is the new full arc length of the traj
  void extend_to_length(double extension_length) {

    if (true_arc_len_ >= extension_length)
      return;

    int step_size = spline_x_.get_knots().size();

    double epsilon = 0.1;
    double ds      = extension_length / (step_size - 1);

    double end      = true_arc_len_ - epsilon;
    Point end_point = (*this)(end);
    Point end_der   = (*this)(end, kFirstOrder);

    Row knots(step_size), xs(step_size), ys(step_size);
    for (int i = 0; i < step_size; ++i) {
      double s = ds * i;
      knots(i) = s;

      if (s < true_arc_len_) {
        xs(i) = spline_x_(s);
        ys(i) = spline_y_(s);
      } else {
        xs(i) = end_der(kX) * (s - true_arc_len_) + end_point(kX);
        ys(i) = end_der(kY) * (s - true_arc_len_) + end_point(kY);
      }
    }

    spline_x_ = Spline(knots, xs);
    spline_y_ = Spline(knots, ys);

    extended_arc_len_ = extension_length;
  }

  double get_extended_length() const { return extended_arc_len_; }

  double get_true_length() const { return true_arc_len_; }

  Point operator()(double s) const { return {spline_x_(s), spline_y_(s)}; }

  // derivative
  Point operator()(double s, unsigned int order) const {
    return {spline_x_(s, order), spline_y_(s, order)};
  }

  Trajectory get_adjusted_traj(double s, unsigned int step_count = 0) const {
    if (step_count == 0)
      step_count = spline_x_.get_knots().size();

    Row knots, xs, ys;  //, abvs, blws;
    knots.resize(step_count);
    xs.resize(step_count);
    ys.resize(step_count);

    // get true end point of trajectory
    double epsilon  = 0.1;
    Point end_point = (*this)(true_arc_len_ - epsilon);

    // capture reference at each sample
    for (int i = 0; i < step_count; ++i) {
      knots(i) = ((double)i) * extended_arc_len_ / (step_count - 1);

      // if sample domain exceeds trajectory, duplicate final point
      if (knots(i) + s <= extended_arc_len_) {
        Point p = (*this)(knots(i) + s);
        xs(i)   = p(kX);
        ys(i)   = p(kY);
      } else {
        xs(i) = end_point(kX);
        ys(i) = end_point(kY);
      }
    }

    Spline x(knots, xs);
    Spline y(knots, ys);
    return Trajectory(x, y);
  }

  double get_closest_s(const Point& state) const {
    double s        = 0;
    double min_dist = 1e6;
    Point pos{state(0), state(1)};
    for (double i = 0.0; i < extended_arc_len_; i += .01) {
      Point p  = (*this)(i);
      double d = (pos - p).squaredNorm();
      if (d < min_dist) {
        min_dist = d;
        s        = i;
      }
    }

    return s;
  }

  double distance_from(const Point& pt) const {
    double min_dist = 1e6;
    for (double i = 0.0; i < extended_arc_len_; i += .01) {
      double d = (pt - (*this)(i)).squaredNorm();
      if (d < min_dist) {
        min_dist = d;
      }
    }

    return min_dist;
  }

  View view() {
    // x and y are forced to have same knots
    return {.knots  = spline_x_.get_knots(),
            .xs     = evaluate_axis_at_knots(kX),
            .ys     = evaluate_axis_at_knots(kY),
            .arclen = extended_arc_len_};
  }

  const Row& get_ctrls_x() { return spline_x_.get_ctrls(); }
  const Row& get_ctrls_y() { return spline_y_.get_ctrls(); }

  static constexpr unsigned int kX = 0;
  static constexpr unsigned int kY = 1;

  static constexpr unsigned int kFirstOrder  = 1;
  static constexpr unsigned int kSecondOrder = 2;

 private:
  double true_arc_len_{0};
  double extended_arc_len_{0};
  Spline spline_x_;
  Spline spline_y_;

 private:
  // axis 0 == x, axis 1 == y
  // use kX and kY...
  Row evaluate_axis_at_knots(unsigned int axis) {
    // spline x and y by construction have the same knots...
    Row knots = spline_x_.get_knots();
    Spline spl;
    if (axis != kX && axis != kY) {
      throw std::runtime_error(
          "[Trajectory] Invalid axis: " + std::to_string(axis) +
          " passed to get_axis_at_knots");
    }

    Row vals(knots.size());
    for (int i = 0; i < knots.size(); ++i) {
      double s = knots(i);
      vals[i]  = axis == 0 ? spline_x_(s) : spline_y_(s);
    }

    return vals;
  }
};

}  // namespace types
}  // namespace mpcc

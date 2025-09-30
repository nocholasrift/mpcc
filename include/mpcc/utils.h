#pragma once
#include <mpcc/types.h>

#include <Eigen/Core>
#include <iostream>

namespace utils {

/**********************************************************************
 * Function: Interp
 * Description: Interpolates a 1D spline through a set of points
 * Parameters:
 * @param pts: const Eigen::RowVectorXd&
 * @param degree: Eigen::DenseIndex
 * @param knot_parameters: const Eigen::RowVectorXd&
 * Returns:
 * Spline1D
 * Notes:
 * This function is a wrapper around the Eigen Spline class, and
 * modified only slightly to allow for the not-a-knot condition
 **********************************************************************/
inline Spline1D Interp(const Eigen::RowVectorXd& pts, Eigen::DenseIndex degree,
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

/**********************************************************************
 * Function: eval_traj
 * Description: Evaluates a polynomial at a given point
 * Parameters:
 * @param coeffs: const Eigen::VectorXd&
 * @param x: double
 * Returns:
 * double
 * Notes:
 * This function evaluates a polynomial (trajectory) at a given point
 **********************************************************************/
inline double eval_traj(const Eigen::VectorXd& coeffs, double x) {
  double ret   = 0;
  double x_pow = 1;

  for (int i = 0; i < coeffs.size(); ++i) {
    ret += coeffs[i] * x_pow;
    x_pow *= x;
  }

  return ret;
}

}  // namespace utils

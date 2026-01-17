#pragma once
#include <mpcc/types.h>

#include <Eigen/Core>

namespace utils {

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vector_to_eigen(
    const std::vector<T>& vec) {

  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>,
                    Eigen::Unaligned>(vec.data(), vec.size());
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

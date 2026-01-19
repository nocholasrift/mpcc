#ifndef MPCC_ORIENTABLE_H
#define MPCC_ORIENTABLE_H

#include <math.h>
#include <algorithm>

namespace mpcc {
class Orientable {
 public:
  virtual ~Orientable() = default;

  // virtual just in case these ever need to be overwritten by some unique MPC impl
  // is aligned effectively only gets checked when user passes in threshold explicitly
  // to a reasonable value
  virtual double get_orient_control(double target_heading,
                                    double current_heading, double prop_gain,
                                    double min_actuation, double max_actuation,
                                    double threshold = -1) const {
    double e = atan2(sin(target_heading - current_heading),
                     cos(target_heading - current_heading));

    return std::max(min_actuation, std::min(max_actuation, prop_gain * e));
  }

  virtual bool is_aligned(double target_heading, double current_heading,
                          double threshold) const {

    double e = atan2(sin(target_heading - current_heading),
                     cos(target_heading - current_heading));

    if (fabs(e) > threshold) {
      return false;
    }

    return true;
  }
};

}  // namespace mpcc

#endif

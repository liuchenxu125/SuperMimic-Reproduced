#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace supermimic {
// Zero endpoint velocity/acceleration for interpolation between fixed targets.
// A moving policy endpoint still has its normal 50 Hz updates.
inline double transition_weight(double elapsed, double duration) {
  if(!std::isfinite(elapsed) || !std::isfinite(duration) || duration<0.)
    throw std::invalid_argument("Invalid transition timing");
  if(duration==0.) return 1.;
  const double t=std::clamp(elapsed/duration,0.,1.);
  return t*t*t*(10.+t*(-15.+6.*t));
}
} // namespace supermimic

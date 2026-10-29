#include "../supermimic/transitions.hpp"
#include <cassert>
#include <limits>

int main() {
  using supermimic::transition_weight;
  assert(transition_weight(-1.,3.)==0.);
  assert(transition_weight(0.,3.)==0.);
  assert(transition_weight(3.,3.)==1.);
  assert(transition_weight(9.,3.)==1.);
  assert(transition_weight(0.,0.)==1.);
  assert(transition_weight(1.5,3.)==.5);
  double previous=0.;
  for(int i=0;i<=3000;++i) {
    const double weight=transition_weight(i*.001,3.);
    assert(weight>=previous && weight>=0. && weight<=1.);
    // Entry/return fixed endpoints stay within their convex joint limits.
    const double target=(1.-weight)*-.18+weight*.36;
    assert(target>=-.18 && target<=.36);
    previous=weight;
  }
  const double h=1.e-4;
  assert(transition_weight(h,1.)/h<1.e-6);
  assert((1.-transition_weight(1.-h,1.))/h<1.e-6);
  for(double duration : {-1.,std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected=false;
    try {transition_weight(0.,duration);} catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
  }
}

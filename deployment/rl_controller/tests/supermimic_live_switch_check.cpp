#include "../supermimic/core.hpp"
#include <iostream>
#include <stdexcept>

void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main(int argc,char** argv) {
  try {
    check(argc==3,"Supply main bundle and next bundle");
    supermimic::Policy policy(argv[1]);
    auto next=policy.load_motion(argv[2]);
    supermimic::State state;state.q=policy.q0;state.position.setConstant(std::nan(""));
    policy.reset(state,0,false);
    supermimic::Output before;
    for(int i=0;i<20;++i) {state.q[0]+=.0001;before=policy.step(state);}
    policy.transition_motion(next,.8);
    const auto first=policy.step(state);
    for(int j=0;j<27;++j)
      check(std::abs(first.obs[600+9*27+j]-before.action[j])<1e-6,"Live switch cleared previous action");
    for(int j=0;j<9*27;++j)
      check(first.obs[j]==before.obs[j+27],"Live switch reset measured joint history");
    auto out=first;
    for(int i=0;i<40;++i) {
      if(i) out=policy.step(state);
      check(out.frame==0,"Destination clip clock advanced during bridge");
      check(policy.transitioning()==(i<39),"Bridge duration is not 40 policy updates");
      for(int t=0;t<10;++t) {
        check(std::isfinite(out.goal[t*64+63]),"Nonfinite bridge feature");
        const Eigen::Vector3f g=Eigen::Map<const Eigen::Vector3f>(out.goal.data()+t*64+54);
        check(std::abs(g.norm()-1.)<1e-5,"Interpolated gravity is not a unit vector");
      }
      if(i==0) {
        // Compare against continuation at the existing reference velocity.
        // SDK stepping can legitimately move >0.1 rad in 20 ms (knee ~5 rad/s);
        // that motion must not be mistaken for a switch discontinuity.
        const supermimic::Joints continued=before.reference.q+policy.dt*before.reference.dq;
        check((out.reference.q-continued).cwiseAbs().maxCoeff()<.01,"Bridge begins with a discontinuity beyond source velocity");
        bool rejected=false;
        try {policy.transition_motion(next,.8);} catch(const std::exception&) {rejected=true;}
        check(rejected,"Nested transition was accepted");
      }
    }
    for(int j=0;j<64;++j) check(out.goal[j]==next->frames.front().features[j],"Bridge endpoint differs from next frame 0");
    const auto after=policy.step(state);
    check(after.frame==1,"Playback did not continue at frame 1");
    for(int t=0;t<10;++t) for(int j=0;j<64;++j)
      check(after.goal[t*64+j]==next->frames[t+1].features[j],"Next clip encoding changed");
    for(int j=0;j<27;++j)
      check(std::abs(after.obs[600+9*27+j]-out.action[j])<1e-6,"Bridge end reset previous action");
    check(policy.step(state).frame==2,"Next clip clock did not advance");
    std::cout<<"PASS: 40-step live bridge preserves history/action; normalized gravity; exact frame-0 endpoint; frame-1 resume; no encoding changes.\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

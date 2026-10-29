#include "../supermimic/core.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool ok, const char* text) {if(!ok) throw std::runtime_error(text);}
int main(int argc,char** argv) {
  try {
    check(argc==2,"Supply hardware bundle path");
    supermimic::Policy policy(argv[1]);
    supermimic::State state;
    state.q=policy.q0;
    if(!policy.requires_root_pose()) state.position.setConstant(std::nan(""));
    policy.reset(state,0,false);
    supermimic::Joints previous=supermimic::Joints::Zero();
    for(int step=0;step<50;++step) {
      state.q[0]=policy.q0[0]+step*.0001;
      const auto out=policy.step(state,false);
      check(out.frame==0,"Frozen calls advanced the reference");
      for(int k=0;k<10;++k) {
        for(int j=0;j<64;++j) check(out.goal[k*64+j]==out.goal[j],"Future window is not stationary");
        for(int j=27;j<54;++j) check(out.goal[k*64+j]==0.,"Held reference has joint velocity");
        for(int j=57;j<63;++j) check(out.goal[k*64+j]==0.,"Held reference has base velocity");
      }
      if(step) {
        check(std::abs(out.obs[9*27]-(state.q[0]-policy.q0[0]))<1.e-7,"Measured history stopped updating");
        for(int j=0;j<27;++j)
          check(std::abs(out.obs[600+9*27+j]-previous[j])<1.e-5,"Previous action history is wrong");
      }
      previous=out.action;
    }
    const auto first=policy.step(state);
    check(first.frame==1,"Playback did not start at f+1");
    for(int j=0;j<27;++j)
      check(std::abs(first.obs[600+9*27+j]-previous[j])<1.e-5,"Playback reset handoff history");
    check(policy.step(state).frame==2,"Playback cursor did not resume advancing");
    policy.reset(state,policy.length()-2,false);
    const auto final=policy.step(state);
    check(final.frame==policy.length()-1 && policy.finished(),"Final playback frame did not finish");
    previous=final.action;
    const auto end_reference=policy.reference(policy.length()-1);
    for(int step=0;step<50;++step) {
      state.q[0]=policy.q0[0]+step*.0002;
      const auto out=policy.step_hold(state,policy.length()-1);
      check(out.frame==policy.length()-1 && policy.finished(),"Final hold changed motion cursor");
      check((out.reference.q-end_reference.q).norm()==0.,"Final hold changed reference pose");
      check(out.reference.dq.isZero(),"Final hold retained reference joint velocity");
      for(int k=0;k<10;++k) {
        for(int j=0;j<64;++j) check(out.goal[k*64+j]==out.goal[j],"Final hold future window is not stationary");
        for(int j=27;j<54;++j) check(out.goal[k*64+j]==0.,"Final hold goal has joint velocity");
        for(int j=57;j<63;++j) check(out.goal[k*64+j]==0.,"Final hold goal has base velocity");
      }
      check(std::abs(out.obs[9*27]-(state.q[0]-policy.q0[0]))<1.e-7,"Final hold stopped measured history");
      for(int j=0;j<27;++j)
        check(std::abs(out.obs[600+9*27+j]-previous[j])<1.e-5,"Final hold reset previous action");
      previous=out.action;
    }
    bool rejected=false;
    try {policy.step(state);} catch(const std::exception&) {rejected=true;}
    check(rejected,"Finished playback incorrectly resumed advancing");
    policy.transition_motion(policy.load_motion(argv[1]),.8);
    const auto bridge=policy.step(state);
    check(bridge.frame==0 && policy.transitioning(),"Cannot live-switch out of final hold");
    for(int j=0;j<27;++j)
      check(std::abs(bridge.obs[600+9*27+j]-previous[j])<1.e-5,"Switch out of final hold reset action history");
    std::cout<<"PASS: start and final reference holds keep stationary goals and live measured/action history; playback and live switching preserve history.\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

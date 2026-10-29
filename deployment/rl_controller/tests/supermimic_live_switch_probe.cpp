// Offline streaming probe for native MuJoCo. No ROS or hardware publications.
// Same 64-float64 input as probe.cpp; output obs,goal,action,target then
// effective reference position3/quaternion4/joints27 and frame/bridge flag.
#include "../supermimic/core.hpp"
#include <iostream>
int main(int argc,char** argv) {
  try {
    if(argc!=6) throw std::runtime_error("Usage: switch_probe BUNDLE NEXT_BUNDLE SWITCH_TICK SECONDS START_FRAME");
    supermimic::Policy policy(argv[1]);const auto next=policy.load_motion(argv[2]);
    const int trigger=std::stoi(argv[3]),start=std::stoi(argv[5]);
    const double seconds=std::stod(argv[4]);int tick=0;
    std::array<double,64> row;
    while(std::cin.read(reinterpret_cast<char*>(row.data()),sizeof(row))) {
      supermimic::State state;
      state.q=Eigen::Map<supermimic::Joints>(row.data());state.dq=Eigen::Map<supermimic::Joints>(row.data()+27);
      state.position=Eigen::Map<Eigen::Vector3d>(row.data()+54);
      state.orientation=Eigen::Quaterniond(row[57],row[58],row[59],row[60]);state.omega=Eigen::Map<Eigen::Vector3d>(row.data()+61);
      if(!tick) policy.reset(state,start,false);
      if(tick==trigger) policy.transition_motion(next,seconds);
      const bool bridge=policy.transitioning();
      const auto out=policy.step(state);++tick;
      std::cout.write(reinterpret_cast<const char*>(out.obs.data()),sizeof(out.obs));
      std::cout.write(reinterpret_cast<const char*>(out.goal.data()),out.goal.size()*sizeof(float));
      const Eigen::Matrix<float,27,1> action=out.action.cast<float>(),target=out.target.cast<float>();
      std::cout.write(reinterpret_cast<const char*>(action.data()),27*sizeof(float));
      std::cout.write(reinterpret_cast<const char*>(target.data()),27*sizeof(float));
      std::array<float,36> extra{};
      for(int j=0;j<3;++j) extra[j]=out.reference.position[j];
      extra[3]=out.reference.orientation.w();extra[4]=out.reference.orientation.x();
      extra[5]=out.reference.orientation.y();extra[6]=out.reference.orientation.z();
      for(int j=0;j<27;++j) extra[7+j]=out.reference.q[j];
      extra[34]=out.frame;extra[35]=bridge?1.f:0.f;
      std::cout.write(reinterpret_cast<const char*>(extra.data()),extra.size()*sizeof(float));std::cout.flush();
    }
    if(std::cin.gcount()) throw std::runtime_error("Truncated input");
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

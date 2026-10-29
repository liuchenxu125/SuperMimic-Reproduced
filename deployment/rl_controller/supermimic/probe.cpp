// Offline binary probe. No ROS initialization and no command publication.
// stdin: repeated 64 little-endian float64: q27,dq27,p3,quat_wxyz4,omega3.
// stdout: obs870,goal[manifest goal_dim],action27,target27 (float32).
#include "core.hpp"
#include <iostream>

int main(int argc,char** argv) {
  try {
    if(argc<2 || argc>4) throw std::runtime_error("Usage: supermimic_probe BUNDLE [--no-align] [START_FRAME]");
    bool align=true; int start=0;
    for(int i=2;i<argc;++i) {
      if(std::string(argv[i])=="--no-align") align=false;
      else start=std::stoi(argv[i]);
    }
    supermimic::Policy policy(argv[1]);
    bool first=true;
    std::array<double,64> row;
    while(std::cin.read(reinterpret_cast<char*>(row.data()),sizeof(row))) {
      supermimic::State state;
      state.q=Eigen::Map<supermimic::Joints>(row.data());
      state.dq=Eigen::Map<supermimic::Joints>(row.data()+27);
      state.position=Eigen::Map<Eigen::Vector3d>(row.data()+54);
      state.orientation=Eigen::Quaterniond(row[57],row[58],row[59],row[60]);
      state.omega=Eigen::Map<Eigen::Vector3d>(row.data()+61);
      if(first) { policy.reset(state,start,align); first=false; }
      const auto out=policy.step(state);
      std::cout.write(reinterpret_cast<const char*>(out.obs.data()),sizeof(out.obs));
      std::cout.write(reinterpret_cast<const char*>(out.goal.data()),out.goal.size()*sizeof(float));
      const Eigen::Matrix<float,27,1> action=out.action.cast<float>(), target=out.target.cast<float>();
      std::cout.write(reinterpret_cast<const char*>(action.data()),sizeof(float)*27);
      std::cout.write(reinterpret_cast<const char*>(target.data()),sizeof(float)*27);
      std::cout.flush();
    }
    if(std::cin.gcount()!=0) throw std::runtime_error("Truncated sensor record");
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

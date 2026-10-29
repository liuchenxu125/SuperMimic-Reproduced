#include "../supermimic/core.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main(int argc,char** argv) {
  try {
    check(argc==3,"Supply baseline bundle and motion bank directory");
    supermimic::Policy policy(argv[1]);
    int clips=0;
    for(const auto& entry:std::filesystem::directory_iterator(argv[2])) {
      if(!std::filesystem::is_regular_file(entry.path()/"manifest.json")) continue;
      const auto clip=policy.load_motion(entry.path());
      policy.select_motion(clip);
      supermimic::Policy fresh(entry.path());
      supermimic::State state;state.q=policy.q0;
      policy.reset(state,0,false);fresh.reset(state,0,false);
      for(int i=0;i<10;++i) {
        const auto a=policy.step(state),b=fresh.step(state);
        check(a.obs==b.obs && a.goal==b.goal,"Switched history/reference differs from a fresh policy");
        check((a.action-b.action).norm()<1e-5,"Switched policy action differs from a fresh policy");
        state.q[0]+=.0001;
      }
      // End at the last frame, without an extra duplicated terminal policy step.
      policy.reset(state,policy.length()-2,false);
      check(!policy.finished(),"Motion ended before its last reference frame");
      check(policy.step(state).frame==policy.length()-1,"Last reference frame skipped");
      check(policy.finished(),"Motion did not end at its final reference frame");
      std::cout<<"PASS "<<policy.motion_name()<<" frames="<<policy.length()<<'\n';
      ++clips;
    }
    const auto catalog=YAML::LoadFile((std::filesystem::path(argv[2])/"catalog.json").string());
    check(clips==static_cast<int>(catalog["clips"].size()) && clips>0,"Deployment clip count differs from catalog");
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

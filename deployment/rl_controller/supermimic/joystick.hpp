#pragma once
#include <set>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace supermimic {
enum class JoyAction {None,Advance,Finish};
// The terminal may expose more motions than the joystick's chosen cycle.
inline void validate_motion_order(const std::vector<std::string>& order,
    const std::vector<std::string>& available) {
  if(order.empty()) throw std::invalid_argument("Joystick motion order is empty");
  const std::set<std::string> known(available.begin(),available.end());
  std::set<std::string> seen;
  for(const auto& name:order) {
    if(!seen.insert(name).second) throw std::invalid_argument("Duplicate joystick motion: "+name);
    if(!known.count(name)) throw std::invalid_argument("Unknown joystick motion: "+name);
  }
}
// Observe the SDK's debug-entry chord as intent only. The SDK still changes
// mode; the runner must separately confirm a fresh raw mode 7 afterwards.
inline bool debug_start_chord(const std::vector<uint32_t>& pressed,
    const std::vector<uint32_t>& held,const std::vector<uint32_t>& clicked,
    const std::vector<uint32_t>& doubled,int x,int y) {
  if(x!=0 || y!=0 || !doubled.empty()) return false;
  std::set<uint32_t> keys(pressed.begin(),pressed.end());
  keys.insert(held.begin(),held.end());keys.insert(clicked.begin(),clicked.end());
  return keys==std::set<uint32_t>{0,4}; // SDK 2.2.37: A=0, LB=4.
}
inline std::string next_motion(const std::vector<std::string>& order,const std::string& current) {
  if(order.empty()) throw std::invalid_argument("Joystick motion order is empty");
  const auto it=std::find(order.begin(),order.end(),current);
  // A terminal-selected motion outside the cycle rejoins it at its first item.
  if(it==order.end()) return order.front();
  return order[(static_cast<size_t>(it-order.begin())+1)%order.size()];
}
// SDK 2.2.37: X=2 advances and plays, B=1 returns. Emit only on release.
// Any chord (especially LB/RB/BACK), initial held input, or receipt gap must
// return to neutral before another action can be emitted.
class JoystickEdges {
 public:
  JoyAction update(const std::vector<uint32_t>& pressed,const std::vector<uint32_t>& held,
                   const std::vector<uint32_t>& doubled,int x,int y,double now) {
    if(!received_ || now-previous_>.5) {armed_=false;active_=JoyAction::None;}
    received_=true;previous_=now;
    std::set<uint32_t> keys(pressed.begin(),pressed.end());
    keys.insert(held.begin(),held.end());keys.insert(doubled.begin(),doubled.end());
    if(keys.empty() && x==0 && y==0) {
      const auto event=armed_?active_:JoyAction::None;
      armed_=true;active_=JoyAction::None;return event;
    }
    JoyAction candidate=JoyAction::None;
    if(x==0 && y==0 && keys.size()==1) {
      if(*keys.begin()==2) candidate=JoyAction::Advance;
      if(*keys.begin()==1) candidate=JoyAction::Finish;
    }
    // Double-click reports never start a second action or consume SDK e-stop.
    if(!doubled.empty() || candidate==JoyAction::None ||
       (active_!=JoyAction::None && candidate!=active_)) {
      armed_=false;active_=JoyAction::None;
    } else if(armed_) active_=candidate;
    return JoyAction::None;
  }
 private:
  bool received_=false,armed_=false;
  double previous_=0;
  JoyAction active_=JoyAction::None;
};
} // namespace supermimic

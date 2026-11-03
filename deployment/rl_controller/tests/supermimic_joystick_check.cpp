#include "../supermimic/joystick.hpp"
#include <iostream>
#include <stdexcept>
using supermimic::JoyAction;
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main() {
  try {
    using supermimic::debug_start_chord;
    check(debug_start_chord({0,4},{},{},{},0,0),"LB+A intent not recognized");
    check(debug_start_chord({0},{4},{},{},0,0),"Held LB+A intent not recognized");
    check(debug_start_chord({4},{4},{0},{},0,0),"SDK A click with held LB not recognized");
    for(const auto& chord:std::vector<std::vector<uint32_t>>{{0},{4},{0,5},{2,4},{0,4,6},{0,1,4}})
      check(!debug_start_chord(chord,{},{},{},0,0),"Unrelated SDK chord started policy");
    check(!debug_start_chord({0,4},{},{},{0},0,0),"Double-click chord started policy");
    check(!debug_start_chord({0,4},{},{},{},1,0),"Ambiguous D-pad chord started policy");
    supermimic::JoystickEdges edges;double now=0;
    const auto send=[&](std::vector<uint32_t> p={},int x=0,std::vector<uint32_t> held={},std::vector<uint32_t> twice={}){
      now+=.05;return edges.update(p,held,twice,x,0,now);
    };
    check(send({2})==JoyAction::None,"Initial held X fired");
    check(send()==JoyAction::None,"Initial X release fired");
    check(send({2})==JoyAction::None,"X fired before release");
    for(int i=0;i<30;++i) check(send({2},0,{2})==JoyAction::None,"Held X repeated");
    check(send()==JoyAction::Advance,"X release not recognized");
    check(send()==JoyAction::None,"X repeated after release");
    check(send({},1)==JoyAction::None && send({},1)==JoyAction::None && send()==JoyAction::None,"D-pad must not select");
    check(send({},-1)==JoyAction::None && send()==JoyAction::None,"D-pad must not select");
    send({1});check(send()==JoyAction::Finish,"B release failed");
    for(uint32_t modifier:{4u,5u,6u,7u,11u,12u}) {
      send({modifier,2},0,{modifier});send({2});
      check(send()==JoyAction::None,"SDK modifier chord leaked into X playback");
      send({modifier,1});send({1});check(send()==JoyAction::None,"SDK modifier chord leaked into B finish");
    }
    send({1,2});check(send()==JoyAction::None,"Ambiguous chord fired");
    send({2},1);check(send()==JoyAction::None,"D-pad/X chord fired");
    send({2});send({},0,{}, {2});check(send()==JoyAction::None,"Double click fired");
    send({2});now+=1;check(send()==JoyAction::None,"Receipt gap caused late play");
    send({2});check(send()==JoyAction::Advance,"Neutral did not rearm");
    const std::vector<std::string> order={"walk1_subject2","walk1_subject1","walk2_subject4",
      "walk3_subject1","walk4_subject1","dance1_subject3","dance1_subject2"};
    supermimic::validate_motion_order(order,order);
    const std::vector<std::string> subset={order[2],order[0]};
    supermimic::validate_motion_order(subset,order);
    check(supermimic::next_motion(subset,order[2])==order[0],"Subset next motion is wrong");
    check(supermimic::next_motion(subset,order[0])==order[2],"Subset cycle does not wrap");
    check(supermimic::next_motion(subset,order[1])==subset.front(),"Terminal-selected motion cannot rejoin cycle");
    supermimic::validate_motion_order({order[0]},order);
    check(supermimic::next_motion({order[0]},order[0])==order[0],"Single-motion cycle failed");
    for(const auto& invalid:std::vector<std::vector<std::string>>{{},{order[0],order[0]},{"missing"}}) {
      bool rejected=false;
      try {supermimic::validate_motion_order(invalid,order);} catch(const std::invalid_argument&) {rejected=true;}
      check(rejected,"Invalid joystick cycle was accepted");
    }
    auto current=order.front();
    for(size_t i=1;i<=order.size();++i) {
      current=supermimic::next_motion(order,current);
      check(current==order[i%order.size()],"X next-motion order or wrap is wrong");
    }
    std::cout<<"PASS: subset/full joystick cycles, invalid names/duplicates/empty rejection, release edges, SDK chords, gaps and rearming\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

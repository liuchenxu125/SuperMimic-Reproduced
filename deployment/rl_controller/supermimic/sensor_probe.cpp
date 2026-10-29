// Passive comparison with the AMP sensor path: no actor, PD loop or motor publishers.
#include <rclcpp/rclcpp.hpp>
#include <rmw/rmw.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <crb_ros_msg/msg/joint_state_data.hpp>
#include <std_msgs/msg/int32.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

class SensorProbe : public rclcpp::Node {
 public:
  SensorProbe() : Node("supermimic_sensor_probe"), began_(Clock::now()) {
    duration_=declare_parameter<double>("seconds",12.);
    if(!std::isfinite(duration_) || duration_<=0. || duration_>300.)
      throw std::runtime_error("seconds must be within (0, 300]");
    // Same state QoS as AMP. No frame, quaternion, timestamp or joint filtering:
    // count every callback before interpreting payloads.
    auto qos=rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    standard_=create_subscription<sensor_msgs::msg::JointState>(
        "/motion/joint_state",qos,[this](const sensor_msgs::msg::JointState::SharedPtr m){
          note_joints(standard_stats_,*m);
        });
    debug_=create_subscription<crb_ros_msg::msg::JointStateData>(
        "/motion/debug/joint_state",qos,[this](const crb_ros_msg::msg::JointStateData::SharedPtr m){
          note_joints(debug_stats_,*m);
        });
    imu_=create_subscription<sensor_msgs::msg::Imu>(
        "/motion/imu",qos,[this](const sensor_msgs::msg::Imu::SharedPtr m){
          ++imu_stats_.rx;
          imu_stats_.stamp=rclcpp::Time(m->header.stamp).seconds();
          const auto& q=m->orientation;
          imu_stats_.detail="frame='"+m->header.frame_id+"' quat_norm="+
              std::to_string(std::sqrt(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z));
        });
    mode_=create_subscription<std_msgs::msg::Int32>(
        "/motion/current_mode",rclcpp::QoS(10).reliable(),
        [this](const std_msgs::msg::Int32::SharedPtr m){++mode_stats_.rx;mode_value_=m->data;});
    RCLCPP_INFO(get_logger(),"PASSIVE SENSOR PROBE: uid=%u rmw=%s domain_env=%s localhost_env=%s",
        static_cast<unsigned>(getuid()),rmw_get_implementation_identifier(),
        env("ROS_DOMAIN_ID"),env("ROS_LOCALHOST_ONLY"));
    RCLCPP_INFO(get_logger(),"No actor or motor command publishers. Exit after %.1f seconds.",duration_);
    timer_=create_wall_timer(std::chrono::seconds(1),[this]{
      report("standard_joints",standard_->get_publisher_count(),standard_stats_);
      report("debug_joints",debug_->get_publisher_count(),debug_stats_);
      report("imu",imu_->get_publisher_count(),imu_stats_);
      RCLCPP_INFO(get_logger(),"mode: pubs=%zu rx=%llu value=%d",
          mode_->get_publisher_count(),static_cast<unsigned long long>(mode_stats_.rx),mode_value_);
      if(std::chrono::duration<double>(Clock::now()-began_).count()>=duration_)
        rclcpp::shutdown();
    });
  }
 private:
  using Clock=std::chrono::steady_clock;
  struct Stats {uint64_t rx=0;double stamp=0.;std::string detail="none";};
  static const char* env(const char* name) {const auto* value=std::getenv(name);return value?value:"<unset>";}
  template<class Message> static void note_joints(Stats& stats,const Message& m) {
    ++stats.rx;
    stats.stamp=rclcpp::Time(m.header.stamp).seconds();
    stats.detail="names="+std::to_string(m.name.size())+" positions="+
        std::to_string(m.position.size())+" velocities="+std::to_string(m.velocity.size());
    if(stats.rx==1) {
      std::string names;
      for(const auto& name:m.name) {if(!names.empty()) names+=",";names+=name;}
      std::cout<<"[FIRST JOINT MESSAGE] "<<names<<std::endl;
    }
  }
  void report(const char* name,size_t publishers,const Stats& s) {
    RCLCPP_INFO(get_logger(),"%s: pubs=%zu rx=%llu stamp_age_ms=%.1f last={%s}",
        name,publishers,static_cast<unsigned long long>(s.rx),
        s.rx?(now().seconds()-s.stamp)*1000.:-1.,s.detail.c_str());
  }
  Clock::time_point began_;
  double duration_=12.;
  int mode_value_=0;
  Stats standard_stats_,debug_stats_,imu_stats_,mode_stats_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr standard_;
  rclcpp::Subscription<crb_ros_msg::msg::JointStateData>::SharedPtr debug_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mode_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  try {
    auto node=std::make_shared<SensorProbe>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),4);
    executor.add_node(node);
    executor.spin();
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<std::endl;
    if(rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
  if(rclcpp::ok()) rclcpp::shutdown();
}

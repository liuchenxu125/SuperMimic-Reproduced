// hl_motion 2.2.37 adapter: 50 Hz policy, SDK position/PD targets at 1 kHz.
// SDK owns PD/actuation. effort is deliberately zero; it is NOT torque passthrough.
// Default dry_run creates NO motor command publisher. No AMP inference thread.
#include "core.hpp"
#include "transitions.hpp"
#include "joystick.hpp"
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <crb_ros_msg/msg/joint_state_data.hpp>
#include <crb_ros_msg/srv/get_robot_mode.hpp>
#include <crb_ros_msg/msg/joystick_cmd_report.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <iostream>
#include <limits>
#include <csignal>
#include <map>

using Clock=std::chrono::steady_clock;
using Seconds=std::chrono::duration<double>;
using Trigger=std_srvs::srv::Trigger;
using GetMode=crb_ros_msg::srv::GetRobotMode;
namespace sm=supermimic;
volatile std::sig_atomic_t shutdown_requested=0;
void request_shutdown(int) {shutdown_requested=1;}

// Lifetime maxima, observable without taking the control/publish lock.
struct MaxMs {
  std::atomic<double> value{0.};
  void note(double ms) {
    double old=value.load(std::memory_order_relaxed);
    while(ms>old && !value.compare_exchange_weak(old,ms,std::memory_order_relaxed)) {}
  }
  void since(Clock::time_point t) {note(Seconds(Clock::now()-t).count()*1000.);}
};
struct ReceiveTiming {
  Clock::time_point previous{}; // sensor callback group is mutually exclusive
  MaxMs gap,lock_wait;
  std::atomic<uint64_t> over_100ms{0};
  Clock::time_point enter() {
    const auto t=Clock::now();
    if(previous!=Clock::time_point{}) {
      const double ms=Seconds(t-previous).count()*1000.;
      gap.note(ms);
      if(ms>100.) ++over_100ms;
    }
    previous=t;return t;
  }
};

class Runner : public rclcpp::Node {
  struct JointInput {
    sm::Joints q=sm::Joints::Zero(),dq=sm::Joints::Zero();
    uint64_t messages=0;
    bool valid=false;
    Eigen::Vector2d head=Eigen::Vector2d::Zero();
    double stamp=0.;
    Clock::time_point received{};
    std::string reason="no messages received";
  };
 public:
  explicit Runner(const std::filesystem::path& path) : Node("supermimic_controller"),
      cfg_(YAML::LoadFile(path.string())),
      policy_(path.parent_path()/cfg_["bundle"].as<std::string>()) {
    dry_=declare_parameter<bool>("dry_run",true);
    console_full_status_=declare_parameter<bool>("console_full_status",cfg_["console_full_status"].as<bool>(false));
    max_compute_seconds_=cfg_["max_inference_seconds"].as<double>(0.04);
    late_compute_seconds_=cfg_["late_inference_seconds"].as<double>(0.025);
    max_consecutive_late_=cfg_["max_consecutive_late_updates"].as<int>(20);
    if(!std::isfinite(max_compute_seconds_) || max_compute_seconds_<policy_.dt ||
       max_compute_seconds_>value("max_target_age") ||
       !std::isfinite(late_compute_seconds_) || late_compute_seconds_<policy_.dt ||
       late_compute_seconds_>max_compute_seconds_ || max_consecutive_late_<1)
      throw std::runtime_error("Inference thresholds must satisfy policy dt <= late <= hard <= max_target_age; late count must be positive");
    // Reading the installed configuration prevents sending zero-gain damping
    // into an SDK configured to substitute unrelated default gains. Feedback
    // below also checks the running SDK, which may not have reloaded this file.
    if(!dry_) {
      const auto sdk_path=cfg_["sdk_config_file"].as<std::string>("/workspace/hl_motion/hl_config/motion_params.yaml");
      const auto sdk=YAML::LoadFile(sdk_path);
      if(sdk["debug_mode_params"]["use_default_kp_kd"].as<bool>(true))
        throw std::runtime_error("SDK 2.2.37 requires debug_mode_params.use_default_kp_kd=false in "+sdk_path+
                                 "; reload the SDK before starting this controller");
    }
    // Match SDK's integer gain conversion explicitly; retain model metadata.
    sdk_kp_=policy_.kp.unaryExpr([](double x){return std::trunc(x);});
    sdk_kd_=policy_.kd.unaryExpr([](double x){return std::trunc(x);});
    if((sdk_kp_.array()<1).any() || (sdk_kd_.array()<1).any())
      throw std::runtime_error("SDK integer gains must remain positive");
    auto_enabled_=declare_parameter<bool>("auto_start",cfg_["auto_start_on_mode7"].as<bool>(false));
    auto_stable_seconds_=cfg_["auto_sensor_ready_seconds"].as<double>(0.5);
    auto_timeout_seconds_=cfg_["auto_start_timeout_seconds"].as<double>(5.0);
    if(!std::isfinite(auto_stable_seconds_) || auto_stable_seconds_<0.1 ||
       !std::isfinite(auto_timeout_seconds_) || auto_timeout_seconds_<auto_stable_seconds_ ||
       auto_timeout_seconds_>30.) throw std::runtime_error("Invalid automatic startup timing");
    if(cfg_["require_anchor_obs"] && cfg_["require_anchor_obs"].as<bool>()!=policy_.requires_root_pose())
      throw std::runtime_error("Configured sensor interface does not match model anchor_obs; export matching weights");
    if(!policy_.requires_root_pose())
      state_.position.setConstant(std::numeric_limits<double>::quiet_NaN()); // unavailable, never fabricated
    start_frame_=cfg_["start_frame"].as<int>();
    if(start_frame_<0 || start_frame_>=policy_.length()-1) throw std::runtime_error("Bad start_frame");
    prepare_seconds_=value("prepare_seconds"); preview_seconds_=cfg_["preview_seconds"].as<double>(0.);
    hold_last_frame_=cfg_["hold_last_frame_on_finish"].as<bool>(false);
    entry_seconds_=cfg_["entry_blend_seconds"].as<double>(0.);
    return_seconds_=cfg_["return_seconds"].as<double>(0.);
    switch_seconds_=cfg_["switch_blend_seconds"].as<double>(.8);
    if(!std::isfinite(switch_seconds_) || switch_seconds_<.2 || switch_seconds_>5.)
      throw std::runtime_error("switch_blend_seconds must be within 0.2..5s");
    const auto prepare_pose=cfg_["prepare_pose"].as<std::string>("motion_start");
    if(prepare_pose!="default" && prepare_pose!="motion_start")
      throw std::runtime_error("prepare_pose must be default or motion_start");
    prepare_default_=prepare_pose=="default";
    default_prepare_=policy_.q0;
    const auto explicit_pose=cfg_["prepare_joint_pos"];
    if(explicit_pose) {
      if(!prepare_default_ || !explicit_pose.IsMap() || explicit_pose.size()!=sm::kJoints)
        throw std::runtime_error("prepare_joint_pos requires prepare_pose=default and all 27 model joints");
      for(int i=0;i<sm::kJoints;++i) {
        const auto name=policy_.joint_names()[i];
        if(!explicit_pose[name]) throw std::runtime_error("Missing preparation joint: "+name);
        default_prepare_[i]=explicit_pose[name].as<double>();
      }
    }
    if(!default_prepare_.allFinite() || (default_prepare_.array()<policy_.lower.array()).any() ||
       (default_prepare_.array()>policy_.upper.array()).any())
      throw std::runtime_error("Default preparation pose exceeds hard joint limits");
    if(!std::isfinite(prepare_seconds_) || !std::isfinite(preview_seconds_) ||
       prepare_seconds_<0 || (prepare_seconds_>0 && prepare_seconds_<2) || preview_seconds_<0)
      throw std::runtime_error("Require prepare=0 (skip) or >=2s, and preview>=0s (0 plays the full clip)");
    if(!std::isfinite(entry_seconds_) || !std::isfinite(return_seconds_) ||
       entry_seconds_<0 || entry_seconds_>30 || return_seconds_<0 || return_seconds_>30)
      throw std::runtime_error("Transition durations must be finite and within 0..30s");
    // Preload references and check the actual actor bytes before publishers exist.
    // Every clip uses this single ONNX session and the same robot/control contract.
    motions_.emplace(policy_.motion_name(),policy_.load_motion(path.parent_path()/text("bundle")));
    if(const auto bank=cfg_["motions"]) {
      if(!bank.IsMap()) throw std::runtime_error("motions must map clip names to bundle paths");
      for(const auto& item:bank) {
        const auto name=item.first.as<std::string>();
        const auto clip=policy_.load_motion(path.parent_path()/item.second.as<std::string>());
        if(name!=clip->name) throw std::runtime_error("Motion name differs from bundle clip: "+name);
        motions_[name]=clip;
      }
    }
    std::vector<std::string> motion_names;
    for(const auto& [name,clip]:motions_) motion_names.push_back(name);
    rcl_interfaces::msg::ParameterDescriptor read_only;
    read_only.read_only=true;
    declare_parameter<std::vector<std::string>>("available_motions",motion_names,read_only);
    const auto selected=declare_parameter<std::string>("motion",policy_.motion_name());
    if(!motions_.count(selected)) throw std::runtime_error("Unknown initial motion: "+selected);
    policy_.select_motion(motions_.at(selected));
    joystick_choice_=selected;
    if(start_frame_>=policy_.length()-1) throw std::runtime_error("Bad start_frame for selected motion");
    motion_callback_=add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& parameters){
      rcl_interfaces::msg::SetParametersResult result; result.successful=true;
      for(const auto& parameter:parameters) {
        if(parameter.get_name()!="motion") continue;
        if(parameters.size()!=1) {
          result.successful=false;result.reason="Set motion as a single parameter operation";return result;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if(parameter.get_type()!=rclcpp::ParameterType::PARAMETER_STRING || !motions_.count(parameter.as_string()))
          {result.successful=false;result.reason="Unknown motion; read available_motions";return result;}
        // Only the X handler may request a live switch. External parameter
        // callers retain the explicit finish/HOLD/select/play contract.
        if(joystick_live_parameter_ && (phase_==Phase::Run || phase_==Phase::EndHold)) {
          std::string reason;
          if(policy_.requires_root_pose() || !valid(reason) || !pd_ready(reason))
            {result.successful=false;result.reason="Live switch unavailable: "+reason;return result;}
          cancel_auto();switch_requested_=parameter.as_string();joystick_choice_=switch_requested_;
          return result;
        }
        if(phase_!=Phase::Idle && phase_!=Phase::Hold && phase_!=Phase::Done)
          {result.successful=false;result.reason="Finish and wait for HOLD, or stop to IDLE before selecting; FAULT requires recovery";return result;}
        std::lock_guard<std::mutex> policy_lock(policy_mutex_);
        cancel_auto(); ++generation_;
        policy_.select_motion(motions_.at(parameter.as_string()));
        joystick_choice_=parameter.as_string();
        start_frame_=frame_=0;
        // Selection alone preserves the current HOLD/damping command. /play is explicit.
      }
      return result;
    });
    joystick_enabled_=cfg_["joystick_enabled"].as<bool>(false);
    joystick_order_=cfg_["joystick_motion_order"].as<std::vector<std::string>>(motion_names);
    if(joystick_enabled_) {
      std::set<std::string> order(joystick_order_.begin(),joystick_order_.end());
      if(order.size()!=motions_.size() || order.size()!=joystick_order_.size())
        throw std::runtime_error("Invalid joystick motion order");
      for(const auto& name:order) if(!motions_.count(name)) throw std::runtime_error("Unknown joystick motion: "+name);
      joystick_=create_subscription<crb_ros_msg::msg::JoystickCmdReport>(
          cfg_["joystick_topic"].as<std::string>("/joystick_events"),
          rclcpp::QoS(1).best_effort(),[this](const crb_ros_msg::msg::JoystickCmdReport::SharedPtr msg){
            joystick_message(*msg);
          });
      joystick_timer_=create_wall_timer(std::chrono::milliseconds(50),[this]{joystick_play();});
    }
    imu_to_base_=Eigen::Quaterniond(cfg_["imu_to_base_quat_wxyz"][0].as<double>(),
        cfg_["imu_to_base_quat_wxyz"][1].as<double>(),cfg_["imu_to_base_quat_wxyz"][2].as<double>(),
        cfg_["imu_to_base_quat_wxyz"][3].as<double>());
    if(!imu_to_base_.coeffs().allFinite() || std::abs(imu_to_base_.norm()-1)>1e-5)
      throw std::runtime_error("Invalid IMU extrinsics");
    // 2.2.37 wire names only. Do not reuse another SDK's numeric joint IDs.
    for(int i=0;i<27;++i) ids_[policy_.names()[i]]=i;
    ids_["head_pitch_joint"]=27;ids_["head_yaw_joint"]=28;
    if(ids_.size()!=29) throw std::runtime_error("Invalid SDK 2.2.37 joint mapping");
    // AMP uses two joint sources and a separate executor for sensor reception.
    // Keep sensor callbacks independent of the 1 kHz controller callback group.
    sensor_group_=create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    control_group_=create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    mode_group_=create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sensor_options;
    sensor_options.callback_group=sensor_group_;
    const auto sensor_qos=rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    const auto command_qos=rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    standard_joint_topic_=cfg_["standard_joint_topic"].as<std::string>("/motion/joint_state");
    joints_=create_subscription<crb_ros_msg::msg::JointStateData>(text("joint_topic"),sensor_qos,
      [this](const crb_ros_msg::msg::JointStateData::SharedPtr msg){
        const auto entered=debug_timing_.enter();
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        debug_timing_.lock_wait.since(entered);
        read_joint(*msg,debug_input_);
        read_pd(*msg);
      },sensor_options);
    standard_joints_=create_subscription<sensor_msgs::msg::JointState>(standard_joint_topic_,sensor_qos,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg){
        const auto entered=standard_timing_.enter();
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        standard_timing_.lock_wait.since(entered);
        read_joint(*msg,standard_input_);
      },sensor_options);
    imu_=create_subscription<sensor_msgs::msg::Imu>(text("imu_topic"),sensor_qos,
      [this](const sensor_msgs::msg::Imu::SharedPtr msg){
        const auto entered=imu_timing_.enter();
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        imu_timing_.lock_wait.since(entered);
        ++imu_messages_;
        imu_ok_=false;
        if(msg->header.frame_id!=text("imu_frame")) {
          imu_reason_="frame_id mismatch: received='"+msg->header.frame_id+
              "' expected='"+text("imu_frame")+"'";return;
        }
        imu_omega_=imu_to_base_*Eigen::Vector3d(msg->angular_velocity.x,msg->angular_velocity.y,msg->angular_velocity.z);
        if(!imu_omega_.allFinite()) {imu_reason_="non-finite angular velocity";return;}
        if(!policy_.requires_root_pose()) {
          // q_WB = q_WI * inverse(q_BI). No world position is estimated here.
          const Eigen::Quaterniond q_wi(msg->orientation.w,msg->orientation.x,
                                       msg->orientation.y,msg->orientation.z);
          if(msg->orientation_covariance[0]==-1.0) {
            imu_reason_="orientation marked unavailable (orientation_covariance[0]=-1)";return;
          }
          if(!q_wi.coeffs().allFinite() || std::abs(q_wi.norm()-1.)>.05) {
            imu_reason_="invalid orientation quaternion: norm="+std::to_string(q_wi.norm());return;
          }
          sensor_orientation_=(q_wi.normalized()*imu_to_base_.conjugate()).normalized();
        }
        imu_stamp_=rclcpp::Time(msg->header.stamp).seconds(); imu_recv_=Clock::now(); imu_ok_=true;
        imu_reason_.clear();
      },sensor_options);
    if(policy_.requires_root_pose()) pose_=create_subscription<geometry_msgs::msg::PoseStamped>(text("pose_topic"),sensor_qos,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg){
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        pose_ok_=false;
        if(msg->header.frame_id!=text("world_frame")) return;
        sensor_position_={msg->pose.position.x,msg->pose.position.y,msg->pose.position.z};
        sensor_orientation_={msg->pose.orientation.w,msg->pose.orientation.x,msg->pose.orientation.y,msg->pose.orientation.z};
        if(!sensor_position_.allFinite() || !sensor_orientation_.coeffs().allFinite() ||
           std::abs(sensor_orientation_.norm()-1.)>.05) return;
        sensor_orientation_.normalize();
        pose_stamp_=rclcpp::Time(msg->header.stamp).seconds(); pose_recv_=Clock::now(); pose_ok_=true;
      },sensor_options);
    mode_=create_subscription<std_msgs::msg::Int32>(text("mode_topic"),command_qos,
      [this](const std_msgs::msg::Int32::SharedPtr msg){
        const auto entered=mode_timing_.enter();
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        mode_timing_.lock_wait.since(entered);
        ++mode_messages_;
        const auto time=Clock::now();
        mode_received_=time; // This topic contains the RAW SDK enum.
        const int previous=mode_value_.exchange(msg->data);
        if(msg->data!=7) seen_non_debug_mode_=true;
        else if(previous!=7 && seen_non_debug_mode_) ++mode_entries_;
      },sensor_options);
    mode_client_=create_client<GetMode>(cfg_["mode_service"].as<std::string>("/get_robot_mode"),
                                       rmw_qos_profile_services_default,mode_group_);
    mode_timer_=create_wall_timer(std::chrono::milliseconds(50),[this]{poll_mode();},mode_group_);
    if(!dry_) command_=create_publisher<crb_ros_msg::msg::JointStateData>(
        text("command_topic"),command_qos);
    status_=create_publisher<std_msgs::msg::String>("~/status",1);
    diagnostics_=create_publisher<crb_ros_msg::msg::JointStateData>("~/dry_run_command",1);
    prepare_=create_service<Trigger>("~/prepare",[this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response){
      std::lock_guard<std::mutex> lock(mutex_);
      cancel_auto();
      std::string reason;
      response->success=prepare_motion(reason);
      response->message=response->success?(prepare_seconds_==0?
          "Ready at measured pose; preparation skipped; reference clock stopped":
          "Preparing selected pose; reference clock stopped"):reason;
    });
    start_=create_service<Trigger>("~/start",[this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response){
      std::lock_guard<std::mutex> lock(mutex_);
      cancel_auto();
      std::string reason;
      response->success=start_motion(reason);
      response->message=response->success?"Running selected reference":reason;
    });
    play_=create_service<Trigger>("~/play",[this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response){
      std::lock_guard<std::mutex> lock(mutex_);
      cancel_auto(); std::string reason;
      response->success=prepare_motion(reason);
      if(response->success) {play_pending_=true;response->message=prepare_seconds_==0?
          "Preparation skipped; will hand off from measured pose and play selected clip":
          "Preparing; will hand off and play selected clip when READY";}
      else response->message=reason;
    });
    stop_=create_service<Trigger>("~/stop",[this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response){
      std::lock_guard<std::mutex> lock(mutex_);
      cancel_auto();phase_=Phase::Idle; ++generation_;
      response->success=true; response->message="Idle: SDK damping while mode 7, no policy restart";
    });
    finish_=create_service<Trigger>("~/finish",[this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr response){
      std::lock_guard<std::mutex> lock(mutex_); std::string reason;
      cancel_auto();
      if(!policy_active() && phase_!=Phase::Ready && phase_!=Phase::Prepare)
        reason="No active motion to finish";
      else if(mode_value_!=7) reason="Requires motion mode 7";
      else if(!valid(reason)) {fail(reason);}
      else {
        begin_return(Clock::now()); response->success=true;
        response->message=return_seconds_>0?"Returning to the pre-prepare measured pose":"Done: SDK damping";
        return;
      }
      response->success=false; response->message=reason;
    });
    timer_=create_wall_timer(std::chrono::milliseconds(1),[this]{control();},control_group_);
    status_timer_=create_wall_timer(std::chrono::seconds(1),[this]{report();});
    worker_=std::thread([this]{worker();});
    RCLCPP_INFO(get_logger(),"SuperMimic: %s, obs=%d goal=%d total_input=%d actions=%d, checkpoint_iteration=%d root_pose=%s, dry_run=%s",
        text("bundle").c_str(),sm::kObs,policy_.goal_dim(),sm::kObs+policy_.goal_dim(),sm::kJoints,
        policy_.checkpoint_iteration(),policy_.requires_root_pose()?"required":"not_required",dry_?"true":"false");
    RCLCPP_INFO(get_logger(),"Checkpoint SHA256: %s",policy_.checkpoint_sha256().c_str());
    RCLCPP_INFO(get_logger(),"Motion: %s, frames=%d, preview_seconds=%.3f (0=full clip), bank=%zu; use motion parameter and ~/play without restarting",
        policy_.motion_name().c_str(),policy_.length(),preview_seconds_,motions_.size());
    RCLCPP_INFO(get_logger(),"Natural clip end: %s; B / ~/finish keeps the explicit return behavior",
        hold_last_frame_?"END_HOLD: stationary last-frame reference with live policy":"RETURN to measured entry pose");
    RCLCPP_INFO(get_logger(),"Sensor inputs: debug_joints='%s', standard_joints='%s', imu='%s', expected imu_frame='%s'",
        text("joint_topic").c_str(),standard_joint_topic_.c_str(),text("imu_topic").c_str(),text("imu_frame").c_str());
    RCLCPP_INFO(get_logger(),"SDK 2.2.37 position_pd: 29 joints, zero effort, integer gains, live PD feedback required for real motion");
    if(joystick_enabled_) RCLCPP_INFO(get_logger(),"Joystick %s: release X to play NEXT motion, B returns; D-pad unused; SDK modifier chords ignored",
        cfg_["joystick_topic"].as<std::string>("/joystick_events").c_str());
    RCLCPP_INFO(get_logger(),"SDK mode: topic='%s', control raw mode=7 via topic; get_robot_mode returns class 2/3/4; auto_start=%s; requires confirmed ZERO/STAND -> fresh raw topic 7; one attempt per process; manual services available",
        text("mode_topic").c_str(),auto_enabled_?"true":"false");
  }
  ~Runner() override {quit_=true; if(worker_.joinable())worker_.join();}
  void stop_for_shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancel_auto();phase_=Phase::Idle;++generation_;
  }
 private:
  void poll_mode() { // mode callback group serializes timeout and responses
    if(mode_pending_) {
      if(Clock::now()-mode_requested_<std::chrono::milliseconds(200)) return;
      mode_client_->remove_pending_request(mode_request_id_);mode_pending_=false;
    }
    if(!mode_client_->service_is_ready()) return;
    mode_pending_=true;mode_requested_=Clock::now();
    const auto requested=mode_requested_;
    auto pending=mode_client_->async_send_request(std::make_shared<GetMode::Request>(),
      [this,requested](rclcpp::Client<GetMode>::SharedFuture reply) {
        if(!mode_pending_ || requested!=mode_requested_) return;
        mode_pending_=false;
        if(Clock::now()-requested>std::chrono::milliseconds(200)) return;
        const int mode=reply.get()->mode;
        std::lock_guard<std::mutex> lock(sensor_mutex_);
        ++mode_query_messages_;mode_query_received_=Clock::now();
        mode_class_=mode; // 2=ZERO, 3=STAND, 4=WALK (includes raw mode 7).
        // 4 is ambiguous: NEVER use it to assert or overwrite raw debug mode.
        // Ignore a negative response sampled before the latest raw topic sample.
        if((mode==2 || mode==3) && requested>=mode_received_) {
          mode_value_=0;seen_non_debug_mode_=true;
        }
      });
    mode_request_id_=pending.request_id;
  }
  void read_pd(const crb_ros_msg::msg::JointStateData& msg) { // sensor_mutex_ held
    pd_valid_=false;
    const bool indexed=msg.kp.size()==msg.name.size() && msg.kd.size()==msg.name.size();
    size_t non_hand=0;
    for(const auto& name:msg.name) if(ids_.count(name)) ++non_hand;
    if(!indexed && (msg.kp.size()!=non_hand || msg.kd.size()!=non_hand)) return;
    feedback_kp_.assign(29,0.);feedback_kd_.assign(29,0.);
    std::array<bool,29> seen{};size_t compact=0;
    for(size_t j=0;j<msg.name.size();++j) {
      const auto it=ids_.find(msg.name[j]);
      if(it==ids_.end()) continue; // SDK's 20 hand entries carry no compact PD.
      const size_t k=indexed?j:compact; ++compact;
      const int dest=it->second;
      if(seen[dest] || !std::isfinite(msg.kp[k]) || !std::isfinite(msg.kd[k])) return;
      seen[dest]=true;feedback_kp_[dest]=msg.kp[k];feedback_kd_[dest]=msg.kd[k];
    }
    if(!std::all_of(seen.begin(),seen.end(),[](bool v){return v;})) return;
    pd_valid_=true;pd_received_=Clock::now();pd_stamp_=rclcpp::Time(msg.header.stamp).seconds();
  }
  bool pd_matches_locked() const { // control mutex_ and sensor_mutex_ held
    return sent_pd_ && pd_valid_ && pd_received_>=pd_changed_ && pd_stamp_>0 &&
      std::abs(now().seconds()-pd_stamp_)<=value("max_sensor_age") &&
      Seconds(Clock::now()-pd_received_).count()<=value("max_sensor_age") &&
      feedback_kp_==expected_kp_ && feedback_kd_==expected_kd_;
  }
  bool pd_ready(std::string& reason) { // control mutex_ held
    if(dry_) return true;
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    if(pd_matches_locked()) return true;
    reason="SDK PD feedback mismatch/missing; require running SDK use_default_kp_kd=false";return false;
  }
  enum class Phase {Idle,Prepare,Ready,Handoff,Run,Switch,Return,Hold,Done,Fault,EndHold};
  bool policy_active() const {return phase_==Phase::Run || phase_==Phase::Handoff || phase_==Phase::Switch || phase_==Phase::EndHold;}
  void cancel_auto() {auto_consumed_=true;auto_pending_=false;play_pending_=false;auto_reason_.clear();joystick_pending_.clear();switch_requested_.clear();}
  void joystick_message(const crb_ros_msg::msg::JoystickCmdReport& msg) { // default callback group
    const auto received=Clock::now();
    const auto action=joystick_edges_.update(msg.pressed_buttons,msg.long_pressed,msg.double_clicked,
        msg.axis_x,msg.axis_y,Seconds(received.time_since_epoch()).count());
    std::string notice;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++joystick_rx_;joystick_received_=received;
      const auto modifier=[](const auto& keys){return std::any_of(keys.begin(),keys.end(),[](uint32_t key){return key>=4;});};
      if(modifier(msg.pressed_buttons) || modifier(msg.long_pressed) || modifier(msg.double_clicked))
        joystick_pending_.clear();
      if(action==sm::JoyAction::None) return;
      std::string reason;
      if(mode_value_!=7 || phase_==Phase::Fault || !valid(reason))
        notice="Joystick ignored: requires healthy mode 7; recover faults manually";
      else if(action==sm::JoyAction::Finish) {
        cancel_auto();
        if(position_active() && phase_!=Phase::Return && phase_!=Phase::Hold) {
          begin_return(received);notice="Joystick B: returning; no next motion queued";
        } else notice="Joystick B: no new motion will start";
      } else if(action==sm::JoyAction::Advance) {
        if(phase_==Phase::Prepare || phase_==Phase::Handoff || phase_==Phase::Switch || phase_==Phase::Return ||
           !joystick_pending_.empty() || !switch_requested_.empty())
          notice="Joystick X ignored: transition already in progress";
        else {
          const auto current=policy_.motion_name();
          const auto choice=sm::next_motion(joystick_order_,current);
          cancel_auto();
          if(phase_==Phase::Ready) begin_return(received);
          joystick_choice_=choice;joystick_pending_=choice;
          notice="Joystick X: "+current+" -> "+choice+((phase_==Phase::Run || phase_==Phase::EndHold)?
            "; continuous-policy reference bridge ("+std::to_string(switch_seconds_)+"s)":
            "; prepare/handoff before playback");
        }
      }
    }
    RCLCPP_INFO(get_logger(),"%s",notice.c_str());
  }
  void joystick_play() { // default callback group; no control lock during parameter API
    std::string selected;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(joystick_pending_.empty()) return;
      if(Seconds(Clock::now()-joystick_received_).count()>.5) {joystick_pending_.clear();return;}
      if(phase_!=Phase::Idle && phase_!=Phase::Hold && phase_!=Phase::Done && phase_!=Phase::Run && phase_!=Phase::EndHold) return;
      selected=joystick_pending_;joystick_pending_.clear();
    }
    joystick_live_parameter_=true;
    const auto result=set_parameter(rclcpp::Parameter("motion",selected));
    joystick_live_parameter_=false;
    std::string reason;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Do not let a control-thread fault occurring between selection and prepare
      // be cleared by the normal manual prepare recovery path.
      if(!result.successful) reason=result.reason;
      else if(!switch_requested_.empty() || phase_==Phase::Switch) return;
      else if(phase_!=Phase::Idle && phase_!=Phase::Hold && phase_!=Phase::Done) reason="Controller state changed";
      else if(prepare_motion(reason)) play_pending_=true;
    }
    if(!reason.empty()) RCLCPP_WARN(get_logger(),"Joystick play rejected: %s",reason.c_str());
  }
  bool prepare_motion(std::string& reason) { // mutex_ held; shared manual/automatic guards
    if(phase_!=Phase::Idle && phase_!=Phase::Hold && phase_!=Phase::Done && phase_!=Phase::Fault)
      {reason="Must stop before preparing again";return false;}
    if(mode_value_!=7) {reason="Requires motion mode 7";return false;}
    if(!valid(reason) || !pd_ready(reason)) return false;
    std::lock_guard<std::mutex> policy_lock(policy_mutex_);
    ++generation_;
    initial_=state_.q; head_hold_=head_q_;
    const sm::Joints preparation=prepare_default_?default_prepare_:policy_.reference(start_frame_).q;
    // With preparation disabled, blend the policy in from measured joints.
    // Do not command the configured preparation pose or reject the measured
    // pose for differing from that unused target in start_motion().
    prepared_target_=prepare_seconds_==0?initial_:
        preparation.cwiseMax(policy_.soft_lower).cwiseMin(policy_.soft_upper);
    target_=initial_; began_=Clock::now();
    phase_=prepare_seconds_==0?Phase::Ready:Phase::Prepare;
    auto_ready_since_=began_; fault_.clear();
    return true;
  }
  bool start_motion(std::string& reason) { // mutex_ held
    if(phase_!=Phase::Ready) {reason="Call prepare and wait for READY; current="+phase_name()+" fault="+fault_;return false;}
    if(mode_value_!=7) {reason="Requires motion mode 7";return false;}
    if(!valid(reason) || !pd_ready(reason)) return false;
    if((state_.q-prepared_target_).cwiseAbs().maxCoeff()>value("joint_start_tolerance"))
      {reason="Start joint pose not reached";return false;}
    const auto ref=policy_.reference(start_frame_);
    if(policy_.requires_root_pose() && std::abs(state_.position.z()-ref.position.z())>value("root_start_height_tolerance"))
      {reason="Start height mismatch (check floor and base_link frame)";return false;}
    std::lock_guard<std::mutex> policy_lock(policy_mutex_);
    ++generation_;policy_.reset(state_,start_frame_,policy_.requires_root_pose());
    phase_=entry_seconds_>0?Phase::Handoff:Phase::Run;
    entry_from_=target_;run_target_=target_;
    began_=Clock::now();play_began_=began_;last_target_=began_;policy_due_=began_;
    consecutive_late_=0;
    return true;
  }
  void automatic_prepare(Clock::time_point time,bool sensors_valid) { // mutex_ held
    if(!auto_pending_ || phase_!=Phase::Idle) return;
    bool mode_fresh;
    {
      std::lock_guard<std::mutex> sensor_lock(sensor_mutex_);
      mode_fresh=mode_value_==7 && Seconds(time-mode_received_).count()<=0.25;
    }
    if(Seconds(time-auto_began_).count()>auto_timeout_seconds_) {
      fail("Automatic startup timed out waiting for fresh sensors/mode");return;
    }
    std::string pd_reason;
    if(!sensors_valid || !mode_fresh || !pd_ready(pd_reason)) {
      auto_stable_since_={};auto_reason_=pd_reason.empty()?"waiting_for_fresh_sensors_and_mode":pd_reason;return;
    }
    if(auto_stable_since_==Clock::time_point{}) auto_stable_since_=time;
    auto_reason_="waiting_for_stable_sensors";
    if(Seconds(time-auto_stable_since_).count()>=auto_stable_seconds_) {
      std::string reason;
      if(!prepare_motion(reason)) {fail("Automatic prepare: "+reason);return;}
      auto_reason_="preparing";
    }
  }
  bool position_active() const {
    return phase_==Phase::Prepare || phase_==Phase::Ready || policy_active() ||
           phase_==Phase::Return || phase_==Phase::Hold;
  }
  void begin_return(Clock::time_point time) { // mutex held; invalidate in-flight inference
    ++generation_;
    if(return_seconds_==0.) {phase_=Phase::Done;return;}
    return_from_=target_;
    return_to_=initial_.cwiseMax(policy_.soft_lower).cwiseMin(policy_.soft_upper);
    return_began_=time; phase_=Phase::Return;
  }
  std::string text(const char* key) const {return cfg_[key].as<std::string>();}
  double value(const char* key) const {return cfg_[key].as<double>();}
  template<class Message> void read_joint(const Message& msg,JointInput& input) {
    // sensor_mutex_ held. Stage a complete sample in the receive cache; messages
    // from two topics can never mix joint subsets or erase the other source.
    ++joint_messages_;++input.messages;input.valid=false;
    const auto decode=[&]() {
      if(msg.position.size()!=msg.name.size() || msg.velocity.size()!=msg.name.size()) {
        input.reason="array sizes differ: names="+std::to_string(msg.name.size())+
            " positions="+std::to_string(msg.position.size())+
            " velocities="+std::to_string(msg.velocity.size());return;
      }
      std::array<bool,27> seen{};
      std::array<bool,2> head_seen{};
      for(size_t j=0;j<msg.name.size();++j) {
        const auto it=ids_.find(msg.name[j]);
        if(it==ids_.end()) continue;
        const int i=it->second;
        if(i>=27) {
          const int h=i-27;
          if(head_seen[h] || !std::isfinite(msg.position[j]) || !std::isfinite(msg.velocity[j])) {
            input.reason="invalid/duplicate head joint: "+msg.name[j];return;
          }
          head_seen[h]=true;input.head[h]=msg.position[j];continue;
        }
        if(seen[i]) {input.reason="duplicate joint: "+msg.name[j];return;}
        if(!std::isfinite(msg.position[j]) || !std::isfinite(msg.velocity[j])) {
          input.reason="non-finite position/velocity: "+msg.name[j];return;
        }
        seen[i]=true;input.q[i]=msg.position[j];input.dq[i]=msg.velocity[j];
      }
      std::string missing;
      for(int i=0;i<27;++i) if(!seen[i]) {
        if(!missing.empty()) missing+=",";
        missing+=policy_.names()[i];
      }
      for(int h=0;h<2;++h) if(!head_seen[h]) missing+=h==0?",head_pitch_joint":",head_yaw_joint";
      if(!missing.empty()) {input.reason="missing joints: "+missing;return;}
      input.stamp=rclcpp::Time(msg.header.stamp).seconds();input.received=Clock::now();
      input.valid=true;input.reason.clear();
    };
    decode();
  }
  void select_joint() { // mutex_ and sensor_mutex_ held; AMP's most-recent receive ordering
    const auto time=Clock::now();
    const double stamp=now().seconds(),age=value("max_sensor_age");
    const auto fresh=[&](const JointInput& input) {
      return input.valid && input.stamp>0 && stamp-input.stamp<=age &&
          input.stamp-stamp<=.02 && Seconds(time-input.received).count()<=age;
    };
    const JointInput* selected=nullptr;
    if(fresh(debug_input_)) selected=&debug_input_;
    if(fresh(standard_input_) && (!selected || standard_input_.received>selected->received))
      selected=&standard_input_;
    joint_ok_=selected!=nullptr;
    if(selected) {
      state_.q=selected->q;state_.dq=selected->dq;head_q_=selected->head;have_joint_sample_=true;
      joint_stamp_=selected->stamp;joint_recv_=selected->received;
      joint_source_=selected==&debug_input_?"debug":"standard";
      joint_reason_.clear();
    } else {
      joint_source_="none";
      const auto unavailable=[&](const JointInput& input) {
        if(!input.valid) return input.reason;
        return "Stale sensor timestamp or receipt: stamp_age_ms="+
            std::to_string((stamp-input.stamp)*1000)+" receipt_age_ms="+
            std::to_string(Seconds(time-input.received).count()*1000);
      };
      joint_reason_=joint_messages_==0?"no messages received":
          "debug: "+unavailable(debug_input_)+"; standard: "+unavailable(standard_input_);
    }
  }
  bool valid(std::string& reason) { // mutex_ held; sensor lock never covers I/O or inference
    std::lock_guard<std::mutex> sensor_lock(sensor_mutex_);
    state_.omega=imu_omega_;
    state_.orientation=sensor_orientation_;
    if(policy_.requires_root_pose()) state_.position=sensor_position_;
    select_joint();
    if(!joint_ok_ || !imu_ok_ || (policy_.requires_root_pose() && !pose_ok_)) {
      reason.clear();
      if(!joint_ok_) reason="joints: "+joint_reason_;
      if(!imu_ok_) {if(!reason.empty()) reason+="; ";reason+="imu: "+imu_reason_;}
      if(policy_.requires_root_pose() && !pose_ok_) {
        if(!reason.empty()) reason+="; ";
        reason+="pose: need valid floor-aligned base pose";
      }
      return false;
    }
    const auto time=Clock::now(); const double stamp=now().seconds(), age=value("max_sensor_age");
    const std::array<std::pair<double,Clock::time_point>,3> sensors={{
        {joint_stamp_,joint_recv_},{imu_stamp_,imu_recv_},{pose_stamp_,pose_recv_}}};
    double min_stamp=joint_stamp_,max_stamp=joint_stamp_;
    for(int i=0;i<(policy_.requires_root_pose()?3:2);++i) {
      const auto& item=sensors[i];
      if(item.first<=0 || stamp-item.first>age || item.first-stamp>.02 || Seconds(time-item.second).count()>age) {
        const char* sensor_names[]={"joints","imu","pose"};
        reason=std::string("Stale sensor timestamp or receipt: ")+sensor_names[i]+
            " stamp_age_ms="+std::to_string((stamp-item.first)*1000)+
            " receipt_age_ms="+std::to_string(Seconds(time-item.second).count()*1000);
        return false;
      }
      min_stamp=std::min(min_stamp,item.first); max_stamp=std::max(max_stamp,item.first);
    }
    if(max_stamp-min_stamp>value("max_sensor_skew")) {
      reason="Sensor timestamps are not synchronized";return false;
    }
    if(mode_received_==Clock::time_point{} || Seconds(time-mode_received_).count()>age) {
      reason="SDK raw mode topic unavailable/stale";return false;
    }
    if(mode_query_received_==Clock::time_point{} || Seconds(time-mode_query_received_).count()>.25) {
      reason="SDK mode query unavailable/stale";return false;
    }
    try {state_.validate(policy_.requires_root_pose());} catch(const std::exception& e){reason=e.what();return false;}
    if((state_.q.array() < policy_.lower.array()-.03).any() || (state_.q.array()>policy_.upper.array()+.03).any()) {
      reason="Measured joint outside model hard limit";return false;
    }
    if(state_.orientation.toRotationMatrix()(2,2)<.5) {reason="Base tilt exceeds 60 degrees";return false;}
    return true;
  }
  void fail(const std::string& reason) {cancel_auto();phase_=Phase::Fault; fault_=reason; ++generation_;}
  void worker() {
    while(!quit_) {
      sm::State snapshot; Clock::time_point sampled{}; uint64_t generation=0;
      bool ready=false,hold_reference=false,hold_final=false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if(policy_active() && Clock::now()>=policy_due_) {
          std::string reason;
          if(mode_value_!=7) {fail("Left mode 7");}
          else if(!valid(reason)) {fail(reason);}
          else {
            // Score/play the final target for its normal control interval first.
            // END_HOLD then keeps feedback/inference alive on a stationary pose.
            if(phase_==Phase::Run && policy_.finished() && switch_requested_.empty()) {
              if(!hold_last_frame_) {begin_return(Clock::now());continue;}
              phase_=Phase::EndHold;
            }
            if(!switch_requested_.empty()) {
              try {
                std::lock_guard<std::mutex> policy_lock(policy_mutex_);
                policy_.transition_motion(motions_.at(switch_requested_),switch_seconds_);
                switch_requested_.clear();++generation_;phase_=Phase::Switch;frame_=0;start_frame_=0;
              } catch(const std::exception& e) {fail(e.what());continue;}
            }
            snapshot=state_; sampled=Clock::now(); generation=generation_; ready=true;
            hold_final=phase_==Phase::EndHold;
            hold_reference=phase_==Phase::Handoff || hold_final;
            // Preserve normal cadence, but never replay overdue 20 ms ticks
            // back-to-back using effectively identical sensor history.
            const auto scheduled=policy_due_+std::chrono::milliseconds(20);
            policy_due_=scheduled<=sampled?sampled+std::chrono::milliseconds(20):scheduled;
          }
        }
      }
      if(!ready) {std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
      // History keeps updating in HANDOFF and RUN. Generation prevents an old
      // inference crossing stop/reset; the reference clock advances only in RUN.
      try {
        const auto t=Clock::now();
        // Serialize policy state with prepare/start services, but do not block the 1 kHz loop.
        sm::Output output;
        {
          std::unique_lock<std::mutex> state_lock(mutex_);
          if(generation!=generation_ || !policy_active()) continue;
          std::lock_guard<std::mutex> policy_lock(policy_mutex_);
          state_lock.unlock();
          output=hold_final?policy_.step_hold(snapshot,policy_.length()-1):policy_.step(snapshot,!hold_reference);
        }
        const double elapsed=Seconds(Clock::now()-t).count();
        const auto commit_began=Clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        policy_commit_wait_.since(commit_began);
        if(generation!=generation_ || !policy_active()) continue;
        inference_ms_=elapsed*1000;
        if(mode_value_!=7) {fail("Left mode 7");continue;}
        if(Seconds(Clock::now()-sampled).count()>value("max_target_age")) {
          fail("Policy result expired before commit");continue;
        }
        max_compute_ms_=std::max(max_compute_ms_,inference_ms_);
        consecutive_late_=elapsed>late_compute_seconds_?consecutive_late_+1:0;
        if(elapsed>policy_.dt) ++late_updates_; // nominal 20 ms diagnostic only
        if(elapsed>max_compute_seconds_) {
          fail("Policy compute exceeded "+std::to_string(max_compute_seconds_*1000.)+" ms hard limit");continue;
        }
        if(consecutive_late_>=max_consecutive_late_) {
          fail("Policy continuously late: "+std::to_string(consecutive_late_)+
               " consecutive updates exceeded "+std::to_string(late_compute_seconds_*1000.)+" ms");continue;
        }
        const auto& ref=output.reference;
        const bool position_error=policy_.requires_root_pose() && (
            std::abs(snapshot.position.z()-ref.position.z())>policy_.height_threshold ||
            (snapshot.position-ref.position).norm()>value("max_root_tracking_error"));
        if(position_error || std::abs(snapshot.orientation.toRotationMatrix()(2,2)-
            ref.orientation.toRotationMatrix()(2,2))>policy_.tilt_threshold) {
          fail("Root tracking error exceeded limit");continue;
        }
        run_target_=output.target; last_target_=Clock::now(); frame_=output.frame;
        if(phase_==Phase::Switch && !policy_.transitioning()) {
          phase_=Phase::Run;play_began_=last_target_;
        }
        if(!hold_reference && policy_.finished())
          policy_due_=last_target_+std::chrono::duration_cast<Clock::duration>(Seconds(policy_.dt));
        if(phase_==Phase::Handoff && hold_reference && Seconds(last_target_-began_).count()>=entry_seconds_) {
          phase_=Phase::Run; play_began_=last_target_;
        }
        if(phase_==Phase::Run && !hold_reference && preview_seconds_>0 && Seconds(last_target_-play_began_).count()>=preview_seconds_) {
          begin_return(last_target_);
        }
      } catch(const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(generation==generation_ && policy_active()) fail(e.what());
      }
    }
  }
  void control() {
    const auto entered=Clock::now();
    if(last_control_!=Clock::time_point{}) control_gap_.note(Seconds(entered-last_control_).count()*1000.);
    last_control_=entered;
    std::lock_guard<std::mutex> lock(mutex_);
    control_lock_wait_.since(entered);
    const auto time=Clock::now();
    if(auto_enabled_ && !auto_consumed_ && mode_entries_.load()>0) {
      auto_consumed_=true;auto_pending_=true;auto_began_=time;auto_stable_since_={};
      auto_reason_="waiting_for_fresh_sensors_and_mode";
    }
    if(mode_value_!=7) {
      if(position_active() || auto_pending_) fail("Left mode 7");
      return; // no interference with the lower controller's other modes
    }
    std::string reason;
    const bool sensors_valid=valid(reason);
    if(!sensors_valid && position_active()) fail(reason);
    automatic_prepare(time,sensors_valid);
    if(phase_==Phase::Prepare) {
      const double progress=std::min(1.,Seconds(time-began_).count()/prepare_seconds_);
      const double blend=sm::transition_weight(progress,1.);
      target_=(1-blend)*initial_+blend*prepared_target_;
      if(progress>=1) {phase_=Phase::Ready;auto_ready_since_=time;}
    }
    if((auto_pending_ || play_pending_) && phase_==Phase::Ready) {
      std::string start_reason;
      if(start_motion(start_reason)) {auto_pending_=false;play_pending_=false;auto_reason_.clear();}
      else if(Seconds(time-auto_ready_since_).count()>auto_timeout_seconds_)
        fail("Automatic start: "+start_reason);
      else auto_reason_=start_reason;
    }
    if(phase_==Phase::Handoff) {
      const double blend=sm::transition_weight(Seconds(time-began_).count(),entry_seconds_);
      target_=(1-blend)*entry_from_+blend*run_target_;
    }
    if(phase_==Phase::Run || phase_==Phase::Switch || phase_==Phase::EndHold) target_=run_target_;
    if(phase_==Phase::Return) {
      const double elapsed=Seconds(time-return_began_).count();
      const double blend=sm::transition_weight(elapsed,return_seconds_);
      target_=(1-blend)*return_from_+blend*return_to_;
      if(elapsed>=return_seconds_) {target_=return_to_;phase_=Phase::Hold;}
    }
    if(policy_active() && Seconds(time-last_target_).count()>value("max_target_age")) fail("Policy target timeout");
    std::string pd_reason;
    if(position_active() && !pd_ready(pd_reason) &&
       time-pd_changed_>std::chrono::milliseconds(250)) fail(pd_reason);
    // Never invent initial joint/head positions before receiving a complete sample.
    if(!have_joint_sample_) return;
    const bool active=sensors_valid && position_active();
    crb_ros_msg::msg::JointStateData command;
    command.header.stamp=now(); command.header.frame_id=text("base_frame");
    command.name=policy_.names();
    command.name.push_back("head_pitch_joint");command.name.push_back("head_yaw_joint");
    command.position.resize(29); command.velocity.assign(29,0.);
    command.effort.assign(29,0.); command.kp.assign(29,0.); command.kd.resize(29);
    for(int i=0;i<27;++i) {
      // Match bounded PD at the sampled state by limiting position error.
      // This is a target envelope, NOT a guarantee of hardware torque bounds:
      // SDK feedback latency, interpolation and firmware remain downstream.
      double qdes=state_.q[i];
      if(active) {
        const double raw=sdk_kp_[i]*(target_[i]-state_.q[i])-sdk_kd_[i]*state_.dq[i];
        const double requested=std::clamp(raw,-policy_.effort[i],policy_.effort[i]);
        qdes=state_.q[i]+(requested+sdk_kd_[i]*state_.dq[i])/sdk_kp_[i];
        qdes=std::clamp(qdes,policy_.lower[i],policy_.upper[i]);
      }
      command.position[i]=qdes;
      command.kp[i]=active?sdk_kp_[i]:0.;command.kd[i]=sdk_kd_[i];
    }
    // Fill both head slots: SDK expands every command into all 17 upper joints.
    // Capture the measured head pose on prepare; do not invent zero head angles.
    for(int h=0;h<2;++h) {
      command.position[27+h]=active?head_hold_[h]:head_q_[h];
      command.kp[27+h]=active?30.:0.;command.kd[27+h]=2.;
    }
    if(!sent_pd_ || command.kp!=expected_kp_ || command.kd!=expected_kd_) {
      expected_kp_=command.kp;expected_kd_=command.kd;pd_changed_=time;sent_pd_=true;
    }
    const auto publish_began=Clock::now();
    // Keep output serialized with stop/finish/prepare. The independent sensor
    // cache remains writable even if DDS blocks here. No queued target is sent
    // after a stop service has acknowledged it.
    if(dry_) {
      // Diagnostics only at policy rate, on a topic with no hardware consumer.
      if(Seconds(time-last_diagnostic_).count()>=.02) {diagnostics_->publish(command);last_diagnostic_=time;}
    } else command_->publish(command);
    publish_time_.since(publish_began);
  }
  std::string phase_name() const { // mutex_ held
    const char* names[]={"IDLE","PREPARE","READY","HANDOFF","RUN","SWITCH","RETURN","HOLD","DONE","FAULT","END_HOLD"};
    return names[static_cast<int>(phase_)];
  }
  void report() {
    std_msgs::msg::String status;
    std::string console;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::string reason; const bool ok=valid(reason);
      std::lock_guard<std::mutex> sensor_lock(sensor_mutex_);
      status.data=phase_name()+" dry_run="+(dry_?"true":"false")+
          " motion="+policy_.motion_name()+" frames="+std::to_string(policy_.length())+
          " preview_seconds="+std::to_string(preview_seconds_)+
          " hold_last_frame_on_finish="+(hold_last_frame_?std::string("true"):std::string("false"))+
          " joystick_choice="+joystick_choice_+" joystick_pending="+joystick_pending_+
          " joystick_rx="+std::to_string(joystick_rx_)+
          " frame="+std::to_string(frame_)+" inference_ms="+std::to_string(inference_ms_)+
          " late_updates="+std::to_string(late_updates_)+
          " consecutive_late="+std::to_string(consecutive_late_)+
          " max_policy_compute_ms="+std::to_string(max_compute_ms_)+
          " sensors="+(ok?"OK":reason)+" fault="+fault_+
          " joint_rx="+std::to_string(joint_messages_)+" imu_rx="+std::to_string(imu_messages_)+
          " mode="+std::to_string(mode_value_.load())+" mode_rx="+std::to_string(mode_messages_)+
          " joint_debug_rx="+std::to_string(debug_input_.messages)+
          " joint_standard_rx="+std::to_string(standard_input_.messages)+" joint_source="+joint_source_+
          " auto_start="+(!auto_enabled_?std::string("disabled"):auto_pending_?std::string("pending"):
                         auto_consumed_?std::string("consumed"):std::string("waiting_mode_edge"))+
          " auto_reason="+auto_reason_+" command_mode=sdk237_position_pd"+
          " mode_query_rx="+std::to_string(mode_query_messages_)+" mode_class="+std::to_string(mode_class_)+
          " pd_feedback="+(dry_?std::string("dry_run_unverified"):pd_matches_locked()?std::string("matched"):std::string("unmatched"));
      // Frame/counter changes stay on the ROS topic. The console shows changes
      // relevant to the operator, leaving each X-switch notice easy to read.
      const auto key=phase_name()+"|"+policy_.motion_name()+"|"+joystick_pending_+"|"+
          std::to_string(mode_value_.load())+"|"+(ok?"ok":"missing")+"|"+fault_;
      if(key!=last_console_key_) {
        last_console_key_=key;
        console="State: "+phase_name()+" | current="+policy_.motion_name()+" | frame="+std::to_string(frame_)+
            " | next="+(joystick_pending_.empty()?std::string("none"):joystick_pending_)+
            " | sensors="+(ok?std::string("OK"):reason)+" | fault="+(fault_.empty()?std::string("none"):fault_);
      }
    }
    const auto add=[&](const char* name,const MaxMs& metric) {
      status.data+=" "+std::string(name)+"="+std::to_string(metric.value.load(std::memory_order_relaxed));
    };
    add("max_control_gap_ms",control_gap_);add("max_control_lock_ms",control_lock_wait_);
    add("max_publish_ms",publish_time_);add("max_report_io_ms",report_io_time_);
    add("max_policy_commit_lock_ms",policy_commit_wait_);
    add("max_debug_gap_ms",debug_timing_.gap);add("max_standard_gap_ms",standard_timing_.gap);
    add("max_imu_gap_ms",imu_timing_.gap);add("max_mode_gap_ms",mode_timing_.gap);
    add("max_debug_lock_ms",debug_timing_.lock_wait);add("max_imu_lock_ms",imu_timing_.lock_wait);
    status.data+=" debug_gaps_over100="+std::to_string(debug_timing_.over_100ms.load())+
        " standard_gaps_over100="+std::to_string(standard_timing_.over_100ms.load())+
        " imu_gaps_over100="+std::to_string(imu_timing_.over_100ms.load())+
        " mode_gaps_over100="+std::to_string(mode_timing_.over_100ms.load());
    const auto io_began=Clock::now();
    status_->publish(status);
    if(console_full_status_) RCLCPP_INFO(get_logger(),"%s",status.data.c_str());
    else if(!console.empty()) RCLCPP_INFO(get_logger(),"%s",console.c_str());
    report_io_time_.since(io_began);
  }
  YAML::Node cfg_; sm::Policy policy_;
  bool auto_enabled_=false,auto_consumed_=false,auto_pending_=false,play_pending_=false;
  std::map<std::string,std::shared_ptr<const sm::MotionClip>> motions_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr motion_callback_;
  sm::JoystickEdges joystick_edges_;
  bool joystick_enabled_=false,joystick_live_parameter_=false; // parameter flag: default callback group only
  bool console_full_status_=false;
  std::string last_console_key_;
  uint64_t joystick_rx_=0;
  std::vector<std::string> joystick_order_;
  std::string joystick_choice_,joystick_pending_,switch_requested_;
  Clock::time_point joystick_received_{};
  rclcpp::Subscription<crb_ros_msg::msg::JoystickCmdReport>::SharedPtr joystick_;
  rclcpp::TimerBase::SharedPtr joystick_timer_;
  bool seen_non_debug_mode_=false; // sensor_mutex_ held
  sm::Joints sdk_kp_,sdk_kd_;
  Eigen::Vector2d head_q_=Eigen::Vector2d::Zero(),head_hold_=Eigen::Vector2d::Zero();
  std::vector<double> expected_kp_,expected_kd_,feedback_kp_,feedback_kd_;
  bool sent_pd_=false,pd_valid_=false;
  Clock::time_point pd_changed_{},pd_received_{};
  double pd_stamp_=0.;
  uint64_t mode_query_messages_=0;
  int mode_class_=0; // coarse service classification, never a raw mode enum
  int64_t mode_request_id_=0;
  bool mode_pending_=false;
  Clock::time_point mode_requested_{};
  std::atomic<uint64_t> mode_entries_{0};
  Clock::time_point mode_received_{},mode_query_received_{},auto_began_{},auto_stable_since_{},auto_ready_since_{};
  double auto_stable_seconds_=0.5,auto_timeout_seconds_=5.;
  double max_compute_seconds_=0.04,late_compute_seconds_=0.025,max_compute_ms_=0.;
  int max_consecutive_late_=20,consecutive_late_=0;
  uint64_t late_updates_=0;
  std::string auto_reason_;
  bool dry_=true,prepare_default_=false; int start_frame_=0,frame_=0;
  double prepare_seconds_=6,preview_seconds_=10,inference_ms_=0,entry_seconds_=0,return_seconds_=0;
  bool hold_last_frame_=false;
  double switch_seconds_=.8;
  // Lock order: control mutex_ -> sensor_mutex_ or policy_mutex_.
  // Sensor callbacks take ONLY sensor_mutex_; never wait for output or ONNX.
  std::mutex mutex_,sensor_mutex_,policy_mutex_; std::atomic<bool> quit_{false}; std::atomic<int> mode_value_{0};
  std::thread worker_; Phase phase_=Phase::Idle; std::string fault_; uint64_t generation_=0;
  sm::State state_; sm::Joints target_=sm::Joints::Zero(),initial_=sm::Joints::Zero(),prepared_target_=sm::Joints::Zero();
  sm::Joints entry_from_=sm::Joints::Zero(),run_target_=sm::Joints::Zero(),return_from_=sm::Joints::Zero(),return_to_=sm::Joints::Zero();
  sm::Joints default_prepare_=sm::Joints::Zero();
  Eigen::Quaterniond imu_to_base_;
  std::unordered_map<std::string,int> ids_;
  bool have_joint_sample_=false,joint_ok_=false,imu_ok_=false,pose_ok_=false;
  uint64_t joint_messages_=0,imu_messages_=0,mode_messages_=0;
  JointInput debug_input_,standard_input_;
  Eigen::Vector3d imu_omega_=Eigen::Vector3d::Zero(),sensor_position_=Eigen::Vector3d::Zero();
  Eigen::Quaterniond sensor_orientation_=Eigen::Quaterniond::Identity();
  ReceiveTiming debug_timing_,standard_timing_,imu_timing_,mode_timing_;
  MaxMs control_gap_,control_lock_wait_,publish_time_,report_io_time_,policy_commit_wait_;
  Clock::time_point last_control_{};
  std::string standard_joint_topic_,joint_source_="none";
  std::string joint_reason_="no messages received",imu_reason_="no messages received";
  double joint_stamp_=0,imu_stamp_=0,pose_stamp_=0;
  Clock::time_point joint_recv_{},imu_recv_{},pose_recv_{},began_{},play_began_{},return_began_{},last_target_{},policy_due_{},last_diagnostic_{};
  rclcpp::Subscription<crb_ros_msg::msg::JointStateData>::SharedPtr joints_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr standard_joints_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mode_;
  rclcpp::Publisher<crb_ros_msg::msg::JointStateData>::SharedPtr command_,diagnostics_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_;
  rclcpp::Service<Trigger>::SharedPtr prepare_,start_,stop_,finish_,play_;
  rclcpp::Client<GetMode>::SharedPtr mode_client_;
  rclcpp::TimerBase::SharedPtr timer_,status_timer_,mode_timer_;
  rclcpp::CallbackGroup::SharedPtr sensor_group_,control_group_,mode_group_;
};

int main(int argc,char** argv) {
  // SDK caches its last command indefinitely. On a normal SIGINT/SIGTERM keep
  // ROS alive briefly to replace position tracking with damping before exit.
  // SIGKILL, process crashes and transport failure still require SDK watchdogs.
  rclcpp::init(argc,argv,rclcpp::InitOptions(),rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT,request_shutdown);std::signal(SIGTERM,request_shutdown);
  try {
    const auto args=rclcpp::remove_ros_arguments(argc,argv);
    if(args.size()!=2) throw std::runtime_error("Usage: supermimic_runner CONFIG.yaml [--ros-args -p dry_run:=true]");
    auto node=std::make_shared<Runner>(std::filesystem::absolute(args[1]));
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),4);
    executor.add_node(node);
    std::exception_ptr spin_error;
    std::atomic<bool> spin_done{false};
    std::thread executor_thread([&]{
      try {executor.spin();} catch(...) {spin_error=std::current_exception();}
      spin_done=true;
    });
    while(rclcpp::ok() && !shutdown_requested && !spin_done)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    node->stop_for_shutdown();
    if(!spin_done && rclcpp::ok()) std::this_thread::sleep_for(std::chrono::milliseconds(150));
    executor.cancel();executor_thread.join();executor.remove_node(node);node.reset();
    if(spin_error) std::rethrow_exception(spin_error);
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';rclcpp::shutdown();return 1;}
  rclcpp::shutdown();
}

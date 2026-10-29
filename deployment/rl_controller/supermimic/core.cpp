#include "core.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <map>

namespace supermimic {
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
// Motion/source metadata may differ; all model/control fields must match.
std::string contract(const YAML::Node& m) {
  const std::unordered_set<std::string> variable={"clip","frames","motion_file",
    "motion_sha256","source_motion","source_motion_sha256","actor_file"};
  std::map<std::string,std::string> fields;
  for(const auto& item:m) {
    const auto key=item.first.as<std::string>();
    if(!variable.count(key)) fields[key]=YAML::Dump(item.second);
  }
  std::string out;
  for(const auto& [key,value]:fields) out+=key+"="+value+"\n";
  return out;
}
bool same_file_bytes(const std::filesystem::path& a,const std::filesystem::path& b) {
  if(!std::filesystem::is_regular_file(a) || !std::filesystem::is_regular_file(b) ||
     std::filesystem::file_size(a)!=std::filesystem::file_size(b)) return false;
  if(std::filesystem::equivalent(a,b)) return true;
  std::ifstream x(a,std::ios::binary),y(b,std::ios::binary);
  std::array<char,65536> u{},v{};
  while(x) {
    x.read(u.data(),u.size()); y.read(v.data(),v.size());
    if(x.gcount()!=y.gcount() || !std::equal(u.begin(),u.begin()+x.gcount(),v.begin())) return false;
  }
  return x.eof() && !y.bad();
}
double yaw(const Eigen::Quaterniond& q) {
  const auto r = q.normalized().toRotationMatrix();
  return std::atan2(r(1,0), r(0,0));
}
}
void State::validate(bool require_position) const {
  require(q.allFinite() && dq.allFinite() && (!require_position || position.allFinite()) && omega.allFinite() &&
          orientation.coeffs().allFinite(), "Nonfinite robot state");
  require(std::abs(orientation.norm() - 1.0) < .05, "Invalid robot quaternion norm");
}
Policy::Policy(const std::filesystem::path& bundle) {
  require(std::filesystem::is_regular_file(bundle / "manifest.json"),
          "Deployment bundle missing: " + bundle.string() +
          ". Export a valid checkpoint first; old 649-goal weights cannot replace a 640-goal policy.");
  const auto m = YAML::LoadFile((bundle / "manifest.json").string());
  require(m["format"].as<std::string>() == "supermimic-hardware-v1" &&
          m["robot"].as<std::string>() == "casbot02", "Unsupported hardware bundle");
  anchor_obs_=m["anchor_obs"].as<bool>();
  goal_dim_=m["goal_dim"].as<int>();
  checkpoint_iteration_=m["checkpoint_iteration"].as<int>(0);
  checkpoint_sha256_=m["checkpoint_sha256"].as<std::string>("unknown");
  require(m["obs_dim"].as<int>() == kObs && goal_dim_ == (anchor_obs_?649:640) &&
          m["history"].as<int>() == kHistory && m["future_frames"].as<int>() == 10 &&
          m["reset_goal_advance"].as<int>() == 1,
          "Checkpoint IO contract mismatch");
  dt = m["control_dt"].as<double>();
  require(std::abs(dt-.02) < 1e-9, "Expected 50 Hz checkpoint");
  height_threshold = m["height_threshold"].as<double>();
  tilt_threshold = m["tilt_threshold"].as<double>();
  names_ = m["sdk_joint_names"].as<std::vector<std::string>>();
  joint_names_ = m["joint_names"].as<std::vector<std::string>>();
  require(names_.size() == kJoints && std::unordered_set<std::string>(names_.begin(), names_.end()).size() == kJoints,
          "SDK mapping must contain 27 unique joints");
  std::vector<std::string> expected;
  for (const auto* side : {"l", "r"})
    for (int j=1; j<=6; ++j) expected.push_back("leg_"+std::string(side)+std::to_string(j)+"_joint");
  expected.push_back("waist_yaw_joint");
  for (const auto* side : {"left", "right"})
    for (const auto* joint : {"shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll"})
      expected.push_back(std::string(side)+"_"+joint+"_joint");
  require(names_ == expected, "Unexpected hardware joint mapping");
  require(m["motors"].size() == kJoints && m["joint_limits"].size() == kJoints &&
          m["soft_joint_limits"].size() == kJoints && m["joint_names"].size() == kJoints,
          "Motor/limit count mismatch");
  for (int i=0; i<kJoints; ++i) {
    const auto p = m["motors"][i];
    require(p["name"].as<std::string>() == m["joint_names"][i].as<std::string>(), "Motor order mismatch");
    q0[i]=p["q0"].as<double>(); scale[i]=p["scale"].as<double>();
    kp[i]=p["kp"].as<double>(); kd[i]=p["kd"].as<double>(); effort[i]=p["effort_limit_nm"].as<double>();
    lower[i]=m["joint_limits"][i][0].as<double>(); upper[i]=m["joint_limits"][i][1].as<double>();
    soft_lower[i]=m["soft_joint_limits"][i][0].as<double>(); soft_upper[i]=m["soft_joint_limits"][i][1].as<double>();
    require(std::isfinite(q0[i]) && kp[i]>0 && kd[i]>=0 && effort[i]>0 && lower[i]<upper[i] &&
            std::isfinite(kp[i]+kd[i]+effort[i]+lower[i]+upper[i]+scale[i]+soft_lower[i]+soft_upper[i]) &&
            std::abs(scale[i] - .25*effort[i]/kp[i]) < 1e-8,
            "Invalid motor constants");
  }
  contract_=contract(m);
  actor_path_=bundle/m["actor_file"].as<std::string>();
  motion_=load_motion(bundle);
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(2); options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_=std::make_unique<Ort::Session>(env_, (bundle/m["actor_file"].as<std::string>()).c_str(), options);
  require(session_->GetInputCount()==2 && session_->GetOutputCount()==1, "Expected two actor inputs and one output");
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i=0; i<2; ++i) {
    auto name=session_->GetInputNameAllocated(i,allocator);
    auto info=session_->GetInputTypeInfo(i);
    auto tensor=info.GetTensorTypeAndShapeInfo();
    require(std::string(name.get())==(i==0?"obs_hist":"goal") &&
            tensor.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            tensor.GetShape()==std::vector<int64_t>({1,i==0?kObs:goal_dim_}), "Actor input schema mismatch");
  }
  auto out_name=session_->GetOutputNameAllocated(0,allocator);
  auto out_info=session_->GetOutputTypeInfo(0);
  auto out_tensor=out_info.GetTensorTypeAndShapeInfo();
  require(std::string(out_name.get())=="action" && out_tensor.GetShape()==std::vector<int64_t>({1,27}) &&
          out_tensor.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "Actor output schema mismatch");
  // Warm the actual ONNX graph and reject nonfinite outputs before the ROS
  // runner creates any command publisher. Real start still resets all history.
  State neutral; neutral.q=q0;
  reset(neutral,0,false);
  step(neutral);
  initialized_=false;
}
std::shared_ptr<const MotionClip> Policy::load_motion(const std::filesystem::path& bundle) const {
  const auto m=YAML::LoadFile((bundle/"manifest.json").string());
  require(contract(m)==contract_, "Motion bank model/control contract mismatch: "+bundle.string());
  require(same_file_bytes(actor_path_,bundle/m["actor_file"].as<std::string>()),
          "Motion bank actor differs from the loaded policy: "+bundle.string());
  std::ifstream file(bundle / m["motion_file"].as<std::string>(), std::ios::binary);
  char magic[8]; uint32_t count=0, width=0;
  file.read(magic,8); file.read(reinterpret_cast<char*>(&count),4); file.read(reinterpret_cast<char*>(&width),4);
  require(file.good() && std::string(magic,8)=="SMREF001" && width==125 && count>=2 && count<10000000 &&
          count == m["frames"].as<uint32_t>(), "Invalid motion header (little-endian float32 expected)");
  auto clip=std::make_shared<MotionClip>();
  clip->name=m["clip"].as<std::string>();
  clip->frames.resize(count);
  for (auto& f : clip->frames) {
    std::array<float,125> row{};
    require(static_cast<bool>(file.read(reinterpret_cast<char*>(row.data()),sizeof(row))), "Truncated motion");
    for (float x : row) require(std::isfinite(x), "Nonfinite reference data");
    f.position=Eigen::Map<Eigen::Vector3f>(row.data()).cast<double>();
    f.orientation=Eigen::Quaterniond(row[3],row[4],row[5],row[6]);
    require(std::abs(f.orientation.norm()-1.)<.01, "Invalid reference quaternion");
    f.orientation.normalize();
    f.q=Eigen::Map<Eigen::Matrix<float,27,1>>(row.data()+7).cast<double>();
    f.dq=Eigen::Map<Eigen::Matrix<float,27,1>>(row.data()+34).cast<double>();
    std::copy(row.begin()+61,row.end(),f.features.begin());
  }
  require(file.peek()==std::char_traits<char>::eof(), "Unexpected motion trailing bytes");
  return clip;
}
void Policy::select_motion(std::shared_ptr<const MotionClip> clip) {
  require(clip && clip->frames.size()>=2,"Invalid selected motion");
  motion_=std::move(clip);
  bridge_.clear(); bridge_cursor_=0;
  initialized_=false; first_=true; frame_=start_frame_=0;
  alignment_.setIdentity(); origin_.setZero(); anchor_.setZero();
}
void Policy::transition_motion(std::shared_ptr<const MotionClip> clip, double seconds) {
  require(initialized_ && !first_ && !transitioning(), "Live switch requires an active policy");
  require(!anchor_obs_, "Live X switching currently supports proprio policies only");
  require(clip && clip->frames.size()>=2 && std::isfinite(seconds) && seconds>=.2 && seconds<=5.,
          "Invalid live reference transition");
  const int steps=static_cast<int>(std::ceil(seconds/dt));
  const double duration=steps*dt;
  const auto a=last_reference_;
  auto b=clip->frames.front();
  // Features are invariant to global XY/yaw. Align only the virtual bridge,
  // so changing a clip's original world origin cannot invent travel or spins.
  const Eigen::Quaterniond heading(Eigen::AngleAxisd(yaw(a.orientation)-yaw(b.orientation),Eigen::Vector3d::UnitZ()));
  b.orientation=heading*b.orientation; b.position.head<2>()=a.position.head<2>();
  const auto world_lin=[](const ReferenceFrame& f) -> Eigen::Vector3d {
    return Eigen::AngleAxisd(yaw(f.orientation),Eigen::Vector3d::UnitZ())*
      Eigen::Vector3d(f.features[60],f.features[61],f.features[62]);
  };
  const Eigen::Vector3d va=world_lin(a), vb=world_lin(b);
  const Eigen::Vector3d wa(a.features[57],a.features[58],a.features[59]);
  const Eigen::Vector3d wb(b.features[57],b.features[58],b.features[59]);
  const auto exp=[](const Eigen::Vector3d& v) -> Eigen::Quaterniond {
    const double angle=v.norm();
    return angle<1e-12?Eigen::Quaterniond::Identity():Eigen::Quaterniond(Eigen::AngleAxisd(angle,v/angle));
  };
  const auto orientation=[&](double t) -> Eigen::Quaterniond {
    const double u=std::clamp(t/duration,0.,1.);
    const double w=u*u*u*(10.+u*(-15.+6.*u));
    return (a.orientation*exp(wa*t)).slerp(w,b.orientation*exp(wb*(t-duration))).normalized();
  };
  std::vector<ReferenceFrame> bridge; bridge.reserve(steps+1);
  for(int i=0;i<=steps;++i) {
    const double t=i*dt,u=t/duration;
    const double w=u*u*u*(10.+u*(-15.+6.*u));
    const double dw=30.*u*u*(1.-u)*(1.-u)/duration;
    ReferenceFrame f;
    const Joints left=a.q+a.dq*t,right=b.q+b.dq*(t-duration);
    f.q=(1.-w)*left+w*right;
    f.dq=(1.-w)*a.dq+w*b.dq+dw*(right-left);
    const Eigen::Vector3d pl=a.position+va*t,pr=b.position+vb*(t-duration);
    f.position=(1.-w)*pl+w*pr; f.orientation=orientation(t);
    const Eigen::Vector3d lin=Eigen::AngleAxisd(-yaw(f.orientation),Eigen::Vector3d::UnitZ())*
      ((1.-w)*va+w*vb+dw*(pr-pl));
    constexpr double eps=1e-5;
    const Eigen::AngleAxisd delta(orientation(t-eps).conjugate()*orientation(t+eps));
    const Eigen::Vector3d omega=delta.axis()*delta.angle()/(2.*eps);
    const Eigen::Vector3d gravity=-f.orientation.toRotationMatrix().row(2).transpose();
    for(int j=0;j<27;++j) {f.features[j]=f.q[j]-q0[j]; f.features[27+j]=f.dq[j];}
    for(int j=0;j<3;++j) {f.features[54+j]=gravity[j];f.features[57+j]=omega[j];f.features[60+j]=lin[j];}
    f.features[63]=f.position.z();
    if(i==0) f=a;
    if(i==steps) f=b; // exact original 64-feature endpoint, including velocities
    bridge.push_back(f);
  }
  motion_=std::move(clip); bridge_=std::move(bridge); bridge_cursor_=1;
  frame_=1;start_frame_=0;
  alignment_.setIdentity();origin_=motion_->frames.front().position;anchor_=origin_;
  // No reset(): first_, history_ and previous_ deliberately remain untouched.
}
std::array<float,kFrame> Policy::observe(const State& state) const {
  std::array<float,kFrame> f{};
  for (int i=0; i<27; ++i) { f[i]=state.q[i]-q0[i]; f[27+i]=state.dq[i]; f[60+i]=previous_[i]; }
  const Eigen::Vector3d gravity = -state.orientation.normalized().toRotationMatrix().row(2).transpose();
  for (int i=0; i<3; ++i) { f[54+i]=state.omega[i]; f[57+i]=gravity[i]; }
  return f;
}
void Policy::reset(const State& state, int start_frame, bool align) {
  state.validate(anchor_obs_);
  require(start_frame>=0 && start_frame<length()-1,"Start frame outside reference");
  // Proprio policies consume gravity, not measured position or world heading.
  // Leave reference coordinates alone even if the caller requests alignment.
  align = align && anchor_obs_;
  alignment_=Eigen::AngleAxisd(align?yaw(state.orientation)-yaw(motion_->frames[start_frame].orientation):0., Eigen::Vector3d::UnitZ()).toRotationMatrix();
  origin_=motion_->frames[start_frame].position; anchor_=origin_;
  if (align) anchor_.head<2>()=state.position.head<2>();
  previous_.setZero();
  const auto f=observe(state);
  for(auto& h:history_) h=f;
  start_frame_=start_frame; frame_=start_frame+1; first_=true; initialized_=true;
  bridge_.clear();bridge_cursor_=0;last_reference_=reference(start_frame);
}
ReferenceFrame Policy::reference(int frame) const {
  require(frame>=0 && frame<length(),"Reference frame out of range");
  auto f=motion_->frames[frame];
  f.position=alignment_*(f.position-origin_)+anchor_;
  f.orientation=Eigen::Quaterniond(alignment_)*f.orientation;
  return f;
}
Output Policy::step(const State& state, bool advance_reference) {
  return step_reference(state,advance_reference,start_frame_);
}
Output Policy::step_hold(const State& state, int frame) {
  return step_reference(state,false,frame);
}
Output Policy::step_reference(const State& state, bool advance_reference, int held_frame) {
  state.validate(anchor_obs_);
  require(initialized_ && (!advance_reference || !finished()), "Policy is uninitialized or motion has ended");
  require(held_frame>=0 && held_frame<length(), "Held reference frame out of range");
  if (!first_) {
    for(int i=0;i<kHistory-1;++i) history_[i]=history_[i+1];
    history_.back()=observe(state);
  }
  first_=false;
  const bool bridge=transitioning();
  require(!bridge || advance_reference,"Cannot freeze a live reference bridge");
  Output out; out.frame=bridge?0:(advance_reference?std::min(frame_,length()-1):held_frame); out.goal.resize(goal_dim_);
  out.reference=bridge?bridge_.at(bridge_cursor_):reference(out.frame);
  if(!advance_reference) {
    out.reference.dq.setZero();
    std::fill(out.reference.features.begin()+27,out.reference.features.begin()+54,0.f);
    std::fill(out.reference.features.begin()+57,out.reference.features.begin()+63,0.f);
  }
  int dest=0, offset=0;
  for(int size : {27,27,3,3,27}) {
    for(const auto& h:history_) for(int i=0;i<size;++i) out.obs[dest++]=h[offset+i];
    offset+=size;
  }
  for(int t=0;t<10;++t) {
    const size_t index=bridge_cursor_+t;
    const auto& f=bridge?(index<bridge_.size()?bridge_[index]:motion_->frames[std::min<int>(index-(bridge_.size()-1),length()-1)]):
      motion_->frames[advance_reference?std::min(frame_+t,length()-1):held_frame];
    std::copy(f.features.begin(),f.features.end(),out.goal.begin()+t*64);
    if(!advance_reference) {
      // Fixed pose means zero reference velocity, rather than a moving clip
      // window replayed forever. Keep joints, gravity and reference height.
      std::fill(out.goal.begin()+t*64+27,out.goal.begin()+t*64+54,0.f);
      std::fill(out.goal.begin()+t*64+57,out.goal.begin()+t*64+63,0.f);
    }
  }
  if(anchor_obs_) {
    const auto f=reference(out.frame);
    const Eigen::Matrix3d r=state.orientation.normalized().toRotationMatrix();
    const Eigen::Vector3d pos=r.transpose()*(f.position-state.position);
    const Eigen::Matrix3d ori=r.transpose()*f.orientation.toRotationMatrix();
    for(int i=0;i<3;++i) {
      out.goal[640+i]=pos[i];
      for(int j=0;j<2;++j) out.goal[643+i*2+j]=ori(i,j);
    }
  }
  auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  const int64_t os[]={1,kObs}, gs[]={1,goal_dim_};
  std::array<Ort::Value,2> inputs={
    Ort::Value::CreateTensor<float>(memory,out.obs.data(),kObs,os,2),
    Ort::Value::CreateTensor<float>(memory,out.goal.data(),goal_dim_,gs,2)};
  const char* names[]={"obs_hist","goal"}; const char* outputs[]={"action"};
  auto result=session_->Run(Ort::RunOptions{nullptr},names,inputs.data(),2,outputs,1);
  out.action=Eigen::Map<Eigen::Matrix<float,27,1>>(result[0].GetTensorMutableData<float>()).cast<double>();
  out.target=q0+scale.cwiseProduct(out.action); // No AMP clip, tanh, EMA, or position clamp.
  require(out.action.allFinite() && out.target.allFinite(), "Nonfinite actor output");
  previous_=out.action;
  last_reference_=out.reference;
  if(bridge) {
    if(++bridge_cursor_>=bridge_.size()) {bridge_.clear();bridge_cursor_=0;}
  } else if(advance_reference) ++frame_;
  return out;
}
Joints Policy::torque(const State& state, const Joints& target, double gain) const {
  state.validate(anchor_obs_);
  require(target.allFinite() && std::isfinite(gain) && gain>=0 && gain<=1,"Invalid PD command");
  return (gain*kp.cwiseProduct(target-state.q)-kd.cwiseProduct(state.dq)).cwiseMax(-effort).cwiseMin(effort);
}
} // namespace supermimic

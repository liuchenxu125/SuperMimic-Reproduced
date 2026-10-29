#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace supermimic {
constexpr int kJoints = 27, kFrame = 87, kHistory = 10, kObs = 870;
using Joints = Eigen::Matrix<double, kJoints, 1>;
struct State {
  Joints q = Joints::Zero(), dq = Joints::Zero();
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity(); // base_link -> world
  Eigen::Vector3d omega = Eigen::Vector3d::Zero(); // base_link frame
  void validate(bool require_position = true) const;
};
struct ReferenceFrame {
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  Joints q, dq;
  std::array<float, 64> features;
};
struct Output {
  std::array<float, kObs> obs;
  std::vector<float> goal;
  Joints action, target;
  ReferenceFrame reference;
  int frame = 0;
};
struct MotionClip {
  std::string name;
  std::vector<ReferenceFrame> frames;
};
class Policy {
 public:
  explicit Policy(const std::filesystem::path& bundle);
  // Load/check the motion bank at startup, before any motor publisher exists.
  std::shared_ptr<const MotionClip> load_motion(const std::filesystem::path& bundle) const;
  void select_motion(std::shared_ptr<const MotionClip> clip);
  // Live proprio-only reference bridge; preserves measured history/last action.
  void transition_motion(std::shared_ptr<const MotionClip> clip, double seconds);
  bool transitioning() const { return !bridge_.empty(); }
  const std::string& motion_name() const { return motion_->name; }
  void reset(const State& state, int start_frame = 0, bool align = true);
  // Held calls update measured history/actions but use a stationary start pose
  // for all future reference frames, without advancing the motion cursor.
  Output step(const State& state, bool advance_reference = true);
  // Keep inference/history active on a stationary reference at any valid frame,
  // including the final frame after normal playback has finished. No reset.
  Output step_hold(const State& state, int frame);
  ReferenceFrame reference(int frame) const;
  Joints torque(const State& state, const Joints& target, double gain = 1.0) const;
  bool finished() const { return frame_ >= length(); }
  int length() const { return static_cast<int>(motion_->frames.size()); }
  int goal_dim() const { return goal_dim_; }
  int checkpoint_iteration() const { return checkpoint_iteration_; }
  const std::string& checkpoint_sha256() const { return checkpoint_sha256_; }
  bool requires_root_pose() const { return anchor_obs_; }
  const std::vector<std::string>& names() const { return names_; }
  const std::vector<std::string>& joint_names() const { return joint_names_; }
  Joints q0, scale, kp, kd, effort, lower, upper, soft_lower, soft_upper;
  double dt = .02, height_threshold = .25, tilt_threshold = .8;
 private:
  Output step_reference(const State& state, bool advance_reference, int held_frame);
  std::array<float, kFrame> observe(const State&) const;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "supermimic"};
  std::unique_ptr<Ort::Session> session_;
  std::shared_ptr<const MotionClip> motion_;
  std::vector<ReferenceFrame> bridge_;
  size_t bridge_cursor_ = 0;
  ReferenceFrame last_reference_;
  std::string contract_;
  std::filesystem::path actor_path_;
  std::vector<std::string> names_;
  std::vector<std::string> joint_names_;
  std::array<std::array<float, kFrame>, kHistory> history_{};
  Joints previous_ = Joints::Zero();
  Eigen::Matrix3d alignment_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero(), anchor_ = Eigen::Vector3d::Zero();
  int frame_ = 0, start_frame_ = 0;
  bool initialized_ = false, first_ = true;
  int goal_dim_ = 649;
  bool anchor_obs_ = true;
  int checkpoint_iteration_ = 0;
  std::string checkpoint_sha256_;
};
} // namespace supermimic

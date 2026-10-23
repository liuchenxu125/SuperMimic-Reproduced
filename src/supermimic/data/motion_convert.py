"""CPU conversion of humanoid retargeted CSVs to mjlab-compatible motion arrays.

Reimplemented from the repository's conversion tests and the installed mjlab
MotionLoader conventions. This is not a recovery of the missing upstream file.
CSV quaternions are xyzw; output quaternions are wxyz. Positions are metres,
joint angles are radians, and body velocities refer to link frame origins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass
class ResampledMotion:
    fps: float
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_vel: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.root_pos)


@dataclass
class BodyMotion:
    pos_w: np.ndarray
    quat_w: np.ndarray
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray
    body_names: tuple[str, ...]


def load_csv(path: str | Path, num_joints: int = 29) -> np.ndarray:
    """Read a headerless CSV: xyz, xyzw, then num_joints hinge angles."""
    raw = np.loadtxt(Path(path).expanduser(), delimiter=",", ndmin=2)
    _validate_motion(raw, num_joints)
    return raw


def _validate_motion(raw: np.ndarray, num_joints: int | None = None) -> None:
    width = 7 + num_joints if num_joints is not None else raw.shape[-1]
    if raw.ndim != 2 or width < 8 or raw.shape[1] != width or raw.shape[0] < 2:
        raise ValueError(f"Expected at least 2 frames with {width} columns, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Motion contains NaN or infinite values")
    if np.any(np.linalg.norm(raw[:, 3:7], axis=-1) < 1e-12):
        raise ValueError("Motion contains a zero quaternion")


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the frame of unit wxyz quaternions."""
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).inv().apply(v)


def _so3_derivative(rotations: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity, central difference with repeated endpoint samples."""
    r = Rotation.from_quat(rotations[:, [1, 2, 3, 0]])
    if len(rotations) == 2:
        return np.repeat(((r[1] * r[0].inv()).as_rotvec() / dt)[None], 2, axis=0)
    omega = (r[2:] * r[:-2].inv()).as_rotvec() / (2 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def resample_motion(raw: np.ndarray, input_fps: float, output_fps: float) -> ResampledMotion:
    raw = np.asarray(raw, dtype=np.float64)
    _validate_motion(raw)
    if not (math.isfinite(input_fps) and math.isfinite(output_fps)):
        raise ValueError("Frame rates must be finite")
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError("Frame rates must be positive")
    key_times = np.arange(len(raw), dtype=np.float64) / input_fps
    count = math.floor(key_times[-1] * output_fps + 1e-6) + 1
    if count < 2:
        raise ValueError("Output motion must contain at least 2 frames")
    times = np.minimum(np.arange(count, dtype=np.float64) / output_fps, key_times[-1])
    indices = np.minimum(np.floor(times * input_fps).astype(int), len(raw) - 2)
    blend = (times * input_fps - indices)[:, None]
    linear = raw[indices] * (1 - blend) + raw[indices + 1] * blend
    quat = Slerp(key_times, Rotation.from_quat(raw[:, 3:7]))(times).as_quat()
    quat = quat[:, [3, 0, 1, 2]]
    dt = 1 / output_fps
    return ResampledMotion(
        fps=output_fps,
        root_pos=linear[:, :3],
        root_quat=quat,
        joint_pos=linear[:, 7:],
        root_lin_vel_w=np.gradient(linear[:, :3], dt, axis=0),
        root_ang_vel_w=_so3_derivative(quat, dt),
        joint_vel=np.gradient(linear[:, 7:], dt, axis=0),
    )


def load_g1_model() -> mujoco.MjModel:
    """Compile the pinned mjlab G1 XML without creating a GPU simulation."""
    from mjlab import MJLAB_SRC_PATH

    xml = MJLAB_SRC_PATH / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if (model.nq, model.nv, model.nbody) != (36, 35, 31) or names != G1_JOINT_NAMES:
        raise ValueError("mjlab G1 model dimensions/joint order differ from the CSV convention")
    return model


def forward_kinematics(model: mujoco.MjModel, motion: ResampledMotion) -> BodyMotion:
    data = mujoco.MjData(model)
    shape = (motion.num_frames, model.nbody - 1)
    pos = np.empty((*shape, 3))
    quat = np.empty((*shape, 4))
    lin = np.empty((*shape, 3))
    ang = np.empty((*shape, 3))
    ang_body = quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w)
    for t in range(motion.num_frames):
        data.qpos[:3] = motion.root_pos[t]
        data.qpos[3:7] = motion.root_quat[t]
        data.qpos[7:] = motion.joint_pos[t]
        data.qvel[:3] = motion.root_lin_vel_w[t]
        data.qvel[3:6] = ang_body[t]
        data.qvel[6:] = motion.joint_vel[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        pos[t] = data.xpos[1:]
        quat[t] = data.xquat[1:]
        ang[t] = data.cvel[1:, :3]
        # MuJoCo cvel is centered at each tree's subtree COM. Transport to the
        # link origin, matching mj_objectVelocity(mjOBJ_XBODY, flg_local=0).
        offset = data.xpos[1:] - data.subtree_com[model.body_rootid[1:]]
        lin[t] = data.cvel[1:, 3:] + np.cross(ang[t], offset)
    return BodyMotion(pos, quat, lin, ang, tuple(model.body(i).name for i in range(1, model.nbody)))


def convert_csv(
    source: str | Path,
    output: str | Path,
    input_fps: float = 30,
    output_fps: float = 50,
    model: mujoco.MjModel | None = None,
    robot: str = "g1",
) -> int:
    """Write all motion-library keys, including names, as a compressed NPZ."""
    from supermimic.robots import get_robot

    profile = get_robot(robot)
    model = profile.model() if model is None else model
    motion = resample_motion(load_csv(source, profile.num_joints), input_fps, output_fps)
    bodies = forward_kinematics(model, motion)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "fps": np.array([output_fps], dtype=np.float32),
        "joint_pos": motion.joint_pos.astype(np.float32),
        "joint_vel": motion.joint_vel.astype(np.float32),
        "body_pos_w": bodies.pos_w.astype(np.float32),
        "body_quat_w": bodies.quat_w.astype(np.float32),
        "body_lin_vel_w": bodies.lin_vel_w.astype(np.float32),
        "body_ang_vel_w": bodies.ang_vel_w.astype(np.float32),
        "joint_names": np.array([model.joint(i).name for i in range(1, model.njnt)]),
        "robot": np.array(robot),
        "body_names": np.array(bodies.body_names),
        "source": np.array(str(source)),
        "input_fps": np.array([input_fps], dtype=np.float32),
    }
    # A failed write must not leave a partial file that a rerun would skip.
    import os
    import tempfile

    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return motion.num_frames

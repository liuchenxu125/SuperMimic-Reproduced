"""CASBOT SDK 61-column .data -> the 27-joint SuperMimic motion schema.

Root conventions follow AMP_mjlab/scripts/casbot02_data_to_npz.py: swap XY,
subtract pi/2 from pitch when that file uses the standing offset, and negate
source column 6 for yaw. The source's Euler header labels are misleading;
column 5 is treated as roll, as in the existing AMP converter.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.motion_convert import _so3_derivative, forward_kinematics, resample_motion

# Unlike the AMP 22-joint converter, preserve waist and both complete 7-DOF arms.
JOINT_COLUMNS = (*range(12, 24), 60, *range(24, 31), *range(31, 38))
JOINT_NAMES = (
    *(f"leg_{side}{i}_joint" for side in ("l", "r") for i in range(1, 7)),
    "waist_yaw_joint",
    *(f"upper_{side}_{i}_joint" for side in ("left", "right") for i in range(1, 8)),
)


def load_data(path: Path) -> tuple[np.ndarray, dict]:
    rows, header, blanks = [], None, 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            fields = [value.strip() for value in fields]
            while fields and not fields[-1]: fields.pop()
            if not fields:
                blanks += 1
                continue
            if not rows and header is None and fields[0] == "Base_Latation_X":
                if len(fields) != 61 or fields[60] != "waist_yaw_joint":
                    raise ValueError(f"{path}:{line}: unexpected SDK header")
                header = fields
                continue
            if len(fields) != 61:
                raise ValueError(f"{path}:{line}: expected 61 numeric fields, got {len(fields)}")
            try:
                rows.append([float(value) for value in fields])
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: nonnumeric data; no motion rows were silently dropped") from exc
    raw = np.asarray(rows, dtype=np.float64)
    if len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"{path}: need at least two finite motion frames")
    return raw, dict(header_detected=header is not None, blank_lines=blanks, source_frames=len(raw))


def decode_data(raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the existing converter's xyz, xyzw, q27 input convention."""
    pitch = raw[:, 3].copy()
    mean = float(pitch.mean())
    offset = float(np.pi/2) if abs(mean-np.pi/2) < abs(mean) else 0.
    pitch -= offset
    orientation = Rotation.from_euler("xyz", np.column_stack((raw[:, 4], pitch, -raw[:, 5])))
    return np.column_stack((raw[:, [1, 0, 2]], orientation.as_quat(), raw[:, JOINT_COLUMNS])), offset


def limit_statistics(q, limits, names):
    excess = np.maximum(limits[:, 0]-q, q-limits[:, 1])
    bad = excess > 1e-6
    return dict(frames=int(len(q)), frames_outside_hard_limits=int(bad.any(axis=1).sum()),
                joint_values_outside_hard_limits=int(bad.sum()), max_excess_rad=float(max(0., excess.max())),
                joints={name: dict(values=int(bad[:, i].sum()), max_excess_rad=float(excess[:, i].max()))
                        for i, name in enumerate(names) if bad[:, i].any()})


def convert_data(source: Path, model, input_fps: float, output_fps: float = 50., *,
                 clamp_joints: bool = True) -> tuple[dict, dict]:
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if names != JOINT_NAMES:
        raise ValueError("SDK .data mapping requires the exact Casbot02 27-joint model order")
    raw, report = load_data(source)
    decoded, offset = decode_data(raw)
    motion = resample_motion(decoded, input_fps, output_fps)
    limits = model.jnt_range[1:]
    before = motion.joint_pos.astype(np.float32)
    if clamp_joints:
        # Float32 rounding must not put a reference just outside the XML limit.
        low, high = limits[:, 0].astype(np.float32), limits[:, 1].astype(np.float32)
        low = np.where(low.astype(np.float64)<limits[:, 0], np.nextafter(low, np.float32(np.inf)), low)
        high = np.where(high.astype(np.float64)>limits[:, 1], np.nextafter(high, np.float32(-np.inf)), high)
        joints = np.clip(before, low, high)
    else:
        joints = before.copy()
    # Derive velocities from the positions actually saved, including clipping.
    motion.joint_pos = joints.astype(np.float64)
    motion.root_pos = motion.root_pos.astype(np.float32).astype(np.float64)
    motion.root_quat /= np.linalg.norm(motion.root_quat, axis=1, keepdims=True)
    flips = np.r_[1., np.where(np.sum(motion.root_quat[1:]*motion.root_quat[:-1], axis=1)<0., -1., 1.)]
    motion.root_quat *= np.cumprod(flips)[:, None]
    motion.joint_vel = np.gradient(motion.joint_pos, 1/output_fps, axis=0)
    motion.root_lin_vel_w = np.gradient(motion.root_pos, 1/output_fps, axis=0)
    motion.root_ang_vel_w = _so3_derivative(motion.root_quat, 1/output_fps)
    bodies = forward_kinematics(model, motion)
    arrays = dict(fps=np.array([output_fps], dtype=np.float32), joint_pos=joints,
                  joint_vel=motion.joint_vel.astype(np.float32),
                  body_pos_w=bodies.pos_w.astype(np.float32), body_quat_w=bodies.quat_w.astype(np.float32),
                  body_lin_vel_w=bodies.lin_vel_w.astype(np.float32), body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
                  joint_names=np.array(names), body_names=np.array(bodies.body_names), robot=np.array("casbot02"),
                  source=np.array(str(source.resolve())), input_fps=np.array([input_fps], dtype=np.float32))
    changes = np.abs(joints.astype(np.float64)-before.astype(np.float64))
    # Recorded root velocities are diagnostics only: first samples contain a
    # large initialization spike. Always recompute all training velocities.
    delta = np.diff(raw[:, :3], axis=0)
    moving = np.linalg.norm(delta, axis=1)>1e-5
    fps_estimate = float((delta[moving]*raw[1:, 6:9][moving]).sum()/(delta[moving]**2).sum()) if moving.any() else None
    report.update(input_fps=input_fps, output_fps=output_fps, output_frames=motion.num_frames,
                  seconds=motion.num_frames/output_fps, source_time_span_seconds=(len(raw)-1)/input_fps,
                  pitch_offset_subtracted_rad=offset, root_lateral_sign=1, root_yaw_sign=-1,
                  fps_estimate_from_recorded_linear_velocity=fps_estimate,
                  source_limit_statistics=limit_statistics(raw[:, JOINT_COLUMNS], limits, names),
                  resampled_before_limits=limit_statistics(before, limits, names),
                  output_limit_statistics=limit_statistics(joints, limits, names), clamp_joints=clamp_joints,
                  frames_clamped=int((changes>0).any(axis=1).sum()), joint_values_clamped=int((changes>0).sum()),
                  max_joint_adjustment_rad=float(changes.max()),
                  joints_clamped={name: dict(values=int((changes[:, i]>0).sum()), max_adjustment_rad=float(changes[:, i].max()))
                                  for i, name in enumerate(names) if (changes[:, i]>0).any()})
    return arrays, report

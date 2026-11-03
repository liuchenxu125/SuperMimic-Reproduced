"""Cut discontinuous motion edges without inventing intermediate robot poses."""

from __future__ import annotations

import itertools

import numpy as np

from .motion_convert import ResampledMotion, _so3_derivative, forward_kinematics

TIME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
THRESHOLDS = {"joint_step_rad": 1.0, "root_step_m": 0.2, "root_rotation_rad": 1.0}


def jump_edges(arrays, root_name="base_link"):
    """Return LEFT frame indices; q and -q represent the same orientation."""
    root = list(arrays["body_names"]).index(root_name)
    q = arrays["body_quat_w"][:, root].astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    joint = np.max(abs(np.diff(arrays["joint_pos"], axis=0)), axis=-1)
    pos = np.linalg.norm(np.diff(arrays["body_pos_w"][:, root], axis=0), axis=-1)
    angle = 2 * np.arccos(np.clip(abs((q[1:] * q[:-1]).sum(-1)), 0, 1))
    return np.flatnonzero(
        (joint > THRESHOLDS["joint_step_rad"])
        | (pos > THRESHOLDS["root_step_m"])
        | (angle > THRESHOLDS["root_rotation_rad"])
    )


def clean_spans(frames, edges, fps, min_frames):
    """Half-open source spans; startup-only artifacts use 0.1 / 0.5 s trims."""
    edges = np.asarray(edges, dtype=np.int64)
    if frames < 2 or fps <= 0 or min_frames < 3:
        raise ValueError("Need positive FPS and at least three frames per span")
    if len(edges) and (
        np.any(edges < 0) or np.any(edges >= frames - 1) or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("Jump edges must be sorted, unique and within the motion")
    if not len(edges):
        candidates, method = [(0, frames)], "unchanged"
    elif edges[-1] < round(0.1 * fps) - 1:
        candidates, method = [(round(0.1 * fps), frames)], "trim_0.1s"
    elif edges[-1] < round(0.5 * fps) - 1:
        candidates, method = [(round(0.5 * fps), frames)], "trim_0.5s"
    else:
        cuts = [0, *(int(i) + 1 for i in edges), frames]
        candidates, method = list(itertools.pairwise(cuts)), "split_at_jumps"
    return [(a, b) for a, b in candidates if b - a >= min_frames], method


def rebuild_span(arrays, model, start, end, root_name="base_link"):
    """Preserve joint/root poses and tracked body order; derive velocities locally.

    No joint clamping, root reorientation, smoothing or interpolation is performed.
    FK rebuilds body targets from the unchanged poses, for compact or full NPZs.
    """
    n = len(arrays["joint_pos"])
    if not 0 <= start < end <= n or end - start < 3:
        raise ValueError("Invalid source span")
    joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if tuple(arrays["joint_names"]) != joint_names:
        raise ValueError("Joint order differs from model")
    fps = float(arrays["fps"].item())
    root = list(arrays["body_names"]).index(root_name)
    qpos = arrays["joint_pos"][start:end].astype(np.float64)
    pos = arrays["body_pos_w"][start:end, root].astype(np.float64)
    quat = arrays["body_quat_w"][start:end, root].astype(np.float64)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    motion = ResampledMotion(
        fps,
        pos,
        quat,
        qpos,
        np.gradient(pos, 1 / fps, axis=0),
        _so3_derivative(quat, 1 / fps),
        np.gradient(qpos, 1 / fps, axis=0),
    )
    bodies = forward_kinematics(model, motion)
    ids = [bodies.body_names.index(str(name)) for name in arrays["body_names"]]
    result = {k: v.copy() for k, v in arrays.items() if k not in TIME_KEYS}
    result.update(
        joint_pos=arrays["joint_pos"][start:end].copy(),
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w[:, ids].astype(np.float32),
        body_quat_w=bodies.quat_w[:, ids].astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w[:, ids].astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w[:, ids].astype(np.float32),
    )
    for key in TIME_KEYS:
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite {key}")
    if len(jump_edges(result, root_name)):
        raise ValueError("Rebuilt span still contains a discontinuity")
    np.testing.assert_array_equal(result["joint_pos"], arrays["joint_pos"][start:end])
    np.testing.assert_allclose(result["body_pos_w"][:, root], pos, atol=1e-6, rtol=0)
    return result

"""Data repairs must keep joint derivatives and all rigid-body targets consistent."""

import importlib.util
from pathlib import Path

import numpy as np

from supermimic.data.motion_convert import ResampledMotion, forward_kinematics
from supermimic.robots import get_robot

spec = importlib.util.spec_from_file_location(
    "clean_motion_npz", Path(__file__).resolve().parents[1] / "scripts/data/clean_motion_npz.py"
)
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)


def test_trim_clip_and_rebuild_dependent_arrays():
    profile = get_robot("casbot02")
    model = profile.model()
    frames, dt = 24, 0.02
    q = np.tile(model.qpos0[7:], (frames, 1))
    elbow = [model.joint(i).name for i in range(1, model.njnt)].index("upper_left_4_joint")
    elbow_lower = model.jnt_range[model.joint('upper_left_4_joint').id, 0]
    q[:, elbow] = np.linspace(elbow_lower - .65, elbow_lower + .5, frames)
    root = np.tile([0.0, 0.0, 0.9], (frames, 1))
    root[:, 0] = np.arange(frames) * 0.002
    root[0, 0] = 10.0  # Bad startup sample must not leak into repaired derivatives.
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    motion = ResampledMotion(
        50.0,
        root,
        quat,
        q,
        np.gradient(root, dt, axis=0),
        np.zeros((frames, 3)),
        np.gradient(q, dt, axis=0),
    )
    bodies = forward_kinematics(model, motion)
    arrays = {
        "fps": np.array([50.0], np.float32),
        "joint_pos": q.astype(np.float32),
        "joint_vel": motion.joint_vel.astype(np.float32),
        "body_pos_w": bodies.pos_w.astype(np.float32),
        "body_quat_w": bodies.quat_w.astype(np.float32),
        "body_lin_vel_w": bodies.lin_vel_w.astype(np.float32),
        "body_ang_vel_w": bodies.ang_vel_w.astype(np.float32),
        "body_names": np.array(bodies.body_names),
        "joint_names": np.array([model.joint(i).name for i in range(1, model.njnt)]),
        "robot": np.array("casbot02"),
        "source": np.array("untouched.csv"),
    }
    original_q = arrays["joint_pos"].copy()
    repaired, stats = cleaner.clean_arrays(arrays, model, profile.root_body, 5)
    validation = cleaner.validate_arrays(repaired, model, profile.root_body)
    assert len(repaired["joint_pos"]) == frames - 5
    assert stats["trimmed_seconds"] == 0.1
    assert stats["after"]["frames_outside_hard_limits"] == 0
    assert stats["max_joint_adjustment_rad"] > 0.2
    assert stats["first_root_speed_before_m_s"] > 100
    assert stats["first_root_speed_after_m_s"] < 0.11
    assert validation["all_frames_joint_and_root_derivatives_checked"]
    np.testing.assert_array_equal(arrays["joint_pos"], original_q)
    np.testing.assert_array_equal(repaired["body_pos_w"][:, 0], arrays["body_pos_w"][5:, 0])
    assert repaired["source"].item() == "untouched.csv"
    # Old arm targets no longer describe the clamped skeleton.
    assert np.max(np.abs(repaired["body_pos_w"] - arrays["body_pos_w"][5:])) > 0.01
    assert np.all(repaired["joint_vel"][:3, elbow] == 0)


def test_float32_limits_do_not_round_outside_model():
    model = get_robot("casbot02").model()
    lo, hi = cleaner.float32_bounds(model.jnt_range[1:])
    assert (lo.astype(np.float64) >= model.jnt_range[1:, 0]).all()
    assert (hi.astype(np.float64) <= model.jnt_range[1:, 1]).all()

import numpy as np
import pytest

from supermimic.data.bones_seed import (
    SONIC_ACROBATICS,
    _foot_sole_height,
    feasibility_filter,
    name_filter,
)
from supermimic.data.motion_convert import forward_kinematics, load_g1_model, resample_motion


@pytest.mark.parametrize("keyword", SONIC_ACROBATICS)
def test_acrobatics_keywords_restored(keyword):
    assert name_filter(f"{keyword.upper()}_001__A407").startswith("name:")


def test_walk_and_flip_are_not_blanket_excluded():
    assert name_filter("walk_001__A407") is None
    assert name_filter("flip_ff_001__A407") is None
    assert name_filter("jump_off_50cm_001") is not None


def make_npz(path, height=0.8, foot_height=0.0, robot="g1"):
    n = 50
    pos = np.zeros((n, 3, 3))
    pos[:, 0, 2] = height
    pos[:, 1:, 2] = foot_height
    quat = np.zeros((n, 3, 4))
    quat[..., 0] = 1.0
    np.savez(
        path,
        fps=np.array([50.0]),
        robot=robot,
        body_names=np.array(["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]),
        joint_pos=np.zeros((n, 29)),
        joint_vel=np.zeros((n, 29)),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros_like(pos),
    )


def test_feasibility_elevated_penetrating_and_floating(tmp_path, monkeypatch):
    path = tmp_path / "walk.npz"
    make_npz(path, foot_height=0.21)
    assert feasibility_filter(path) == "elevated_support"
    make_npz(path, foot_height=-0.13)
    assert feasibility_filter(path) == "ground_penetration"
    make_npz(path)
    monkeypatch.setattr(
        "supermimic.data.bones_seed._foot_sole_height",
        lambda path, gate: np.full(gate.sum(), 0.021),
    )
    assert feasibility_filter(path) == "floating_feet"
    monkeypatch.setattr(
        "supermimic.data.bones_seed._foot_sole_height",
        lambda path, gate: np.full(gate.sum(), 0.019),
    )
    assert feasibility_filter(path) is None
    make_npz(path, robot="casbot02")
    with pytest.raises(ValueError, match="G1-only"):
        feasibility_filter(path)


def test_capsule_foot_height_tracks_root_translation(tmp_path):
    model = load_g1_model()
    raw = np.zeros((3, 36))
    raw[:, 2] = 0.8
    raw[:, 6] = 1.0
    motion = resample_motion(raw, 50, 50)
    body = forward_kinematics(model, motion)
    path = tmp_path / "feet.npz"

    def save(offset):
        pos = body.pos_w.copy()
        pos[..., 2] += offset
        np.savez(
            path,
            body_names=body.body_names,
            body_pos_w=pos,
            body_quat_w=body.quat_w,
            joint_pos=motion.joint_pos,
        )

    save(0.0)
    before = _foot_sole_height(path, np.ones(3, dtype=bool))
    save(0.035)
    after = _foot_sole_height(path, np.ones(3, dtype=bool))
    np.testing.assert_allclose(after - before, 0.035, atol=1e-12)

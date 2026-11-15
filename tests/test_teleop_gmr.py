"""Actual upstream GMR integration checks; optional dependencies live in .venv-pico."""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.teleop.gmr_retarget import DEFAULT_CONFIG, CasbotGMRRetargeter, resolve_config
from supermimic.teleop.pico import JOINTS, UNITY_TO_ZUP, zup_body
from supermimic.teleop.reference import PreviewBuilder, active_sim_bundle, load_contract, encode, decode

ROOT = Path(__file__).resolve().parents[1]


def skeleton_from_robot(r, qpos):
    """Manufactured 24-joint PICO observation with reachable robot task targets."""
    r.data.qpos[:] = qpos
    mujoco.mj_forward(r.model, r.data)
    root = qpos[:3] / r.gmr.human_scale_table["Pelvis"]
    positions = np.tile(root, (24, 1))
    rotations = np.tile([1., 0, 0, 0], (24, 1))
    for body, (human, _, _, offset, rotation) in r.config["ik_match_table1"].items():
        i, j = r.model.body(body).id, JOINTS.index(human)
        target_rotation = Rotation.from_matrix(r.data.xmat[i].reshape(3, 3))
        rotations[j] = (target_rotation * Rotation.from_quat(rotation, scalar_first=True).inv()).as_quat(scalar_first=True)
        scaled = r.data.xpos[i] - target_rotation.apply(offset)
        positions[j] = root + (scaled - qpos[:3]) / r.gmr.human_scale_table[human]
    for side in ("Left", "Right"):
        positions[JOINTS.index(side + "_Ankle")] = positions[JOINTS.index(side + "_Foot")]
    return np.c_[UNITY_TO_ZUP.inv().apply(positions),
                 (UNITY_TO_ZUP.inv() * Rotation.from_quat(rotations, scalar_first=True)).as_quat()]


@pytest.fixture
def retarget():
    pytest.importorskip("mink")
    if not (Path.home() / "Desktop/GMR/general_motion_retargeting/motion_retarget.py").is_file():
        pytest.skip("GMR checkout unavailable")
    bundle = active_sim_bundle(ROOT)
    return CasbotGMRRetargeter(bundle, load_contract(bundle))


def test_wrist_basis_and_toe_origin():
    config = json.loads(DEFAULT_CONFIG.read_text())
    table = config["ik_match_table1"]
    change = Rotation.from_euler("y", -90, degrees=True)
    for side, quat in (("left", [0, 0, 1, 0]), ("right", [0, 1, 0, 0])):
        entry = table[side + "_wrist_roll_link"]
        expected = Rotation.from_quat(quat, scalar_first=True) * change
        np.testing.assert_allclose(Rotation.from_quat(entry[4], scalar_first=True).as_matrix(), expected.as_matrix(), atol=1e-10)
        np.testing.assert_allclose(table[side + "_leg_ankle_roll_link"][3], [-0.05, 0, 0.02], atol=1e-10)


def test_resolve_rejects_wrong_model_and_inconsistent_offsets(retarget):
    config = json.loads(DEFAULT_CONFIG.read_text())
    bad = copy.deepcopy(config)
    bad["ik_match_table2"]["base_link"][3] = [0, 0, 1]
    with pytest.raises(ValueError, match="offsets must match"):
        resolve_config(bad, retarget.model)
    bad = copy.deepcopy(config)
    bad["robot_root_name"] = "not_a_casbot_body"
    with pytest.raises(ValueError, match="missing"):
        resolve_config(bad, retarget.model)


def test_actual_gmr_full_body_and_protocol(retarget):
    r = retarget
    assert r.gmr.__class__.__name__ == "GeneralMotionRetargeting"
    assert len(r.gmr.tasks1) == len(r.gmr.tasks2) == 14
    with pytest.raises(ValueError, match="24x7"):
        r.calibrate(np.tile([0, 0, 0, 0, 0, 0, 1.], (3, 1)))
    raw = skeleton_from_robot(r, r.neutral)
    r.calibrate(raw)
    assert abs(r.previous[2] - r.neutral[2]) < 0.015
    windows = PreviewBuilder()
    for i in range(30):
        expected = r.neutral.copy()
        expected[7 + 14] += 0.12 * np.sin(i / 30 * np.pi)
        expected[7 + 21] -= 0.12 * np.sin(i / 30 * np.pi)
        q = r.retarget(skeleton_from_robot(r, expected))
        assert (q[7:] >= r.limits[:, 0]).all() and (q[7:] <= r.limits[:, 1]).all()
        assert r.metrics["max_joint_step_rad"] <= r.max_step + 1e-10
        assert r.metrics["max_position_error_m"] < 0.025
        assert r.metrics["max_orientation_error_rad"] < 0.06
        window = windows.push(q)
    meta = load_contract(active_sim_bundle(ROOT))
    packet = decode(encode(window, meta), meta)
    assert packet.frames.shape == (10, 67)


def test_world_yaw_invariance(retarget):
    r = retarget
    raw = skeleton_from_robot(r, r.neutral)
    r.calibrate(raw)
    expected = r.retarget(raw)
    positions, rotations = zup_body(raw)
    turn = Rotation.from_euler("z", 1.2)
    transformed = np.c_[UNITY_TO_ZUP.inv().apply(turn.apply(positions) + [2., -3., 0]),
                        (UNITY_TO_ZUP.inv() * turn * rotations).as_quat()]
    r.calibrate(transformed)
    actual = r.retarget(transformed)
    np.testing.assert_allclose(actual[:3], expected[:3], atol=2e-5)
    np.testing.assert_allclose(actual[7:], expected[7:], atol=2e-5)

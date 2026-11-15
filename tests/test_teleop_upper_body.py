"""Three measured PICO poses -> standing-reference arm IK, without body tracking."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.teleop.pico import UNITY_TO_ZUP, PicoUpperBodySource, Resampler, zup_poses
from supermimic.teleop.reference import PreviewBuilder, active_sim_bundle, load_contract
from supermimic.teleop.upper_body import CasbotUpperBodyRetargeter

ROOT = Path(__file__).resolve().parents[1]


def neutral_poses():
    return np.array(
        [
            [0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-0.3, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.3, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def ik():
    bundle = active_sim_bundle(ROOT)
    return CasbotUpperBodyRetargeter(bundle, load_contract(bundle), max_joint_speed=2.0)


def test_three_point_source_does_not_require_body_and_rejects_bad_poses():
    class SDK:
        timestamp = 1_000_000_000
        poses = neutral_poses()

        def get_time_stamp_ns(self):
            return self.timestamp

        def get_headset_pose(self):
            return self.poses[0]

        def get_left_controller_pose(self):
            return self.poses[1]

        def get_right_controller_pose(self):
            return self.poses[2]

        def get_A_button(self):
            return True

        def get_B_button(self):
            return False

    source = PicoUpperBodySource.__new__(PicoUpperBodySource)
    source.sdk, source.last_timestamp, source.last_invalid_reason = SDK(), None, ""
    stamp, poses, a, b = source.read()
    assert stamp == 1_000_000_000 and a and not b
    np.testing.assert_array_equal(poses, neutral_poses())
    assert source.read() is None
    source.sdk.timestamp += 20_000_000
    source.sdk.poses[1, 3:] = 0.0
    assert source.read() is None and source.last_invalid_reason
    assert source.last_timestamp == stamp  # invalid inputs cannot refresh the stream


def test_neutral_head_turn_and_common_translation_do_not_move_reference(ik):
    poses = neutral_poses()
    ik.calibrate(poses)
    np.testing.assert_allclose(ik.retarget(poses), ik.neutral, atol=1e-8)
    poses[:, :3] += [0.3, 0.1, -0.2]
    poses[0, 3:] = Rotation.from_euler("y", 0.5).as_quat()
    np.testing.assert_allclose(ik.retarget(poses), ik.neutral, atol=1e-8)


def test_only_arms_move_with_limits_slew_and_consistent_preview(ik):
    poses = neutral_poses()
    ik.calibrate(poses)
    preview = PreviewBuilder()
    for i in range(60):
        poses[1, 1] = 0.9 + 0.08 * np.sin(i * 0.04)
        old = ik.previous.copy()
        q = ik.retarget(poses)
        np.testing.assert_array_equal(q[:20], ik.neutral[:20])  # free root + 12 legs + waist
        assert np.abs(q[7:] - old[7:]).max() <= 0.04 + 1e-8
        assert np.all(q[7:] >= ik.limits[:, 0]) and np.all(q[7:] <= ik.limits[:, 1])
        window = preview.push(q)
    assert np.max(np.abs(q[7:] - ik.q0)) > 0.01
    assert not window.frames[:, 34:47].any()  # no fabricated leg/waist velocity
    assert not window.frames[:, 61:].any()  # no root motion
    assert ik.metrics["max_position_error_m"] < 0.04


def test_calibration_world_yaw_invariance_and_workspace_saturation(ik):
    other = CasbotUpperBodyRetargeter(
        active_sim_bundle(ROOT), load_contract(active_sim_bundle(ROOT)), max_joint_speed=2.0
    )
    original = neutral_poses()
    turn = Rotation.from_euler("z", 0.8)

    def transformed(poses):
        p, r = zup_poses(poses, 3)
        x = poses.copy()
        x[:, :3] = UNITY_TO_ZUP.inv().apply(turn.apply(p) + [2.0, -1.0, 0.4])
        x[:, 3:] = (UNITY_TO_ZUP.inv() * turn * r).as_quat()
        return x

    ik.calibrate(original)
    other.calibrate(transformed(original))
    for i in range(1, 31):
        original[1, 1] += 0.02
        np.testing.assert_allclose(
            ik.retarget(original), other.retarget(transformed(original)), atol=1e-8
        )
    assert ik.metrics["workspace_limited"]
    broken = original.copy()
    broken[1, 0] += 0.5
    with pytest.raises(ValueError, match="jumped"):
        ik.retarget(broken)


def test_three_point_resampling_preserves_pose_quaternions():
    rs = Resampler(num_poses=3)
    poses = neutral_poses()
    assert len(rs.push(1_000_000_000, poses)) == 1
    poses[1, 1] += 0.04
    frames = rs.push(1_040_000_000, poses)
    assert len(frames) == 2 and frames[0].shape == (3, 7)
    assert frames[0][1, 1] == pytest.approx(0.92)
    np.testing.assert_allclose(np.linalg.norm(frames[0][:, 3:], axis=1), 1.0)

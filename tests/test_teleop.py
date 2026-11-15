"""Live reference contracts, causality and existing-policy parity (CPU only)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from supermimic.teleop.pico import UNITY_TO_ZUP, Resampler
from supermimic.teleop.reference import (
    LatestWindow,
    PreviewBuilder,
    Window,
    active_sim_bundle,
    decode,
    encode,
    load_contract,
    stationary,
    validate_frames,
)
from supermimic.teleop.runtime import HoldOnTimeout, StreamingActor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def meta():
    return {
        "joint_names": [f"j{i}" for i in range(27)],
        "checkpoint_sha256": "a" * 64,
        "joint_limits": [[-3.0, 3.0]] * 27,
        "motors": [{"q0": 0.0, "scale": 0.2} for _ in range(27)],
    }


@pytest.fixture
def frames():
    x = np.zeros((10, 67))
    x[:, 2], x[:, 3] = 0.88, 1.0
    x[:, 0] = np.arange(10) * 0.002
    x[:, 61] = 0.1
    return x


def test_binary_roundtrip_and_same_decoder_as_ros(meta, frames):
    packet = encode(Window(123, 9, frames), meta)
    actual = decode(packet, meta)
    np.testing.assert_allclose(actual.frames, frames, atol=3e-8)
    spec = importlib.util.spec_from_file_location(
        "reference_bridge", ROOT / "rl_controller/scripts/teleop_reference_bridge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    header, data = module.decode_reference(packet, meta)
    assert header["session"] == 123 and data[:3] == [123.0, 9.0, 1.0]
    np.testing.assert_array_equal(np.asarray(data[3:]).reshape(10, 67), actual.frames)
    malformed = packet[:8] + (2).to_bytes(4, "little") + b"[]" + packet[-670 * 4 :]
    for decoder in (decode, module.decode_reference):
        with pytest.raises(ValueError, match="header object"):
            decoder(malformed, meta)


def test_rejects_wrong_model_bad_quaternion_and_truncated_packet(meta, frames):
    packet = encode(Window(1, 2, frames), meta)
    for bad in [b"BADMAGIC" + packet[8:], packet[:-1], packet + b"x"]:
        with pytest.raises(ValueError):
            decode(bad, meta)
    with pytest.raises(ValueError, match="mismatch"):
        decode(packet, {**meta, "checkpoint_sha256": "b" * 64})
    frames[0, 3] = 0
    with pytest.raises(ValueError, match="quaternion"):
        validate_frames(frames, meta)


def test_real_preview_and_causal_pose_velocities():
    builder = PreviewBuilder(session=3)
    q = np.r_[np.zeros(3), [1.0, 0, 0, 0], np.zeros(27)]
    for i in range(9):
        assert builder.push(q) is None
    q[0], q[7] = 0.01, 0.02
    win = builder.push(q)
    assert win.frame == 9 and win.session == 3
    assert win.frames[0, 0] == 0 and win.frames[-1, 0] == 0.01
    assert win.frames[-1, 34] == 1.0 and win.frames[-1, 61] == 0.5
    held = builder.push(q, "HOLD")
    assert np.max(np.abs(held.frames[:, 34:])) == 0
    assert held.frames[0, 0] == win.frames[0, 0]  # no jump to the newest preview pose
    np.testing.assert_array_equal(held.frames, stationary(held.frames[0]))


def test_duplicate_frames_dont_refresh_timeout_and_new_session_requires_rearm(frames):
    clock = [0.0]
    receiver = LatestWindow(timeout=0.3, clock=lambda: clock[0])
    hold = HoldOnTimeout(receiver)
    receiver.push(Window(1, 0, frames))
    np.testing.assert_array_equal(hold.window(), frames)
    clock[0] = 0.31
    assert not receiver.push(Window(1, 0, frames))
    assert not receiver.fresh()
    held = hold.window()
    assert hold.latched and not held[:, 34:].any()
    receiver.push(Window(1, 1, frames))
    assert hold.latched
    hold.resume()
    assert not hold.latched
    with pytest.raises(ValueError, match="session"):
        receiver.push(Window(2, 2, frames))


def test_resampler_preserves_sign_continuity_and_drops_long_gaps():
    x = np.zeros((24, 7))
    x[:, 6] = 1
    rs = Resampler()
    assert len(rs.push(1_000_000_000, x)) == 1
    y = x.copy()
    y[:, 0] = 0.1
    y[:, 3:] *= -1
    result = rs.push(1_040_000_000, y)
    assert len(result) == 2
    np.testing.assert_allclose(result[0][:, 0], 0.05)
    np.testing.assert_allclose(result[0][:, 6], 1.0)
    assert rs.push(1_030_000_000, x) == []
    assert len(rs.push(2_000_000_000, x)) == 1


def test_pico_uses_body_timestamp_not_controller_updates():
    from supermimic.teleop.pico import PicoSource

    class SDK:
        timestamp = 1_000_000_000
        available = True
        a = False

        def get_time_stamp_ns(self):
            return self.timestamp

        def is_body_data_available(self):
            return self.available

        def get_body_timestamp_ns(self):
            return self.timestamp

        def get_body_joints_pose(self):
            return np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (24, 1))

        def get_A_button(self):
            return self.a

        def get_B_button(self):
            return False

    source = PicoSource.__new__(PicoSource)
    source.sdk, source.last_timestamp = SDK(), None
    assert source.read()[0] == 1_000_000_000
    assert source.read() is None
    source.sdk.timestamp -= 1
    assert source.read() is None
    source.sdk.timestamp += 20_000_001
    assert source.read()[0] == 1_020_000_000
    # A can be received while the full-body retargeter must still wait.
    source.sdk.available, source.sdk.a = False, True
    assert source.read() is None
    assert source.status()["A"] and not source.status()["body_available"]


def test_live_rot6d_keeps_reference_feedback_in_every_history_row(meta, frames):
    actor = StreamingActor(lambda o, g: torch.zeros(1, 27), meta)
    qpos = np.r_[np.zeros(3), [1.0, 0, 0, 0], np.zeros(27)]
    qvel = np.zeros(33)
    actor.align(qpos, frames[0])
    frames[:, 3:7] = [np.cos(0.1), 0, 0, np.sin(0.1)]
    _, _, obs, goal = actor.act(qpos, qvel, frames)
    expected = Rotation.from_quat([0, 0, np.sin(0.1), np.cos(0.1)]).as_matrix()[:, :2].reshape(6)
    np.testing.assert_allclose(obs[-60:].reshape(10, 6), np.tile(expected, (10, 1)), atol=3e-8)
    assert goal.shape == (690,) and obs.shape == (930,)
    assert not np.allclose(expected, [1, 0, 0, 1, 0, 0])


@pytest.fixture
def real_bundle():
    try:
        bundle = active_sim_bundle(ROOT)
    except (FileNotFoundError, ValueError):
        pytest.skip("Local exported Casbot02 actor not available")
    return bundle, load_contract(bundle)


def synthetic_pico(ik):
    from supermimic.teleop.pico import JOINTS

    body = np.zeros((24, 7))
    positions = np.tile(ik.neutral[:3], (24, 1))
    rotations = Rotation.identity(24)
    for human, robot, *_ in ik.ids:
        positions[human] = ik.robot_positions[robot]
        matrices = rotations.as_matrix()
        matrices[human] = ik.robot_rotations[robot]
        rotations = Rotation.from_matrix(matrices)
    positions[0] = ik.neutral[:3]
    body[:, :3] = UNITY_TO_ZUP.inv().apply(positions)
    body[:, 3:] = (UNITY_TO_ZUP.inv() * rotations).as_quat()
    assert body[JOINTS.index("Left_Ankle"), :3].any()
    return body


def test_neutral_calibrated_ik_roundtrip_and_wrist_movement(real_bundle):
    from supermimic.teleop.pico import JOINTS
    from supermimic.teleop.retarget import CasbotRetargeter

    bundle, meta = real_bundle
    ik = CasbotRetargeter(bundle, meta)
    body = synthetic_pico(ik)
    ik.calibrate(body)
    np.testing.assert_allclose(ik.retarget(body), ik.neutral, atol=1e-7)
    body[JOINTS.index("Left_Wrist"), 0] += 0.02
    out = ik.retarget(body)
    assert np.max(np.abs(out[7:] - ik.q0)) > 1e-5
    assert np.max(np.abs(out[7:] - ik.q0)) <= ik.max_step + 1e-9
    assert ik.metrics["max_position_error_m"] < 0.025


def test_ik_is_invariant_to_vr_world_heading(real_bundle):
    from supermimic.teleop.pico import JOINTS, zup_body
    from supermimic.teleop.retarget import CasbotRetargeter

    bundle, meta = real_bundle
    a, b = CasbotRetargeter(bundle, meta), CasbotRetargeter(bundle, meta)
    body = synthetic_pico(a)
    turn = Rotation.from_euler("z", 1.2)

    def turn_world(raw):
        positions, rotations = zup_body(raw)
        result = raw.copy()
        result[:, :3] = UNITY_TO_ZUP.inv().apply(turn.apply(positions) + [2.0, -1.0, 0.3])
        result[:, 3:] = (UNITY_TO_ZUP.inv() * turn * rotations).as_quat()
        return result

    a.calibrate(body)
    b.calibrate(turn_world(body))
    body[JOINTS.index("Left_Wrist"), 0] += 0.02
    np.testing.assert_allclose(a.retarget(body), b.retarget(turn_world(body)), atol=1e-8)


def test_streaming_inputs_and_actions_match_original_sim2sim(real_bundle):
    from supermimic.deployment.sim2sim import Sim2Sim

    bundle, meta = real_bundle
    sim = Sim2Sim(bundle, threads=1)
    sim.reset("踏步")
    live = StreamingActor(sim.actor, meta)
    ref = sim.reference
    rows = np.c_[
        ref.root_pos, ref.root_quat, ref.joint_pos, ref.joint_vel, ref.root_lin, ref.root_ang
    ]
    live.align(sim.data.qpos, rows[0])
    for _ in range(40):
        indices = np.minimum(sim.target_frame + np.arange(10), len(rows) - 1)
        packet = encode(Window(1, sim.steps, rows[indices]), meta)
        obs_expected, goal_expected = sim.inputs()
        action, target, obs, goal = live.act(
            sim.data.qpos, sim.data.qvel, decode(packet, meta).frames
        )
        np.testing.assert_allclose(obs, obs_expected, atol=5e-7)
        np.testing.assert_allclose(goal, goal_expected, atol=1e-6)
        with torch.inference_mode():
            expected = sim.actor(
                torch.from_numpy(obs_expected[None]), torch.from_numpy(goal_expected[None])
            )[0].numpy()
        np.testing.assert_allclose(action, expected, atol=3e-5)
        np.testing.assert_allclose(target, sim.q0 + sim.scale * expected, atol=8e-6)
        sim.step()
        assert not sim.failed

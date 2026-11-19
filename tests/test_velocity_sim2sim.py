"""Velocity-command playback: body-frame measurement and actor wiring."""

from pathlib import Path

import numpy as np
import pytest

from supermimic.velocity.sim2sim import VelocitySim, body_velocity, clamp_command, command_limits

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs/casbot02_seed_motion_velocity_muon_sim2sim"
ACTOR = ROOT / "outputs/casbot02_seed_motion_velocity_muon_export/velocity_actor.pt"
LIMITS = np.array([[-1.5, 1.5], [-0.5, 0.5], [-3.14, 3.14]])


def test_command_clamps_to_the_training_box():
    np.testing.assert_allclose(clamp_command([3, -1, 9], LIMITS), [1.5, -0.5, 3.14])
    np.testing.assert_allclose(clamp_command([0.4, 0.1, -0.2], LIMITS), [0.4, 0.1, -0.2])
    with pytest.raises(ValueError):
        clamp_command([0, 0], LIMITS)


def test_body_velocity_uses_the_current_root_axes():
    angle = np.pi / 2
    qpos = np.zeros(34)
    qpos[3:7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    qvel = np.zeros(33)
    qvel[0] = 1
    qvel[5] = 0.2
    np.testing.assert_allclose(body_velocity(qpos, qvel), [0, -1, 0.2], atol=1e-6)


@pytest.mark.skipif(not ACTOR.with_name("manifest.json").exists(), reason="local velocity export")
def test_export_manifest_supplies_command_limits():
    np.testing.assert_allclose(command_limits(ACTOR), LIMITS)


@pytest.mark.skipif(not BUNDLE.exists() or not ACTOR.exists(), reason="local velocity export")
def test_velocity_actor_steps_from_the_standing_keyframe():
    sim = VelocitySim(BUNDLE, ACTOR)
    sim.reset([0.6, 0.0, 0.0])
    assert sim.history.packed().shape == (sim.proprio_dim,)
    assert sim.proprio_dim == 870
    row = sim.step()
    assert row.shape == (9,) and np.isfinite(row).all()
    assert sim.report()["steps"] == 1

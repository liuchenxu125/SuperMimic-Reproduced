"""Physics-step command latency, reset isolation, and checkpoint compatibility."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.actuator import BuiltinActuatorGroup

from supermimic.deployment.sim2sim import PositionCommandDelay
from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.robots import get_robot


def delay_bus(batch_size=8, mode="step"):
    """Run the actual mjlab builtin control path on CPU, without a physics engine."""
    configs = get_robot("casbot02").entity_cfg(
        actuator_delay=True, actuator_delay_mode=mode,
    ).articulation.actuators
    actuators = []
    for i, cfg in enumerate(configs):
        act = cfg.build(None, [i], [cfg.target_names_expr[0]])
        act._target_ids = torch.tensor([i])
        act._ctrl_ids = torch.tensor([i])
        actuators.append(act)
    bus, custom = BuiltinActuatorGroup.process(actuators)
    assert not custom and len(bus._delayed_groups) == 1
    bus.initialize(batch_size, "cpu")
    ctrl = torch.empty(batch_size, 27)
    data = SimpleNamespace(
        joint_pos_target=torch.zeros(batch_size, 27),
        write_ctrl=lambda target, ids: ctrl.__setitem__((slice(None), ids), target),
    )
    return bus, actuators, data, ctrl


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_delays_targets_in_physics_steps_and_clears_old_episode(lag):
    bus, actuators, data, ctrl = delay_bus(mode="episode")
    native = PositionCommandDelay(2)
    for episode in range(2):
        for act in actuators:
            act.reset()
        actuators[0].set_lags(torch.full((8,), lag))
        native.reset()
        native.lag = lag
        for t in range(8):
            # A distinct target at every physics step exposes a misplaced
            # 20 ms policy-step buffer and unintended per-step resampling.
            data.joint_pos_target[:] = 100 * episode + t + 1
            bus.apply_controls(data)
            expected = 100 * episode + max(0, t - lag) + 1
            torch.testing.assert_close(ctrl, torch.full_like(ctrl, expected))
            np.testing.assert_array_equal(
                native.apply(data.joint_pos_target[0].numpy()), ctrl[0].numpy()
            )


def test_sample_once_per_episode_and_partial_reset_isolation():
    bus, actuators, data, ctrl = delay_bus(256, mode="episode")
    data.joint_pos_target[:] = 7
    bus.apply_controls(data)
    buffer = actuators[0]._delay_buffer
    lags = buffer.current_lags.clone()
    assert set(lags.tolist()) == {0, 1, 2}
    for t in range(6):
        data.joint_pos_target[:] = 8 + t
        bus.apply_controls(data)
        torch.testing.assert_close(buffer.current_lags, lags)
    for _ in range(3):
        bus.apply_controls(data)  # All non-reset rows now have constant history.
    before = ctrl.clone()
    ids = torch.arange(0, 256, 2)
    for act in actuators:
        act.reset(ids)
    data.joint_pos_target[ids] = 99
    bus.apply_controls(data)
    torch.testing.assert_close(ctrl[ids], torch.full_like(ctrl[ids], 99))
    torch.testing.assert_close(ctrl[1::2], before[1::2])
    torch.testing.assert_close(buffer.current_lags[1::2], lags[1::2])


def test_new_training_default_and_legacy_environment_default():
    from supermimic.config import TrainConfig

    assert TrainConfig(robot="casbot02").env.actuator_delay
    assert TrainConfig(robot="casbot02").env.actuator_delay_mode == "step"
    args = SuperMimicEnvCfgArgs(npz_files=["unused.npz"], robot="casbot02")
    legacy = make_env_cfg(args)
    assert all(a.delay_max_lag == 0 for a in legacy.scene.entities["robot"].articulation.actuators)
    args.actuator_delay = True
    args.actuator_delay_mode = "step"
    delayed = make_env_cfg(args)
    assert delayed.sim.mujoco.timestep == 0.005 and delayed.decimation == 4
    assert all(a.delay_max_lag == 2 and a.delay_hold_prob == 0.0 and a.delay_update_period == 0
               for a in delayed.scene.entities["robot"].articulation.actuators)
    args.robot = "g1"
    g1_on = make_env_cfg(args).scene.entities["robot"].articulation.actuators
    args.actuator_delay = False
    g1_off = make_env_cfg(args).scene.entities["robot"].articulation.actuators
    assert g1_on == g1_off


def test_old_checkpoint_requires_matching_no_delay_for_resume(tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(robot="casbot02", run_name="delay", log_root=tmp_path)
    saved = asdict(cfg)
    saved["env"].pop("actuator_delay")
    saved["policy"]["goal_dim"] = 640
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"iteration_unit": "update", "iteration": 10, "env_steps": 65,
                "num_updates": 10, "collection_accumulator": 0., "config": saved}, checkpoint)
    cfg.resume = checkpoint
    with pytest.raises(ValueError, match="Resume actuator delay differs"):
        Trainer(cfg)
    assert not (tmp_path / "delay").exists()


def test_step_resampling_and_partial_reset_preserve_target_history():
    torch.manual_seed(113)
    bus, actuators, data, ctrl = delay_bus(256)
    buffer = actuators[0]._delay_buffer
    native = PositionCommandDelay(2, mode="step")
    previous_lags = None
    for t in range(12):
        data.joint_pos_target[:] = t + 1
        bus.apply_controls(data)
        lags = buffer.current_lags.clone()
        assert set(lags.tolist()) == {0, 1, 2}
        if previous_lags is not None:
            assert (lags != previous_lags).sum() > 100
        expected = (t - lags).clamp_min(0) + 1
        torch.testing.assert_close(ctrl[:, 0], expected.float())
        # Replay the actual mjlab lag for one environment through native MuJoCo's bus.
        native.rng = SimpleNamespace(integers=lambda high, lag=int(lags[0]): lag)
        np.testing.assert_array_equal(native.apply(data.joint_pos_target[0].numpy()), ctrl[0])
        previous_lags = lags
    ids = torch.arange(0, 256, 2)
    for act in actuators:
        act.reset(ids)
    data.joint_pos_target[ids] = 99
    data.joint_pos_target[1::2] = 13
    bus.apply_controls(data)
    torch.testing.assert_close(ctrl[ids], torch.full_like(ctrl[ids], 99))
    expected = 13 - buffer.current_lags[1::2]
    torch.testing.assert_close(ctrl[1::2, 0], expected.float())


@pytest.mark.parametrize("missing_field,error", [
    ("actuator_delay_mode", "Resume actuator delay mode differs"),
    ("training_floor_friction", "Resume training floor friction differs"),
])
def test_resume_rejects_silent_physics_change(tmp_path, missing_field, error):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(robot="casbot02", run_name="delay", log_root=tmp_path)
    saved = asdict(cfg)
    saved["env"].pop(missing_field)
    saved["policy"]["goal_dim"] = 640
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"iteration_unit": "update", "iteration": 10, "env_steps": 65,
                "num_updates": 10, "collection_accumulator": 0., "config": saved}, checkpoint)
    cfg.resume = checkpoint
    with pytest.raises(ValueError, match=error):
        Trainer(cfg)
    assert not (tmp_path / "delay").exists()


@pytest.mark.parametrize("training", [True, False])
def test_compiled_floor_and_actual_foot_contact_friction(training):
    import mujoco
    from mjlab.scene import Scene

    args = SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="casbot02", num_envs=1,
        domain_randomization=training,
    )
    cfg = make_env_cfg(args)
    model = Scene(cfg.scene, "cpu").compile()
    floor = model.geom("terrain").id
    feet = [model.geom(f"robot/{side}_foot_collision").id for side in ("left", "right")]
    np.testing.assert_allclose(model.geom_friction[floor], [0.3 if training else 1., .005, .0001])
    if training:
        assert cfg.events["foot_friction"].mode == cfg.events["encoder_bias"].mode == "startup"
    # Check contact coefficients after MuJoCo mixes the floor and foot geoms,
    # not just the config value. Low foot friction must no longer be masked by 1.0.
    for mu in (.3, .6, 1.2):
        model.geom_friction[feet, 0] = mu
        data = mujoco.MjData(model)
        data.qpos[:] = model.key_qpos[0]
        contacts = []
        for z in np.linspace(.9, .1, 41):
            data.qpos[2] = z
            mujoco.mj_forward(model, data)
            contacts = [c for c in data.contact if floor in c.geom and any(f in c.geom for f in feet)]
            if contacts:
                break
        assert contacts
        expected = max(.3 if training else 1., mu)
        for contact in contacts:
            np.testing.assert_allclose(contact.friction[:2], expected)

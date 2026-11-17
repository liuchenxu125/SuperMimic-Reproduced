"""Independent velocity semantics, no-reference observation leak, joint BPTT."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.velocity.commands import VelocityCommands, VelocityTaskCfg, centered_mean
from supermimic.velocity.policy import (
    VelocityActor,
    VelocityEncoder,
    latent_bc,
    mixed_actions,
    velocity_history,
)
from supermimic.velocity.training import VelocityPolicyTrainer


def policy_config():
    return PolicyCfg(
        num_joints=27,
        goal_dim=690,
        obs_term_scale=(1.0, 5.0, 2.0, 1.0, 1.0, 1.0),
        encoder_hidden=(16, 12),
        decoder_hidden=(16, 12),
    )


def fake_library():
    return SimpleNamespace(
        device=torch.device("cpu"),
        library_transfer="sync",
        root_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]] * 3 + [[-1.0, 0.0, 0.0]] * 3),
        root_ang_vel_w=torch.tensor([[0.0, 0.0, 0.2]] * 6),
        clips=[SimpleNamespace(start=0, length=3), SimpleNamespace(start=3, length=3)],
    )


def test_centered_commands_are_clip_local_and_use_actual_body_axes():
    source = fake_library()
    cfg = VelocityTaskCfg(
        window_frames=25,
        command_noise=(0.0, 0.0, 0.0),
        lin_vel_x_range=(-2, 2),
        lin_vel_y_range=(-2, 2),
    )
    commands = VelocityCommands(source, cfg)
    # Huge window never mixes unrelated neighboring clips.
    torch.testing.assert_close(commands.track[:, 0], source.root_lin_vel_w[:, 0])
    angle = torch.tensor(torch.pi / 2)
    quat = torch.tensor([[torch.cos(angle / 2), 0.0, 0.0, torch.sin(angle / 2)]])
    c = commands(torch.tensor([0]), quat, noisy=False)
    torch.testing.assert_close(c, torch.tensor([[0.0, -1.0, 0.2]]), atol=1e-6, rtol=0)
    raw = torch.arange(5.0).unsqueeze(-1)
    torch.testing.assert_close(
        centered_mean(raw, 3).flatten(), torch.tensor([1 / 3, 1.0, 2.0, 3.0, 11 / 3])
    )


def test_noise_is_fresh_bounded_and_clamped_and_rotation_is_differentiable():
    commands = VelocityCommands(fake_library(), VelocityTaskCfg())
    frames = torch.zeros(128, dtype=torch.long)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(128, -1).clone().requires_grad_(True)
    a = commands(frames, q)
    b = commands(frames, q)
    assert not torch.equal(a, b)
    assert (a >= commands.low).all() and (a <= commands.high).all()
    c = commands(frames, q, noisy=False)
    c.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_velocity_cannot_read_motion_goal_or_reference_pose_and_motion_is_unchanged():
    torch.manual_seed(2)
    cfg = policy_config()
    policy = Policy(cfg)
    h = torch.randn(4, 930)
    goal = torch.randn(4, 690)
    command = torch.randn(4, 3)
    before = policy(h, goal).detach()
    policy.encoders["velocity"] = VelocityEncoder(cfg)
    torch.testing.assert_close(policy(h, goal), before, rtol=0, atol=0)
    changed = h.clone()
    changed[:, 870:] = 100 * torch.randn(4, 60)
    torch.testing.assert_close(
        velocity_history(h, cfg), velocity_history(changed, cfg), rtol=0, atol=0
    )
    a = mixed_actions(policy, h, goal, command, 2)
    new_goal = goal.clone()
    new_goal[2:] *= 100
    b = mixed_actions(policy, changed, new_goal, command, 2)
    torch.testing.assert_close(a[2:], b[2:], rtol=0, atol=0)
    actor = VelocityActor(policy)
    torch.testing.assert_close(
        actor(h[:, :870], command),
        policy.decoder(policy.encoders["velocity"](command), velocity_history(h, cfg)),
    )


def test_bc_gradient_only_reaches_velocity_encoder():
    cfg = policy_config()
    policy = Policy(cfg)
    policy.encoders["velocity"] = VelocityEncoder(cfg)
    c = torch.randn(3, 3, requires_grad=True)
    g = torch.randn(3, 690, requires_grad=True)
    latent_bc(policy, c, g).mean().backward()
    assert c.grad is None and g.grad is None
    assert all(p.grad is None for p in policy.encoders["motion"].parameters())
    assert all(p.grad is None for p in policy.decoder.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.encoders["velocity"].parameters()
    )


def test_velocity_export_trace_handles_multiple_batch_sizes():
    cfg = policy_config()
    policy = Policy(cfg)
    policy.encoders["velocity"] = VelocityEncoder(cfg)
    actor = VelocityActor(policy).eval()
    script = torch.jit.trace(actor, (torch.zeros(2, 870), torch.zeros(2, 3)), check_trace=False)
    for n in (1, 5):
        h, c = torch.randn(n, 870), torch.randn(n, 3)
        torch.testing.assert_close(script(h, c), actor(h, c))


@pytest.fixture
def native_library(tmp_path):
    from supermimic.data.motion_convert import convert_csv
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.robots import get_robot

    profile = get_robot("casbot02")
    model = profile.model()
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    q0 = np.array([profile.motor_parameters()["default_joint_pos"][n] for n in names])
    raw = np.zeros((70, 34))
    raw[:, 2] = 0.9
    raw[:, 6] = 1.0
    raw[:, 7:] = q0
    raw[:, 0] = np.arange(70) * 0.002
    csv, npz = tmp_path / "walk.csv", tmp_path / "walk.npz"
    np.savetxt(csv, raw, delimiter=",")
    convert_csv(csv, npz, 50, 50, model=model, robot="casbot02")
    lib = MotionLibrary(
        [npz],
        profile.tracked_bodies,
        q0,
        device="cpu",
        compact=True,
        root_body_name="base_link",
        robot_name="casbot02",
        joint_names=names,
        library_device="cpu",
    )
    return lib, model, profile, q0


def test_joint_bptt_updates_both_encoders_shared_decoder_critic_not_world_model(native_library):
    from supermimic.data.replay_buffer import ReplayBuffer
    from supermimic.imagination.policy_trainer import PolicyTrainerCfg
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    lib, model, profile, q0 = native_library
    heads = AnalyticHeads(
        G1Kinematics(model, "cpu"),
        profile.tracked_bodies,
        profile.anchor_body,
        profile.root_body,
        torch.from_numpy(model.jnt_range[1:].copy()).float(),
        torch.tensor(q0).float(),
    )
    wm = WorldModel(
        WorldModelCfg(hidden=(16, 16), state_spec=StateSpecCfg(num_joints=27)), heads, "cpu"
    )
    cfg = policy_config()
    policy = Policy(cfg)
    policy.encoders["velocity"] = VelocityEncoder(cfg)
    critic = Critic(CriticCfg(hidden=(16,)), pcfg=cfg)
    buffer = ReplayBuffer(2, 64, device="cpu", num_joints=27)
    for t in range(24):
        q, v = lib.ref_qpos_qvel(torch.tensor([t, t]))
        state = torch.cat([q, v], -1)
        buffer.add_step(
            state,
            torch.zeros(2, 27),
            torch.zeros(2, 27),
            wm.spec.xi_offset.expand(2, -1),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.long),
            torch.full((2,), t),
            torch.full((2,), t == 0),
        )
    trainer = VelocityPolicyTrainer(
        wm,
        policy,
        critic,
        lib,
        PolicyTrainerCfg(
            batch_size=4, horizon=3, compile=None, anchor_ori_obs=True, obs_noise=False
        ),
        VelocityCommands(lib, VelocityTaskCfg()),
        VelocityTaskCfg(),
    )
    modules = {
        "motion": policy.encoders["motion"],
        "velocity": policy.encoders["velocity"],
        "decoder": policy.decoder,
        "critic": critic,
        "wm": wm,
    }
    old = {k: [p.detach().clone() for p in m.parameters()] for k, m in modules.items()}
    metrics = trainer.update(buffer)
    assert all(np.isfinite(v) for v in metrics.values())
    for name, module in modules.items():
        changed = any(not torch.equal(a, b) for a, b in zip(old[name], module.parameters()))
        assert changed == (name != "wm"), name


def test_native_defaults_are_not_changed_and_robot_contract_is_stable():
    from supermimic.config import TrainConfig
    from supermimic.velocity.runner import restore_config, robot_contract

    default = TrainConfig()
    before = asdict(default)
    clone = restore_config(before)
    clone.env.anchor_ori_obs = True
    assert asdict(default) == before
    assert robot_contract("casbot02") == robot_contract("casbot02")


def test_manifest_identity_survives_json_checkpoint_config(tmp_path):
    import json

    from supermimic.config import TrainConfig
    from supermimic.velocity.runner import dataset_contract, restore_config

    np.savez(tmp_path / "clip.npz", joint_pos=np.zeros((2, 27)))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("name,keep\nclip,1\n")
    cfg = TrainConfig(robot="casbot02")
    cfg.data.npz_dir, cfg.data.manifest = tmp_path, manifest
    before = dataset_contract(cfg)
    restored = restore_config(json.loads(json.dumps(asdict(cfg), default=str)))
    assert dataset_contract(restored) == before


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_velocity_updates_preserve_torch_cache(monkeypatch, device):
    from supermimic.training.loop import Trainer
    from supermimic.velocity.runner import VelocityTrainer

    trainer = object.__new__(VelocityTrainer)
    trainer.device = device
    trainer._velocity_ready = True
    trainer.iteration = 1
    trainer.cfg = SimpleNamespace(schedule=SimpleNamespace(log_interval=50))
    trainer.policy_trainer = SimpleNamespace(latest={})
    events = []
    monkeypatch.setattr(Trainer, "_synchronize", lambda self: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("release"))

    # Synchronization and completed updates must leave the allocator cache alone.
    trainer._synchronize()
    assert events == ["sync"]
    events.clear()
    trainer.policy_trainer.latest = {"imag_objective": 0.1}
    trainer._synchronize()
    assert events == ["sync"]
    assert trainer.policy_trainer.latest == {}

    # Subsequent physics steps must not flush the cache again.
    events.clear()
    trainer._synchronize()
    assert events == ["sync"]

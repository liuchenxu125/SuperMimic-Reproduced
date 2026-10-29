"""Actual collector -> action term -> observations/reward/replay noise contract."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp import action_rate_l2, last_action

from supermimic.config import TrainConfig
from supermimic.data.recorder import ReplayRecorder
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.actions import NoisyJointPositionAction
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.training.loop import Trainer


@pytest.mark.parametrize("mode,internal_std", [("collection", 0.0), ("execution", 0.1)])
def test_trainer_wires_internal_noise_once(monkeypatch, tmp_path, mode, internal_std):
    import supermimic.training.loop as module

    cfg = TrainConfig(robot="casbot02", log_root=tmp_path, run_name=mode, device="cpu")
    cfg.env.action_noise_mode = mode
    monkeypatch.setattr(module, "init_distributed", lambda: (0, 0, 1))
    monkeypatch.setattr(module, "SummaryWriter", lambda path: None)
    monkeypatch.setattr(module, "discover_npz", lambda *args, **kw: [tmp_path / "unused.npz"])

    class Captured(Exception):
        pass

    def capture(args, **kwargs):
        assert args.action_noise_std == internal_std
        assert make_env_cfg(args).actions["joint_pos"].noise_std == internal_std
        raise Captured

    monkeypatch.setattr(module, "make_env", capture)
    with pytest.raises(Captured):
        Trainer(cfg)


@pytest.mark.parametrize("std", [0.1, 0.0])
def test_collector_noisy_action_reaches_history_penalty_and_replay(monkeypatch, std):
    n, joints = 2, 27
    cfg = TrainConfig(robot="casbot02")
    cfg.env.action_noise_std = std
    action = NoisyJointPositionAction.__new__(NoisyJointPositionAction)
    action.cfg = SimpleNamespace(clip=None)
    action.noise_std = 0.0  # Same setting verified from the real Trainer wiring above.
    action._raw_actions = torch.zeros(n, joints)
    action._executed_raw = torch.zeros(n, joints)
    action._scale, action._offset = 2.0, 0.3
    manager = SimpleNamespace(action=torch.zeros(n, joints), prev_action=torch.zeros(n, joints))
    env = SimpleNamespace(action_manager=manager, reward_buf=torch.zeros(n),
                          reset_terminated=torch.zeros(n, dtype=torch.bool))
    rec = ReplayRecorder.__new__(ReplayRecorder)
    rec._env, rec.action = env, action
    rec.buffer = ReplayBuffer(n, 8, "cpu", num_joints=joints)
    rec._state, rec._xi = torch.zeros(n, 67), torch.zeros(n, 31)
    rec._frame, rec._first = torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.bool)
    rec.command = SimpleNamespace(library=SimpleNamespace(frame_clip=torch.zeros(8, dtype=torch.long)))
    penalties, received = [], []

    def step(value):
        received.append(value.clone())
        manager.prev_action = manager.action.clone()
        manager.action = value.clone()
        action.process_actions(value)
        penalties.append(action_rate_l2(env))
        env.reward_buf = -0.1 * 0.02 * penalties[-1]
        rec._record(torch.arange(n))
        return {"policy": last_action(env).clone()}, env.reward_buf, env.reset_terminated, env.reset_terminated, {}

    env.step = step
    trainer = Trainer.__new__(Trainer)
    trainer.cfg, trainer.env = cfg, env
    clean = torch.zeros(n, joints)
    trainer.act = lambda obs: clean
    trainer._obs = {}
    trainer.env_steps = 0
    trainer.ep_reward_sum, trainer.ep_len = torch.zeros(n), torch.zeros(n)
    draws = []

    def noise(value):
        draws.append(1)
        return torch.full_like(value, 1.0 if len(draws) == 1 else -1.0)

    monkeypatch.setattr(torch, "randn_like", noise)
    trainer._collect_step()
    trainer._collect_step()
    assert len(draws) == (2 if std else 0)  # One draw per control step, no second action-term draw.
    expected = [torch.full_like(clean, std), torch.full_like(clean, -std)]
    for i, value in enumerate(expected):
        torch.testing.assert_close(received[i], value)
        torch.testing.assert_close(rec.buffer.a_clean[:, i], value)
        torch.testing.assert_close(rec.buffer.a_exec[:, i], value)
    torch.testing.assert_close(penalties[0], torch.full((n,), joints * std**2))
    torch.testing.assert_close(penalties[1], torch.full((n,), joints * (2 * std)**2))
    torch.testing.assert_close(trainer._obs["policy"], expected[1])
    torch.testing.assert_close(action._processed_actions, expected[1] * 2.0 + 0.3)
    torch.testing.assert_close(rec.buffer.prev_actions[:, 1], expected[0])
    torch.testing.assert_close(clean, torch.zeros_like(clean))


def test_legacy_resume_cannot_silently_change_noise_semantics(tmp_path):
    cfg = TrainConfig(robot="casbot02", run_name="new", log_root=tmp_path)
    saved = asdict(cfg)
    saved["env"].pop("action_noise_mode")
    saved["policy"]["goal_dim"] = 640
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"iteration_unit": "update", "iteration": 10, "env_steps": 65,
                "num_updates": 10, "collection_accumulator": 0., "config": saved}, checkpoint)
    cfg.resume = checkpoint
    with pytest.raises(ValueError, match="Resume action noise mode differs"):
        Trainer(cfg)
    assert not (tmp_path / "new").exists()

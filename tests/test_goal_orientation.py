"""Independent relative orientation: schema, environment parity and yaw gradients."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.goal_spec import anchor_goal_dim
from supermimic.kinematics.rotations import robot_relative_rot6d


@pytest.mark.parametrize(
    "anchor,ori,extra,history_ori",
    [(False, False, 0, False), (False, True, 0, True), (True, False, 9, False), (True, True, 9, False)],
)
def test_goal_schema_adds_orientation_once(anchor, ori, extra, history_ori):
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="casbot02", anchor_obs=anchor, anchor_ori_obs=ori,
        obs_noise=False,
    ))
    terms = cfg.observations["goal"].terms
    assert list(terms) == ["future_frames"] + (["anchor_pos", "anchor_ori"] if anchor else [])
    assert anchor_goal_dim(anchor, ori) == extra
    policy = ["joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "actions"]
    if history_ori:
        policy.append("reference_pelvis_ori")
    assert list(cfg.observations["policy"].terms) == policy


def test_orientation_matches_environment_and_provides_yaw_gradient_without_position():
    angle = torch.tensor([0.6, -0.4], requires_grad=True)
    zeros = torch.zeros_like(angle)
    robot_quat = torch.stack([torch.cos(angle / 2), zeros, zeros, torch.sin(angle / 2)], -1)
    ref_quat = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
    cmd = SimpleNamespace(robot_root_quat_w=robot_quat, root_quat_w=ref_quat)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd), num_envs=2)
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="casbot02", anchor_ori_obs=True, obs_noise=False,
    ))
    term = cfg.observations["policy"].terms["reference_pelvis_ori"]
    actual = term.func(env, **term.params)
    expected = robot_relative_rot6d(robot_quat, ref_quat)
    assert actual.shape == (2, 6) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    identity = torch.tensor([1., 0., 0., 1., 0., 0.])
    (actual - identity).square().sum().backward()
    assert torch.isfinite(angle.grad).all() and (angle.grad.abs() > 0.1).all()


def test_old_checkpoint_cannot_resume_with_new_orientation_inputs(tmp_path):
    from dataclasses import asdict

    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    old = TrainConfig(robot="casbot02")
    saved = asdict(old)
    saved["env"].pop("anchor_ori_obs")  # A real legacy schema lacks this field.
    saved["policy"]["goal_dim"] = 690
    checkpoint = tmp_path / "old.pt"
    torch.save({
        "iteration_unit": "update", "iteration": 10, "env_steps": 65,
        "num_updates": 10, "collection_accumulator": 0., "config": saved,
    }, checkpoint)
    cfg = TrainConfig(robot="casbot02", run_name="orientation", log_root=tmp_path,
                      resume=checkpoint)
    cfg.env.anchor_ori_obs = True
    with pytest.raises(ValueError, match="observation terms differ"):
        Trainer(cfg)
    assert not (tmp_path / "orientation").exists()
    assert cfg.policy_trainer.anchor_ori_obs and cfg.policy.goal_dim == 690
    assert len(cfg.policy.obs_term_scale) == 6

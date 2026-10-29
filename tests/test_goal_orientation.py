"""Independent relative orientation: schema, environment parity and yaw gradients."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.goal_spec import anchor_goal_dim
from supermimic.world_model.heads import anchor_relative_orientation


@pytest.mark.parametrize("anchor,ori,extra", [(False, False, 0), (False, True, 6),
                                            (True, False, 9), (True, True, 9)])
def test_goal_schema_adds_orientation_once(anchor, ori, extra):
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="casbot02", anchor_obs=anchor, anchor_ori_obs=ori,
        obs_noise=False,
    ))
    terms = cfg.observations["goal"].terms
    assert list(terms) == ["future_frames"] + (["anchor_pos"] if anchor else []) + (
        ["anchor_ori"] if anchor or ori else []
    )
    assert anchor_goal_dim(anchor, ori) == extra
    assert list(cfg.observations["policy"].terms) == [
        "joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "actions"
    ]


def test_orientation_matches_environment_and_provides_yaw_gradient_without_position():
    angle = torch.tensor([0.6, -0.4], requires_grad=True)
    zeros = torch.zeros_like(angle)
    robot_quat = torch.stack([torch.cos(angle / 2), zeros, zeros, torch.sin(angle / 2)], -1)
    ref_quat = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
    cmd = SimpleNamespace(
        robot_anchor_pos_w=torch.full((2, 3), float("nan")),
        anchor_pos_w=torch.full((2, 3), float("nan")),
        robot_anchor_quat_w=robot_quat, anchor_quat_w=ref_quat,
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd), num_envs=2)
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=["unused.npz"], robot="casbot02", anchor_ori_obs=True, obs_noise=False,
    ))
    term = cfg.observations["goal"].terms["anchor_ori"]
    actual = term.func(env, **term.params)
    expected = anchor_relative_orientation(robot_quat, ref_quat)
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
    saved["policy"]["goal_dim"] = 640
    checkpoint = tmp_path / "old.pt"
    torch.save({
        "iteration_unit": "update", "iteration": 10, "env_steps": 65,
        "num_updates": 10, "collection_accumulator": 0., "config": saved,
    }, checkpoint)
    cfg = TrainConfig(robot="casbot02", run_name="orientation", log_root=tmp_path,
                      resume=checkpoint)
    cfg.env.anchor_ori_obs = True
    with pytest.raises(ValueError, match="saved=640, requested=646"):
        Trainer(cfg)
    assert not (tmp_path / "orientation").exists()
    assert cfg.policy_trainer.anchor_ori_obs and cfg.policy.goal_dim == 646

"""CPU reference storage must preserve CUDA queries, goals, FK and gradients."""

import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import convert_csv
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.robots import get_robot


@pytest.fixture
def clip(tmp_path):
    robot = get_robot("casbot02")
    model = robot.model(collision_model="primitive")
    joints = tuple(model.joint(j).name for j in range(1, model.njnt))
    q0 = np.array([robot.motor_parameters()["default_joint_pos"][j] for j in joints])
    raw = np.zeros((50, 34))
    raw[:, 0], raw[:, 2], raw[:, 6] = np.arange(50) * .002, .88, 1.
    raw[:, 7:] = q0
    raw[:, 7] += .01 * np.sin(np.arange(50) * .1)
    source, output = tmp_path / "clip.csv", tmp_path / "clip.npz"
    np.savetxt(source, raw, delimiter=",")
    convert_csv(source, output, 50, 50, model=model, robot="casbot02")
    return output, q0, joints


def library(clip, device="cpu", storage="cpu", dtype=torch.float16, transfer=None):
    path, q0, joints = clip
    options = {} if transfer is None else {"library_transfer": transfer}
    return MotionLibrary([path], get_robot("casbot02").tracked_bodies, q0, device=device,
                         compact=True, store_dtype=dtype, root_body_name="base_link",
                         robot_name="casbot02", joint_names=joints, library_device=storage,
                         **options)


def test_cpu_storage_and_clip_end_queries(clip):
    cpu = library(clip)
    assert cpu.library_transfer == "pinned"
    assert cpu._ref.device.type == "cpu"
    assert cpu._ref.dtype == torch.float16
    frames = torch.tensor([0, 20, 49])
    assert cpu.ref71(frames).shape == (3, 67)
    torch.testing.assert_close(cpu.ref71(frames), cpu._ref[frames].float())
    goals = cpu.goal_window(frames, 10)
    assert goals.shape == (3, 690) and torch.isfinite(goals).all()
    assert "on cpu" in cpu.summary()
    with pytest.raises(ValueError, match="requires compact_library"):
        MotionLibrary([clip[0]], (), clip[1], device="cpu", library_device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("transfer", ["sync", "pinned"])
def test_host_storage_matches_gpu_storage_for_cuda_indices(clip, dtype, transfer):
    from supermimic.world_model.wm_trainer import reference_at

    gpu = library(clip, "cuda:0", "device", dtype)
    host = library(clip, "cuda:0", "cpu", dtype, transfer)
    frames = torch.tensor([0, 20, 49, 48], device="cuda:0")
    assert host._ref.device.type == "cpu" and gpu._ref.device.type == "cuda"
    assert host.ref71(frames).device.type == "cuda"
    for index in [frames, frames.reshape(2, 2)]:
        torch.testing.assert_close(host.ref71(index), gpu.ref71(index), rtol=0, atol=0)
    torch.testing.assert_close(host.goal_window(frames, 10), gpu.goal_window(frames, 10))
    a, b = reference_at(host, frames), reference_at(gpu, frames)
    for key in a.__dataclass_fields__:
        torch.testing.assert_close(getattr(a, key), getattr(b, key))
    # Gradient through a policy's encoding must survive the CPU-backed lookup.
    from supermimic.policy.nets import Policy, PolicyCfg

    policy = Policy(PolicyCfg(num_joints=27, goal_dim=690, encoder_hidden=(32,),
                              decoder_hidden=(32,))).cuda()
    goal = host.goal_window(frames, 10)
    policy(torch.zeros(4, 870, device="cuda"), goal).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())

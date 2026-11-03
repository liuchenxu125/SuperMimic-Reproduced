"""Pelvis displacement and orientation use the current reference heading."""

import numpy as np
import torch

from supermimic.deployment.sim2sim import reference_heading_goal
from supermimic.envs.mdp.motion_library import RefState, goal_features
from supermimic.kinematics.rotations import quat_from_yaw


def _window(anchor_pos, anchor_quat, future_pos, future_quat, future_lin, future_ang, joints, joint_vel, q0):
    anchor = RefState(
        anchor_pos, anchor_quat, joints[:, 0], future_lin[:, 0], future_ang[:, 0], joint_vel[:, 0]
    )
    future = RefState(future_pos, future_quat, joints, future_lin, future_ang, joint_vel)
    return goal_features(anchor, future, q0)


def test_future_pelvis_is_measured_in_the_current_heading():
    q0 = torch.zeros(4)
    yaw = quat_from_yaw(torch.tensor([torch.pi / 2]))
    anchor_pos = torch.tensor([[2.0, 3.0, 0.8]])
    future_pos = torch.tensor([[[2.0, 3.0, 0.8], [2.0, 5.0, 1.1]]])
    future_quat = yaw.view(1, 1, 4).expand(1, 2, 4).contiguous()
    future_lin = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    future_ang = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]]])
    joints = torch.zeros(1, 2, 4)
    joints[..., 1] = 0.2
    joint_vel = torch.full((1, 2, 4), 0.5)
    goal = _window(
        anchor_pos, yaw, future_pos, future_quat, future_lin, future_ang, joints, joint_vel, q0
    )
    torch.testing.assert_close(goal[0, 0, :4], torch.tensor([0.0, 0.2, 0.0, 0.0]))
    torch.testing.assert_close(goal[0, :, 4:8], joint_vel[0])
    torch.testing.assert_close(goal[0, 0, 8:11], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(goal[0, 1, 8:11], torch.tensor([2.0, 0.0, 0.3]), atol=1e-5, rtol=0)
    identity = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(goal[0, :, 11:17], identity.expand(2, 6), atol=1e-5, rtol=0)
    torch.testing.assert_close(goal[0, 1, 17:20], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(goal[0, 1, 20:23], torch.tensor([0.0, 0.0, 0.4]), atol=1e-5, rtol=0)


def test_numpy_goal_matches_the_training_window():
    rng = np.random.default_rng(11)
    length, joints = 12, 5
    quat = rng.normal(size=(length, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    pos = rng.normal(size=(length, 3))
    lin = rng.normal(size=(length, 3))
    ang = rng.normal(size=(length, 3))
    q = rng.normal(size=(length, joints))
    qd = rng.normal(size=(length, joints))
    q0 = rng.normal(size=joints)
    native = reference_heading_goal(pos, quat, lin, ang, q, qd, q0, 3, 4)
    anchor = RefState(
        torch.tensor(pos[3:4]), torch.tensor(quat[3:4]), torch.tensor(q[3:4]),
        torch.tensor(lin[3:4]), torch.tensor(ang[3:4]), torch.tensor(qd[3:4]),
    )
    index = np.minimum(3 + np.arange(4), length - 1)
    future = RefState(
        torch.tensor(pos[index][None]), torch.tensor(quat[index][None]), torch.tensor(q[index][None]),
        torch.tensor(lin[index][None]), torch.tensor(ang[index][None]), torch.tensor(qd[index][None]),
    )
    expected = goal_features(anchor, future, torch.tensor(q0)).reshape(-1).numpy()
    np.testing.assert_allclose(native, expected, atol=1e-5)

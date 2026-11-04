from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand
from supermimic.kinematics.rotations import robot_relative_rot6d

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Future reference-frame pelvis and joint features."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def reference_pelvis_rot6d(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Current reference pelvis orientation in the robot pelvis frame, rot6d.

    The six values are the first two columns of ``R_robot.T @ R_reference``.
    The command frame has already advanced, so this is the reference the
    policy is about to track.
    """
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return robot_relative_rot6d(command.robot_root_quat_w, command.root_quat_w)

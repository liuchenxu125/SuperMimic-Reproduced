"""Rotation helpers (torch, batched, quaternions in **wxyz**).

Re-exports mjlab's `lab_api.math` primitives so the whole project shares one
convention with the environment's reward and command code, and adds the few
operations the world model needs: rotation-vector exp/log, 6D rotation
representation and heading (yaw) frames.

All functions are differentiable and work on arbitrary leading batch dims.
"""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import (  # noqa: F401  (re-exported)
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    yaw_quat,
)

_EPS = 1.0e-8
_CONSTANTS: dict[tuple, torch.Tensor] = {}


def cached_constant(values: tuple[float, ...], like: torch.Tensor) -> torch.Tensor:
    """Constant tensor on ``like``'s device. Later calls reuse the same storage.

    Building ``torch.tensor(values, device=cuda)`` copies from the CPU. CUDA
    graph capture rejects that copy, so a captured rollout must reuse a tensor
    allocated on an earlier eager call.
    """
    key = (values, like.dtype, like.device.type, like.device.index)
    found = _CONSTANTS.get(key)
    if found is None:
        found = torch.tensor(values, dtype=like.dtype, device=like.device)
        _CONSTANTS[key] = found
    return found


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so that w >= 0 (removes the double cover ambiguity)."""
    return torch.where(q[..., :1] < 0, -q, q)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Exponential map: rotation vector (angle * axis) -> unit quaternion (wxyz).

    Uses a Taylor expansion of sin(a/2)/a near zero so gradients stay finite at v = 0.
    """
    angle_sq = (v * v).sum(-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1e-30))
    half = 0.5 * angle
    small = angle < 1e-4
    # sin(half)/angle with series fallback: 1/2 - angle^2/48
    k = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(half) / angle)
    w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(half))
    return torch.cat([w, v * k], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: unit quaternion (wxyz) -> rotation vector in (-pi, pi]."""
    q = quat_canonical(q)
    w = q[..., :1]
    xyz = q[..., 1:]
    s = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, w)
    small = s < 1e-6
    # angle / sin(angle/2) == angle / s ; series: 2 + angle^2/12 ... (with s ~ angle/2)
    k = torch.where(small, 2.0 + (angle * angle) / 12.0, angle / torch.where(small, torch.ones_like(s), s))
    return xyz * k


def quat_integrate(q: torch.Tensor, dtheta_body: torch.Tensor) -> torch.Tensor:
    """Apply a body-frame rotation increment: q ⊗ exp(dtheta)."""
    return quat_normalize(quat_mul(q, quat_from_rotvec(dtheta_body)))


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened (..., 6)."""
    m = matrix_from_quat(q)
    return m[..., :, :2].reshape(*q.shape[:-1], 6)


def robot_relative_rot6d(robot_quat: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
    """rot6d of ``R_robot.T @ R_reference``: first two columns, row-major.

    This is the reference pelvis orientation expressed in the current robot
    pelvis frame. It is not the heading-relative goal rotation, and it is not
    mjlab's row-packed ``motion_anchor_ori_b``.
    """
    relative = quat_mul(quat_inv(robot_quat), reference_quat)
    return rot6d_from_quat(relative)


def heading_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw angle of the heading frame of q (rotation of the body x-axis about world z)."""
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion of the pure-yaw heading frame (same as mjlab yaw_quat, but via atan2)."""
    return quat_from_yaw(heading_yaw(q))


def heading_matrix(q: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation from the root x-axis projected onto the world xy plane.

    This is the reference heading used for pelvis displacement and orientation.
    A pitched or rolled root keeps that projected heading; it is not the
    quaternion Euler yaw.
    """
    rotation = matrix_from_quat(q)
    forward_x, forward_y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(forward_x * forward_x + forward_y * forward_y + 1e-9)
    cosine, sine = forward_x / norm, forward_y / norm
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack(
        (cosine, -sine, zero, sine, cosine, zero, zero, zero, one), -1
    ).reshape(q.shape[:-1] + (3, 3))


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def projected_gravity(q_body_w: torch.Tensor) -> torch.Tensor:
    """Gravity direction (0,0,-1) expressed in the body frame."""
    g = cached_constant((0.0, 0.0, -1.0), q_body_w)
    g = g.expand(*q_body_w.shape[:-1], 3)
    return quat_apply_inverse(q_body_w, g)

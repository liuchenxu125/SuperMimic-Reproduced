"""Reference geometry checks for the selected training collision model.

These are data/model compatibility checks, not proof that a policy failure was
caused by the data. Do not classify fast but continuous movement as corruption.
"""

from __future__ import annotations

import mujoco
import numpy as np


def collision_model_with_floor():
    from supermimic.robots import get_robot

    spec = get_robot("casbot02").spec(collision_model="primitive")
    spec.worldbody.add_geom(
        name="reference_audit_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        friction=[1.0, 0.005, 0.0001],
    )
    return spec.compile()


def collision_depths(arrays, model):
    """Maximum positive penetration depths per frame, including self collisions.

    Native collision detection respects the XML collision masks/exclusions.
    References are not clamped to soft limits: initialization clipping is a
    separate implementation issue, not a defect of otherwise valid joint data.
    """
    root = list(arrays["body_names"]).index("base_link")
    n = len(arrays["joint_pos"])
    ground, own = np.zeros(n), np.zeros(n)
    data = mujoco.MjData(model)
    for t in range(n):
        data.qpos[:3] = arrays["body_pos_w"][t, root]
        data.qpos[3:7] = arrays["body_quat_w"][t, root]
        data.qpos[7:] = arrays["joint_pos"][t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        for contact in data.contact[: data.ncon]:
            i, j = int(contact.geom1), int(contact.geom2)
            depth = max(0.0, -float(contact.dist))
            if model.geom_bodyid[i] == 0 or model.geom_bodyid[j] == 0:
                ground[t] = max(ground[t], depth)
            else:
                own[t] = max(own[t], depth)
    return ground, own


def compatible_spans(ground, own, *, trim_prefix=0, max_depth=0.03, min_frames=51):
    """Half-open, contiguous valid intervals. Never join across invalid frames."""
    ground, own = np.asarray(ground), np.asarray(own)
    if ground.ndim != 1 or ground.shape != own.shape or min_frames < 3:
        raise ValueError("Need equal frame arrays and at least three frames per span")
    if not np.isfinite(ground).all() or not np.isfinite(own).all():
        raise ValueError("Nonfinite geometry diagnostics")
    if max_depth < 0 or not 0 <= trim_prefix <= len(ground):
        raise ValueError("Invalid depth threshold or prefix")
    valid = (ground <= max_depth) & (own <= max_depth)
    valid[:trim_prefix] = False
    delta = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts, ends = np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True) if b - a >= min_frames]

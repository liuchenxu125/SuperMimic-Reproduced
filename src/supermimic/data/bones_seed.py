"""G1 BONES-SEED filters adapted from CuboiLeo/SuperMimic-Reproduced ba61053.

Preserves its keyword and feasibility thresholds. Produces an inclusion
manifest, never deletes or edits motion NPZs. These G1 height/foot thresholds
must not be applied blindly to Casbot02.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

SONIC_KEYWORDS = [
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "_sit_",
    "sit_",
    "ladder",
    "crutch",
    "_bed_",
    "_ride_",
    "scooter",
    "stepdown",
    "box_hspu",
    "50cm_box_",
    "on_box",
    "fall_from",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "stair",
    "box_jump",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
]
SONIC_ACROBATICS = [
    "acrobatics_",
    "cartwheel",
    "handstand",
    "handstand_ff_",
    "safety_roll",
    "kozak_dance",
    "jump_to_shoulder_roll",
    "monkey_jump",
]
PAPER_EXTRA = ["_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over"]
KEYWORDS = sorted(set(SONIC_KEYWORDS + SONIC_ACROBATICS + PAPER_EXTRA))

_MODEL = None
_FOOT_GEOMS = None


def name_filter(name: str) -> str | None:
    for keyword in KEYWORDS:
        if keyword in name.lower():
            return f"name:{keyword}"
    return None


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _foot_sole_height(npz_path: str | Path, gate: np.ndarray) -> np.ndarray:
    """Exact lowest point of G1's 14 foot collision capsules above z=0."""
    global _MODEL, _FOOT_GEOMS
    if _MODEL is None:
        from supermimic.data.motion_convert import load_g1_model

        _MODEL = load_g1_model()
        _FOOT_GEOMS = [
            geom
            for geom in range(_MODEL.ngeom)
            if re.fullmatch(r"(left|right)_foot[1-7]_collision", _MODEL.geom(geom).name)
        ]
        if len(_FOOT_GEOMS) != 14 or not np.all(
            _MODEL.geom_type[_FOOT_GEOMS] == mujoco.mjtGeom.mjGEOM_CAPSULE
        ):
            raise ValueError("BONES feasibility requires the pinned G1 capsule feet")
    model = _MODEL
    data = mujoco.MjData(model)
    with np.load(npz_path, allow_pickle=False) as motion:
        root = list(motion["body_names"].astype(str)).index("pelvis")
        root_pos, root_quat = motion["body_pos_w"][:, root], motion["body_quat_w"][:, root]
        joints = motion["joint_pos"]
    radius, half = model.geom_size[_FOOT_GEOMS, 0], model.geom_size[_FOOT_GEOMS, 1]
    result = []
    for frame in np.flatnonzero(gate):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = (
            root_pos[frame],
            root_quat[frame],
            joints[frame],
        )
        mujoco.mj_kinematics(model, data)
        z = data.geom_xpos[_FOOT_GEOMS, 2]
        axis_z = data.geom_xmat[_FOOT_GEOMS].reshape(-1, 3, 3)[:, 2, 2]
        result.append((z - half * np.abs(axis_z) - radius).min())
    return np.asarray(result)


def feasibility_filter(npz_path: str | Path) -> str | None:
    with np.load(npz_path, allow_pickle=False) as motion:
        names = list(motion["body_names"].astype(str))
        pos, quat = motion["body_pos_w"], motion["body_quat_w"]
        joints, jv, lin = motion["joint_pos"], motion["joint_vel"], motion["body_lin_vel_w"]
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if ("robot" in motion and str(motion["robot"].item()) != "g1") or "pelvis" not in names:
            raise ValueError(f"{npz_path}: BONES-SEED feasibility thresholds are G1-only")
    n = len(joints)
    if (
        n < 2
        or joints.shape != (n, 29)
        or jv.shape != joints.shape
        or pos.shape != (n, len(names), 3)
        or quat.shape != (n, len(names), 4)
        or lin.shape != pos.shape
        or not np.isclose(fps, 50.0)
    ):
        raise ValueError(f"{npz_path}: expected a 50 Hz G1 motion with matching arrays")
    if not all(np.isfinite(array).all() for array in (pos, quat, joints, jv, lin)):
        raise ValueError(f"{npz_path}: nonfinite motion")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-4):
        raise ValueError(f"{npz_path}: invalid body quaternion")
    zmin = pos[:, :, 2].min(axis=1)
    if _longest_run(zmin > 0.20) >= 0.6 * fps:
        return "elevated_support"
    if (zmin < -0.12).any() or (zmin < -0.08).mean() > 0.80:
        return "ground_penetration"
    pelvis = names.index("pelvis")
    q = quat[:, pelvis]
    gz = -(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    gate = (
        (gz < -0.9)
        & (pos[:, pelvis, 2] > 0.60)
        & (pos[:, pelvis, 2] < 0.85)
        & (np.linalg.norm(lin[:, pelvis], axis=1) < 0.1)
        & (np.sqrt((jv**2).mean(axis=1)) < 0.5)
    )
    if _longest_run(gate) >= 0.5 * fps:
        floating = gate.copy()
        floating[gate] = _foot_sole_height(npz_path, gate) > 0.02
        if _longest_run(floating) >= 0.5 * fps:
            return "floating_feet"
    return None


def filter_clip(path: str | Path) -> tuple[str, int, str | None]:
    path = Path(path)
    reason = name_filter(path.stem)
    if reason is None:
        reason = feasibility_filter(path)
    with np.load(path, allow_pickle=False) as motion:
        frames = len(motion["joint_pos"])
    return path.stem, frames, reason

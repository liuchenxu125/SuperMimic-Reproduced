"""Versioned references for the native controller (little-endian float32)."""
import struct

import numpy as np


def encode_motion(arrays, encoding):
    """Keep raw world velocities for goals anchored at the current reference.

    A heading-relative window cannot be precomputed independently per frame.
    SMREF002 stores p3, wxyz4, q27, dq27, world linear3, world angular3.
    SMREF001 remains readable for the previous per-frame 64-feature actors.
    """
    keys = ["root_pos_w", "root_quat_w", "joint_pos", "joint_vel"]
    if encoding == "reference_heading_v1":
        magic, width = b"SMREF002", 67
        keys += ["root_lin_vel_w", "root_ang_vel_w"]
    elif encoding == "per_frame_v1":
        magic, width = b"SMREF001", 125
        keys += ["goal_features"]
    else:
        raise ValueError(f"Unsupported goal encoding: {encoding}")
    records = np.concatenate([arrays[k] for k in keys], axis=1).astype("<f4")
    if records.ndim != 2 or records.shape[1] != width or len(records) < 2 or not np.isfinite(records).all():
        raise ValueError("Invalid motion shape/data")
    if not np.allclose(np.linalg.norm(records[:, 3:7], axis=1), 1., atol=.01):
        raise ValueError("Invalid reference quaternion")
    return magic + struct.pack("<II", len(records), width) + records.tobytes(), len(records)

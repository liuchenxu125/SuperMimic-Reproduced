#!/usr/bin/env python3
"""Derive the Casbot02 full-body GMR config from GMR and the proven HL axis maps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "src/supermimic/teleop/ik_configs/xrobot_to_casbot_casbot02.json"

# Keep GMR's segment frames (yaw links after the shoulder/hip ball joint),
# rather than constraining a roll-only intermediate link to a full 3D rotation.
BODY_MAP = {
    "pelvis": ("base_link", "hips_mean"),
    "torso_link": ("waist_yaw_link", "shoulder_mean"),
}
for side in ("left", "right"):
    BODY_MAP.update({
        f"{side}_hip_yaw_link": (f"{side}_leg_pelvic_yaw_link", f"{side}_up_leg"),
        f"{side}_knee_link": (f"{side}_leg_knee_pitch_link", f"{side}_leg"),
        f"{side}_toe_link": (f"{side}_leg_ankle_roll_link", f"{side}_foot"),
        f"{side}_shoulder_yaw_link": (f"{side}_shoulder_yaw_link", f"{side}_arm"),
        f"{side}_elbow_link": (f"{side}_elbow_pitch_link", f"{side}_fore_arm"),
        f"{side}_wrist_yaw_link": (f"{side}_wrist_roll_link", f"{side}_hand"),
    })


def axis_matrix(config, key):
    item = config["key_frame_config"][key]
    axes = np.column_stack([item["axis_map_cols"][k] for k in "xyz"])
    angles = item["offset_deg_xyz"]
    # HL uses Rx @ Ry @ Rz, not SciPy's extrinsic xyz sequence.
    for axis, angle in zip("xyz", angles):
        axes = axes @ Rotation.from_euler(axis, angle, degrees=True).as_matrix()
    return axes


def build(gmr_root, hl_root):
    source = gmr_root / "general_motion_retargeting/ik_configs/xrobot_to_g1.json"
    paths = [source, hl_root / "config/robot/g1.yaml",
             hl_root / "config/robot/casbot02_primitives.yaml"]
    config = json.loads(source.read_text())
    g1, casbot = [yaml.safe_load(p.read_text()) for p in paths[1:]]
    model = mujoco.MjModel.from_xml_path(str(gmr_root / "assets/unitree_g1/g1_mocap_29dof.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = {}
        for old_body, entry in config[table_name].items():
            new_body, semantic = BODY_MAP[old_body]
            correction = axis_matrix(g1, semantic).T @ axis_matrix(casbot, semantic)
            offset = np.array(entry[3], dtype=float)
            if old_body.endswith("_toe_link"):
                # GMR has a toe helper 10cm ahead/2cm below its ankle.
                # Casbot uses an ankle frame, not that virtual toe frame.
                ankle = old_body.replace("toe_link", "ankle_roll_link")
                i, j = model.body(old_body).id, model.body(ankle).id
                offset += data.xmat[i].reshape(3, 3).T @ (data.xpos[j] - data.xpos[i])
            quat = (Rotation.from_quat(entry[4], scalar_first=True)
                    * Rotation.from_matrix(correction)).as_quat(scalar_first=True)
            table[new_body] = [*entry[:3], np.round(correction.T @ offset, 12).tolist(),
                               np.round(quat, 12).tolist()]
        config[table_name] = table
    config["robot_root_name"] = "base_link"
    config["_provenance"] = {
        "source": "https://github.com/YanjieZe/GMR/blob/master/general_motion_retargeting/ik_configs/xrobot_to_g1.json",
        "axis_formula": "R_new = R_gmr @ inverse(A_g1) @ A_casbot; p_new = inverse(C) @ p_old",
        "files_sha256": {str(p.relative_to(gmr_root if p == source else hl_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "notes": ["Full-body xrobot/PICO input only; quaternions wxyz, offsets local after rotation.",
                  "HL wrist basis correction retained; no palm target.",
                  "GMR segment yaw frames, weights and scale table retained. HL per-bone scaling/contact/export corrections are not encoded by this JSON.",
                  "Toe helper offset converted to ankle origin; table1/table2 offsets must match for upstream GMR."],
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmr-root", type=Path, default=Path.home() / "Desktop/GMR")
    parser.add_argument("--hl-root", type=Path, default=Path.home() / "Desktop/hl_robot_retargeter")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.gmr_root, args.hl_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)

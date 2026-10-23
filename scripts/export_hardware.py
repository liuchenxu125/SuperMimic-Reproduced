"""Export the verified Casbot02 sim2sim bundle for the C++ hardware controller."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.deployment.export import ActorOnly
from supermimic.policy.nets import Policy, PolicyCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("outputs/casbot02_sim2sim"))
    parser.add_argument("--clip", default="walk4_subject1")
    parser.add_argument(
        "--output", type=Path, default=Path("rl_controller/config/model/supermimic_walk4")
    )
    args = parser.parse_args()
    meta = json.loads((args.bundle / "manifest.json").read_text())
    checkpoint = Path(meta["checkpoint_path"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != meta["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since sim2sim validation")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_goal = 640 + (9 if meta["anchor_obs"] else 0)
    if (meta["obs_dim"], meta["goal_dim"]) != (870, expected_goal):
        raise ValueError("Bundle has inconsistent Casbot02 IO dimensions")
    if ckpt["config"]["env"].get("anchor_obs", False) != meta["anchor_obs"]:
        raise ValueError("Cannot change anchor_obs when exporting existing weights")
    policy = Policy(PolicyCfg(**ckpt["config"]["policy"])).eval()
    policy.load_state_dict(ckpt["policy"])
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.onnx.export(
        ActorOnly(policy),
        (torch.zeros(1, meta["obs_dim"]), torch.zeros(1, meta["goal_dim"])),
        str(args.output / "actor.onnx"),
        input_names=["obs_hist", "goal"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    import onnx

    graph = onnx.load(args.output / "actor.onnx")
    onnx.checker.check_model(graph)
    if graph.ir_version > 10:
        raise ValueError("ONNX IR is too new for the bundled ORT 1.19")
    with np.load(args.bundle / meta["motions"][args.clip]) as motion:
        records = np.concatenate(
            [
                motion[k]
                for k in ("root_pos_w", "root_quat_w", "joint_pos", "joint_vel", "goal_features")
            ],
            axis=1,
        ).astype("<f4")
    assert records.shape[1] == 125
    with (args.output / "motion.bin").open("wb") as f:
        f.write(b"SMREF001" + struct.pack("<II", len(records), records.shape[1]))
        f.write(records.tobytes())
    model = mujoco.MjModel.from_xml_path(str(args.bundle / meta["model_file"]))
    # Explicit policy -> SDK mapping: waist is SDK 28, arms are SDK 12..25.
    sdk_names = [f"leg_{side}{j}_joint" for side in ("l", "r") for j in range(1, 7)]
    sdk_names += ["waist_yaw_joint"]
    sdk_names += [
        f"{side}_{name}_joint"
        for side in ("left", "right")
        for name in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        )
    ]
    manifest = {
        k: meta[k]
        for k in (
            "robot",
            "checkpoint_sha256",
            "checkpoint_iteration",
            "joint_names",
            "motors",
            "obs_dim",
            "goal_dim",
            "history",
            "future_frames",
            "anchor_obs",
            "control_dt",
            "reset_goal_advance",
            "height_threshold",
            "tilt_threshold",
            "soft_joint_limits",
        )
    }
    manifest.update(
        format="supermimic-hardware-v1",
        actor_file="actor.onnx",
        motion_file="motion.bin",
        clip=args.clip,
        frames=len(records),
        sdk_joint_names=sdk_names,
        joint_limits=model.jnt_range[1:].tolist(),
        actor_sha256=hashlib.sha256((args.output / "actor.onnx").read_bytes()).hexdigest(),
        motion_sha256=hashlib.sha256((args.output / "motion.bin").read_bytes()).hexdigest(),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "clip": args.clip,
                "frames": len(records),
                "onnx_opset": 17,
                "onnx_ir": graph.ir_version,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Build a portable actor + MuJoCo scene + reference-motion bundle on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from torch import nn

from supermimic.envs.mdp.motion_library import RefState, goal_features
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.goal_spec import anchor_goal_dim
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots import get_robot


class ActorOnly(nn.Module):
    """Export both inputs explicitly, with the fixed scaling kept inside Policy."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs_hist, goal):
        return self.policy(obs_hist, goal)


def validate_policy(policy: Policy) -> None:
    """Reject divergent checkpoints before writing a deployable bundle."""
    bad = [name for name, value in policy.state_dict().items() if not torch.isfinite(value).all()]
    if bad:
        raise ValueError(
            f"Policy contains NaN/Inf in {len(bad)} tensors (e.g. {bad[0]}). "
            "Training diverged; use a finite checkpoint or retrain. Cannot export for deployment."
        )
    obs_dim = (3 * policy.cfg.num_joints + 6) * policy.cfg.history
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(1301)
        for random in (False, True):
            obs = torch.randn(1, obs_dim) * 0.2 if random else torch.zeros(1, obs_dim)
            goal = (
                torch.randn(1, policy.cfg.goal_dim) * 0.2
                if random
                else torch.zeros(1, policy.cfg.goal_dim)
            )
            action = policy(obs, goal)
            if action.shape != (1, policy.cfg.num_joints) or not torch.isfinite(action).all():
                raise ValueError(
                    "Policy inference produced invalid actions; cannot export for deployment"
                )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_bundle(checkpoint: Path, motions: list[Path], output: Path) -> dict:
    checkpoint, output = Path(checkpoint), Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Bundle already exists: {output}; choose a new output directory")
    if not motions or len({p.stem for p in motions}) != len(motions):
        raise ValueError("Supply reference motions with unique clip names")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg.get("robot", "g1") != "casbot02":
        raise ValueError("This sim2sim exporter currently supports Casbot02 only")
    pc = cfg["policy"]
    pcfg = PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in pc.items()})
    policy = Policy(pcfg).cpu().eval()
    policy.load_state_dict(ck["policy"])
    validate_policy(policy)
    n, k = pcfg.num_joints, cfg["env"]["future_frames"]
    anchor = cfg["env"].get("anchor_obs", False)
    anchor_ori = cfg["env"].get("anchor_ori_obs", False)
    obs_dim = (3 * n + 6) * pcfg.history
    if n != 27 or pcfg.goal_dim != k * (2 * n + 10) + anchor_goal_dim(anchor, anchor_ori):
        raise ValueError("Checkpoint dimensions are inconsistent with the selected robot/interface")
    output.mkdir(parents=True, exist_ok=True)
    profile = get_robot("casbot02")
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(motions[0])],
        robot="casbot02",
        num_envs=1,
        record_transitions=False,
        domain_randomization=False,
        reset_noise=False,
        obs_noise=False,
        anchor_obs=anchor,
        anchor_ori_obs=anchor_ori,
        actuator_delay=cfg["env"].get("actuator_delay", False),
        actuator_delay_mode=cfg["env"].get("actuator_delay_mode", "episode"),
        collision_model=cfg["env"].get("collision_model", "mesh"),
    )
    env_cfg = make_env_cfg(args)
    scene = Scene(env_cfg.scene, "cpu")  # No simulator, CUDA allocation, or training.
    model = scene.compile()
    env_cfg.sim.mujoco.apply(model)
    entity = scene.entities["robot"]
    _, names = entity.find_joints_by_actuator_names((".*",))
    if list(entity.joint_names) != names:
        raise ValueError("Observation and action orders differ; exporter needs an explicit remap")
    full_names = [model.joint(i).name for i in range(1, model.njnt)]
    if [name.removeprefix("robot/") for name in full_names] != names:
        raise ValueError("Scene joint order differs from policy")
    q0 = model.key_qpos[0, 7:].astype(np.float32)
    all_motors = profile.motor_parameters()["motors"]
    scales = profile.action_scale()
    motors = [
        {
            "name": name,
            "q0": float(q0[i]),
            "scale": scales[name],
            "kp": all_motors[name]["stiffness"],
            "kd": all_motors[name]["damping"],
            "effort_limit_nm": all_motors[name]["effort_limit"],
            "armature": all_motors[name]["armature"],
        }
        for i, name in enumerate(names)
    ]
    limits = model.jnt_range[1:]
    mid, half = limits.mean(-1), (limits[:, 1] - limits[:, 0]) * 0.5 * 0.9
    feet = [
        model.geom(i).name
        for i in range(model.ngeom)
        if re.fullmatch(profile.foot_geom_regex, model.geom(i).name.removeprefix("robot/"))
    ]
    if len(feet) not in (2, 14) or not all(
        any(name.startswith("robot/" + side + "_foot") for name in feet)
        for side in ("left", "right")
    ):
        raise ValueError("Expected both Casbot02 foot collision groups")
    opts = {}
    for name in [
        "timestep",
        "integrator",
        "impratio",
        "cone",
        "jacobian",
        "solver",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
        "gravity",
        "disableflags",
        "enableflags",
    ]:
        value = getattr(model.opt, name)
        opts[name] = value.tolist() if isinstance(value, np.ndarray) else value
    scene.write(output / "scene")  # Includes portable XML + STL assets.
    # MuJoCo's XML writer rounds floating-point actuator gains. Keep the
    # training gains and torque caps exactly equal to the deployment manifest.
    scene_xml = output / "scene/scene.xml"
    tree = ET.parse(scene_xml)
    by_name = {p["name"]: p for p in motors}
    for actuator in tree.getroot().find("actuator"):
        p = by_name[actuator.attrib["joint"].removeprefix("robot/")]
        actuator.set("gainprm", str(p["kp"]))
        actuator.set("biasprm", f"0 {-p['kp']} {-p['kd']}")
        actuator.set("forcerange", f"{-p['effort_limit_nm']} {p['effort_limit_nm']}")
    tree.write(scene_xml, encoding="unicode")
    (output / "motions").mkdir(exist_ok=True)
    clip_files, source_motions = {}, {}
    for path in motions:
        with np.load(path, allow_pickle=False) as data:
            if data["joint_names"].tolist() != names:
                raise ValueError(f"{path}: reference joint order differs from checkpoint/model")
            fps = float(data["fps"].item())
            if not np.isclose(fps, 50):
                raise ValueError(f"{path}: expected 50 Hz reference")
            root_idx = data["body_names"].tolist().index(profile.root_body)
            q, qd = data["joint_pos"], data["joint_vel"]
            root_pos = data["body_pos_w"][:, root_idx]
            root_quat = data["body_quat_w"][:, root_idx]
            root_lin = data["body_lin_vel_w"][:, root_idx]
            root_ang = data["body_ang_vel_w"][:, root_idx]
            ref = RefState(
                *[
                    torch.from_numpy(a.copy()).float()
                    for a in (root_pos, root_quat, q, root_lin, root_ang, qd)
                ]
            )
            with torch.inference_mode():
                features = goal_features(ref, torch.from_numpy(q0)).numpy()
            dest = output / "motions" / f"{path.stem}.npz"
            np.savez_compressed(
                dest,
                fps=np.array([fps]),
                joint_names=np.array(names),
                joint_pos=q,
                joint_vel=qd,
                root_pos_w=root_pos,
                root_quat_w=root_quat,
                root_lin_vel_w=root_lin,
                root_ang_vel_w=root_ang,
                goal_features=features,
            )
        clip_files[path.stem] = str(dest.relative_to(output))
        source_motions[path.stem] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "frames": len(q),
        }
    wrapper = ActorOnly(policy)
    torch.manual_seed(1301)
    obs = torch.randn(1, obs_dim) * 0.1
    goal = torch.randn(1, pcfg.goal_dim) * 0.1
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (obs, goal), check_trace=True).eval()
        traced = torch.jit.freeze(traced)
        actor_file = output / "actor.pt"
        torch.jit.save(traced, str(actor_file))
        loaded = torch.jit.load(str(actor_file)).eval()
        errors = []
        for batch in [1, 4, 1]:
            x, y = torch.randn(batch, obs_dim) * 0.2, torch.randn(batch, pcfg.goal_dim) * 0.2
            before, after = wrapper(x, y), loaded(x, y)
            torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)
            errors.append(float((before - after).abs().max()))
    manifest = {
        "format": "supermimic-native-mujoco-v1",
        "robot": "casbot02",
        "actor_file": "actor.pt",
        "model_file": "scene/scene.xml",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_iteration": ck.get("iteration"),
        "joint_names": names,
        "model_joint_names": full_names,
        "root_body_name": "robot/" + profile.root_body,
        "foot_geom_names": feet,
        "motors": motors,
        "soft_joint_limits": np.stack((mid - half, mid + half), -1).tolist(),
        "obs_dim": obs_dim,
        "goal_dim": pcfg.goal_dim,
        "history": pcfg.history,
        "future_frames": k,
        "anchor_obs": anchor,
        "anchor_ori_obs": anchor_ori,
        "actuator_delay": cfg["env"].get("actuator_delay", False),
        "actuator_delay_mode": cfg["env"].get("actuator_delay_mode", "episode"),
        "collision_model": cfg["env"].get("collision_model", "mesh"),
        "control_dt": 0.02,
        "decimation": 4,
        "mujoco_options": opts,
        "height_threshold": cfg.get("world_model", {}).get("height_threshold", 0.25),
        "tilt_threshold": cfg.get("world_model", {}).get("tilt_threshold", 0.8),
        "reset_goal_advance": 1,
        "reference_timing_note": "Current MultiClipMotionCommand advances one frame during reset command compute; state starts at f, first goal at f+1. Kept for checkpoint parity.",
        "motions": clip_files,
        "source_motions": source_motions,
        "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__},
        "export_validation": {
            "actor_max_abs_error": max(errors),
            "batches_checked": [1, 4, 1],
            "model_file_sha256": digest(output / "scene/scene.xml"),
            "actor_sha256": digest(actor_file),
        },
        "notes": [
            "No synthetic sensor/action/reset noise or domain randomization.",
            "Foot sliding friction fixed to 1.0, matching benchmark.",
            "No action EMA, acceleration clamp, tanh, or target clipping added: not used in SuperMimic training.",
            "Native CPU MuJoCo and GPU MuJoCo Warp may have different contact trajectories.",
            "Root position/orientation read from simulation ground truth, not a hardware estimator.",
        ],
    }
    shutil.copyfile(Path(__file__).with_name("sim2sim.py"), output / "sim2sim.py")
    manifest["export_validation"]["runtime_sha256"] = digest(output / "sim2sim.py")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("logs/casbot02_lafan1/ckpt_latest.pt"))
    p.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1_casbot02"))
    p.add_argument(
        "--clips", nargs="+", default=["walk4_subject1", "walk3_subject5", "dance1_subject1"]
    )
    p.add_argument("--output", type=Path, default=Path("outputs/casbot02_sim2sim"))
    a = p.parse_args()
    torch.set_num_threads(2)
    paths = [a.npz_dir / f"{name}.npz" for name in a.clips]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        p.error(f"Missing reference files: {missing}")
    result = export_bundle(a.ckpt, paths, a.output)
    print(
        f"Exported {a.output}: {result['obs_dim']} + {result['goal_dim']} -> {len(result['joint_names'])}"
    )
    print("Actor maximum absolute error:", result["export_validation"]["actor_max_abs_error"])
    print("Clips:", ", ".join(result["motions"]))


if __name__ == "__main__":
    main()

"""Compare deployment inputs and actuator targets against the training environment.

Uses GPU MuJoCo Warp only as a source of states. This checks the inference
boundary, not bitwise equality of CPU/GPU contact trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.deployment.sim2sim import Sim2Sim, observation
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.eval.benchmark import load_policy_from_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("outputs/casbot02_sim2sim"))
    p.add_argument("--ckpt", type=Path, default=Path("logs/casbot02_lafan1/ckpt_latest.pt"))
    p.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1_casbot02"))
    p.add_argument("--clip", default="walk4_subject1")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("outputs/casbot02_sim2sim/input_parity.json"))
    a = p.parse_args()
    if a.steps < 1:
        p.error("--steps must be positive")
    from mjlab.envs import ManagerBasedRlEnv

    sim = Sim2Sim(a.bundle)
    sim.reset(a.clip)
    policy, _ = load_policy_from_checkpoint(a.ckpt, "cpu")
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(a.npz_dir / f"{a.clip}.npz")],
        robot="casbot02",
        num_envs=1,
        record_transitions=False,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        action_noise_std=0,
        anchor_obs=sim.meta["anchor_obs"],
        future_frames=sim.meta["future_frames"],
        history_length=sim.meta["history"],
        episode_length_s=1e5,
        sampling_mode="start",
    )
    cfg = make_env_cfg(args)
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    max_errors = {name: 0.0 for name in ["observation", "goal", "actor", "position_target"]}
    count = 0
    try:
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = 0
        obs, _ = env.reset()
        for t in range(a.steps):
            qp = env.sim.data.qpos[0].cpu().numpy().copy()
            qv = env.sim.data.qvel[0].cpu().numpy().copy()
            qp[:3] -= env.scene.env_origins[0].cpu().numpy()
            sim.data.qpos[:] = qp
            sim.data.qvel[:] = qv
            frame = observation(qp, qv, sim.previous_action, sim.q0)
            if t == 0:
                sim.history.reset(frame)
            else:
                sim.history.append(frame)
            sim.target_frame = int(cmd.frame[0])
            x, g = sim.inputs()
            max_errors["observation"] = max(
                max_errors["observation"], float(np.abs(x - obs["policy"][0].cpu().numpy()).max())
            )
            max_errors["goal"] = max(
                max_errors["goal"], float(np.abs(g - obs["goal"][0].cpu().numpy()).max())
            )
            with torch.inference_mode():
                tx, tg = torch.from_numpy(x[None]), torch.from_numpy(g[None])
                original = policy(tx, tg)
                exported = sim.actor(tx, tg)
            max_errors["actor"] = max(max_errors["actor"], float((original - exported).abs().max()))
            action = exported[0].numpy()
            targets = sim.position_targets(action)
            obs, _, term, tout, _ = env.step(exported.to("cuda:0"))
            expected = env.action_manager.get_term("joint_pos")._processed_actions[0].cpu().numpy()
            max_errors["position_target"] = max(
                max_errors["position_target"], float(np.abs(targets - expected).max())
            )
            sim.previous_action[:] = action
            count += 1
            if bool((term | tout)[0]):
                break
    finally:
        env.close()
    result = {
        "clip": a.clip,
        "samples": count,
        "max_abs_errors": max_errors,
        "tolerance": 2e-5,
        "passed": all(v < 2e-5 for v in max_errors.values()),
        "scope": "Same GPU-simulated states through independent deployment input builder and CPU exported actor; not a CPU/GPU dynamics equivalence test.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

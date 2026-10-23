"""Exercise the hardware C++ actor in closed-loop MuJoCo without ROS or motors."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.deployment.sim2sim import Sim2Sim, observation, rotation


class CppActor:
    def __init__(self, sim, probe, bundle, start_frame, explicit_pd):
        self.sim, self.torch_actor = sim, sim.actor
        self.proc = subprocess.Popen(
            [str(probe.resolve()), str(bundle.resolve()), str(start_frame)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.max_errors = {"observation": 0.0, "goal": 0.0, "actor": 0.0, "target": 0.0}
        self.latency = []
        self.calls = 0
        self.goal_dim = sim.meta["goal_dim"]
        if explicit_pd:
            # Use the same model/transmissions, replacing implicit position PD
            # with bounded explicit effort PD at 1 kHz, as in the ROS adapter.
            sim.model.opt.timestep = 0.001
            sim.decimation = 20
            sim.model.actuator_gainprm[:] = 0
            sim.model.actuator_gainprm[:, 0] = 1
            sim.model.actuator_biasprm[:] = 0

    def __call__(self, obs, goal):
        sim, data = self.sim, self.sim.data
        record = np.concatenate(
            (data.qpos[7:], data.qvel[6:], data.qpos[:3], data.qpos[3:7], data.qvel[3:6])
        ).astype("<f8")
        began = time.perf_counter()
        self.proc.stdin.write(record.tobytes())
        self.proc.stdin.flush()
        count = 870 + self.goal_dim + 54
        raw = self.proc.stdout.read(count * 4)
        if len(raw) != count * 4:
            raise RuntimeError(f"C++ probe stopped: {self.proc.poll()}")
        self.latency.append(time.perf_counter() - began)
        result = np.frombuffer(raw, dtype="<f4")
        end_goal = 870 + self.goal_dim
        cpp_obs, cpp_goal = result[:870], result[870:end_goal]
        action, target = result[end_goal : end_goal + 27], result[end_goal + 27 :]
        for name, actual, expected in (
            ("observation", cpp_obs, obs.numpy()[0]),
            ("goal", cpp_goal, goal.numpy()[0]),
            ("target", target, (sim.q0 + sim.scale * action)),
        ):
            error = float(np.max(np.abs(actual - expected)))
            self.max_errors[name] = max(self.max_errors[name], error)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=3e-6)
        if self.calls < 32 or self.calls % 100 == 0:
            with torch.inference_mode():
                expected = self.torch_actor(
                    torch.from_numpy(cpp_obs.copy()[None]), torch.from_numpy(cpp_goal.copy()[None])
                )[0].numpy()
            self.max_errors["actor"] = max(
                self.max_errors["actor"], float(np.max(np.abs(expected - action)))
            )
            np.testing.assert_allclose(action, expected, atol=2e-4, rtol=2e-5)
        self.calls += 1
        return torch.from_numpy(action.copy()[None])

    def close(self):
        self.proc.stdin.close()
        self.proc.stdout.close()
        if self.proc.wait(timeout=5) != 0:
            raise RuntimeError("C++ probe failed")


def explicit_step(sim):
    obs, goal = sim.inputs()
    action = sim.actor(torch.from_numpy(obs[None]), torch.from_numpy(goal[None]))[0].numpy()
    target = sim.q0 + sim.scale * action
    motors = sim.meta["motors"]
    kp, kd, effort = (np.array([m[key] for m in motors]) for key in ("kp", "kd", "effort_limit_nm"))
    for _ in range(20):
        tau = np.clip(kp * (target - sim.data.qpos[7:]) - kd * sim.data.qvel[6:], -effort, effort)
        sim.data.ctrl[:] = tau[sim.actuator_to_joint]
        mujoco.mj_step(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)
    f, ref = sim.target_frame, sim.reference
    height = abs(sim.data.qpos[2] - ref.root_pos[f, 2])
    tilt = abs(rotation(sim.data.qpos[3:7])[2, 2] - rotation(ref.root_quat[f])[2, 2])
    sim.failed = bool(height > sim.meta["height_threshold"] or tilt > sim.meta["tilt_threshold"])
    sim.steps += 1
    sim.previous_action[:] = action
    sim.metrics.append(
        [
            height,
            tilt,
            np.linalg.norm(sim.data.qpos[:3] - ref.root_pos[f]),
            np.sqrt(np.mean((sim.data.qpos[7:] - ref.joint_pos[f]) ** 2)),
            np.max(np.abs(action)),
        ]
    )
    sim.history.append(observation(sim.data.qpos, sim.data.qvel, action, sim.q0))
    sim.target_frame = min(f + 1, ref.length - 1)
    sim.completed = sim.steps >= ref.length - sim.start_frame
    sim.finished = sim.failed or sim.completed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-bundle", type=Path, default=Path("outputs/casbot02_sim2sim"))
    p.add_argument(
        "--hardware-bundle", type=Path, default=Path("rl_controller/config/model/supermimic_walk4")
    )
    p.add_argument(
        "--probe",
        type=Path,
        default=Path("rl_controller/build/supermimic_offline/supermimic_probe"),
    )
    p.add_argument("--seconds", type=float)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--from-rest", action="store_true")
    p.add_argument("--explicit-pd", action="store_true")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    if args.seconds is not None and args.seconds < 0.02:
        p.error("--seconds must include at least one 20 ms control step")
    sim = Sim2Sim(args.sim_bundle)
    meta = json.loads((args.hardware_bundle / "manifest.json").read_text())
    sim.reset(meta["clip"], args.start_frame, start_xy=[1.2, -0.6], start_yaw=0.7)
    if args.from_rest:
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        sim.history.reset(observation(sim.data.qpos, sim.data.qvel, sim.previous_action, sim.q0))
    cpp = CppActor(sim, args.probe, args.hardware_bundle, args.start_frame, args.explicit_pd)
    sim.actor = cpp
    try:
        limit = sim.reference.length if args.seconds is None else round(args.seconds / sim.dt)
        while not sim.finished and sim.steps < limit:
            if args.explicit_pd:
                explicit_step(sim)
            else:
                sim.step()
    finally:
        cpp.close()
    report = sim.report()
    warm_latency = cpp.latency[1:] or cpp.latency
    report.update(
        actor_runtime="C++ ONNX Runtime 1.19 CPU",
        max_abs_errors=cpp.max_errors,
        from_rest=args.from_rest,
        explicit_pd_1000hz=args.explicit_pd,
        host_roundtrip_ms={
            k: float(v)
            for k, v in zip(
                ("median", "p99", "max_after_warmup"),
                np.array(
                    [np.median(warm_latency), np.percentile(warm_latency, 99), np.max(warm_latency)]
                )
                * 1000,
            )
        },
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if sim.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

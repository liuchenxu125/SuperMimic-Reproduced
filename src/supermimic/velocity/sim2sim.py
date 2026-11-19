"""Body-frame velocity playback for the exported joint-training actor.

Reuses a native MuJoCo bundle for the robot, PD gains and timing. The policy
input is the 870-d proprio history plus ``[vx, vy, wz]`` in the root frame
(m/s, m/s, rad/s). There is no motion reference and no reference-orientation
history. Playback does not apply actuator delay, matching the motion player.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from supermimic.deployment.sim2sim import (
    History,
    load_model,
    native_viewer,
    observation,
    rotation,
)

DEFAULT_LIMITS = np.array([[-1.5, 1.5], [-0.5, 0.5], [-3.14, 3.14]], dtype=np.float64)


def command_limits(actor: Path) -> np.ndarray:
    """Training clamp from the export manifest, otherwise the task defaults."""
    meta_path = Path(actor).parent / "manifest.json"
    if not meta_path.exists():
        return DEFAULT_LIMITS.copy()
    cfg = json.loads(meta_path.read_text()).get("velocity_config") or {}
    keys = ("lin_vel_x_range", "lin_vel_y_range", "ang_vel_z_range")
    if not all(key in cfg for key in keys):
        return DEFAULT_LIMITS.copy()
    limits = np.array([cfg[key] for key in keys], dtype=np.float64)
    if limits.shape != (3, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError(f"Invalid velocity limits in {meta_path}")
    return limits


def clamp_command(command, limits) -> np.ndarray:
    value = np.asarray(command, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Velocity command must be three finite numbers: vx, vy, wz")
    return np.clip(value, limits[:, 0], limits[:, 1])


def body_velocity(qpos, qvel) -> np.ndarray:
    """Root linear xy and yaw rate in the current body frame.

    Free-joint linear qvel is world-frame; angular qvel is already body-local.
    """
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    linear = rotation(quat).T @ np.asarray(qvel[:3], dtype=np.float64)
    return np.array([linear[0], linear[1], qvel[5]], dtype=np.float64)


class VelocitySim:
    def __init__(self, bundle: Path, actor: Path, threads: int = 2):
        self.bundle = Path(bundle).resolve()
        self.actor_path = Path(actor).resolve()
        self.meta = json.loads((self.bundle / "manifest.json").read_text())
        if self.meta.get("format") != "supermimic-native-mujoco-v1":
            raise ValueError("Unsupported deployment bundle")
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.limits = command_limits(self.actor_path)
        self.actor = torch.jit.load(str(self.actor_path), map_location="cpu").eval()
        self.model, self.data, self.actuator_to_joint = load_model(self.bundle, self.meta)
        self.q0 = np.array([p["q0"] for p in self.meta["motors"]], dtype=np.float64)
        self.scale = np.array([p["scale"] for p in self.meta["motors"]], dtype=np.float64)
        joints = len(self.q0)
        self.history = History(joints, self.meta["history"], reference_pelvis_ori=False)
        self.proprio_dim = (3 * joints + 6) * self.meta["history"]
        with torch.inference_mode():
            probe = self.actor(torch.zeros(1, self.proprio_dim), torch.zeros(1, 3))
        if tuple(probe.shape) != (1, joints):
            raise ValueError(
                f"Velocity actor returned {tuple(probe.shape)}; expected (1, {joints}) "
                f"from history {self.proprio_dim} and command 3"
            )
        self.dt = self.meta["control_dt"]
        self.decimation = self.meta["decimation"]
        if not np.isclose(self.dt, self.model.opt.timestep * self.decimation):
            raise ValueError("Control timing does not match physics decimation")
        self.root_id = self.model.body(self.meta["root_body_name"]).id
        self.previous_action = np.zeros(joints, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float64)
        self.samples = []

    def reset(self, command):
        import mujoco

        self.command = clamp_command(command, self.limits)
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey < 1:
            raise ValueError("Scene has no standing keyframe")
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = self.q0[self.actuator_to_joint]
        mujoco.mj_forward(self.model, self.data)
        self.previous_action[:] = 0
        self.history.reset(self._observation())
        self.stand_height = float(self.data.xpos[self.root_id, 2])
        self.steps = 0
        self.failed = False
        self.failure = None
        self.samples = []

    def _observation(self):
        return observation(self.data.qpos, self.data.qvel, self.previous_action, self.q0)

    def _tilt(self):
        return abs(float(rotation(self.data.xquat[self.root_id])[2, 2]) - 1.0)

    def step(self, command=None):
        import mujoco

        if self.failed:
            raise RuntimeError("Robot has fallen; reset before stepping again")
        if command is not None:
            self.command = clamp_command(command, self.limits)
        obs = self.history.packed()
        if obs.shape != (self.proprio_dim,):
            raise ValueError(f"Proprio history is {obs.shape}, expected ({self.proprio_dim},)")
        cmd = self.command.astype(np.float32)
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(obs[None]), torch.from_numpy(cmd[None]))[0].numpy()
        if np.shape(action) != (len(self.q0),) or not np.isfinite(action).all():
            raise FloatingPointError("Invalid policy action; refusing to advance simulation")
        targets = self.q0 + self.scale * action
        self.data.ctrl[:] = targets[self.actuator_to_joint]
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo state became nonfinite")
        mujoco.mj_forward(self.model, self.data)
        measured = body_velocity(self.data.qpos, self.data.qvel)
        height_drop = self.stand_height - float(self.data.xpos[self.root_id, 2])
        tilt = self._tilt()
        if height_drop > self.meta["height_threshold"]:
            self.failed, self.failure = True, "height"
        elif tilt > self.meta["tilt_threshold"]:
            self.failed, self.failure = True, "tilt"
        self.steps += 1
        self.previous_action[:] = action
        self.history.append(self._observation())
        self.samples.append(
            np.concatenate((self.command, measured, [height_drop, tilt, np.max(np.abs(action))]))
        )
        return self.samples[-1]

    def report(self):
        values = np.asarray(self.samples, dtype=np.float64).reshape(-1, 9)
        mean = values.mean(0) if len(values) else np.full(9, np.nan)
        error = np.abs(values[:, 3:6] - values[:, :3]).mean(0) if len(values) else np.full(3, np.nan)
        root = self.data.qpos[:2]
        return {
            "mode": "velocity",
            "actor": str(self.actor_path),
            "bundle": str(self.bundle),
            "initial_command": values[0, :3].tolist() if len(values) else self.command.tolist(),
            "final_command": self.command.tolist(),
            "mean_command": mean[:3].tolist(),
            "mean_body_velocity": mean[3:6].tolist(),
            "mean_abs_velocity_error": error.tolist(),
            "command_names": ["vx_body_mps", "vy_body_mps", "wz_body_radps"],
            "command_limits": self.limits.tolist(),
            "steps": self.steps,
            "simulated_seconds": self.steps * self.dt,
            "failed": self.failed,
            "failure": self.failure,
            "stand_height_m": getattr(self, "stand_height", None),
            "max_height_drop_m": float(values[:, 6].max()) if len(values) else None,
            "max_tilt_from_upright": float(values[:, 7].max()) if len(values) else None,
            "final_root_xy_m": [float(root[0]), float(root[1])],
            "max_abs_action": float(values[:, 8].max()) if len(values) else None,
            "termination": (
                "Drop from the standing keyframe or tilt from upright. "
                "Not compared with a motion reference."
            ),
            "actuator_delay_physics_steps": 0,
            "physics": "native MuJoCo CPU",
            "actor_runtime": "TorchScript CPU",
        }


def _print_status(sim: VelocitySim):
    row = sim.samples[-1]
    print(
        f"[INFO] t={sim.steps * sim.dt:.2f}s  "
        f"command=[{row[0]:+.2f}, {row[1]:+.2f}, {row[2]:+.2f}]  "
        f"body=[{row[3]:+.2f}, {row[4]:+.2f}, {row[5]:+.2f}]  "
        f"drop={row[6]:.3f}m tilt={row[7]:.3f}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle",
        type=Path,
        default=Path("outputs/casbot02_seed_motion_velocity_muon_sim2sim"),
    )
    p.add_argument(
        "--actor",
        type=Path,
        default=Path("outputs/casbot02_seed_motion_velocity_muon_export/velocity_actor.pt"),
    )
    p.add_argument("--vx", type=float, default=0.0, help="Forward speed in the root frame, m/s")
    p.add_argument("--vy", type=float, default=0.0, help="Leftward speed in the root frame, m/s")
    p.add_argument("--wz", type=float, default=0.0, help="Yaw rate in the root frame, rad/s")
    p.add_argument("--seconds", type=float, help="Stop after this simulated duration")
    p.add_argument("--render", action="store_true", help="Native MuJoCo viewer")
    p.add_argument("--no-realtime", action="store_true")
    p.add_argument("--report", type=Path)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        p.error("--seconds must be positive and finite")
    if not args.render and args.seconds is None:
        args.seconds = 8.0
    sim = VelocitySim(args.bundle, args.actor, args.threads)
    initial = [args.vx, args.vy, args.wz]
    sim.reset(initial)
    print(
        f"[INFO] velocity | {len(sim.q0)} joints | {1 / sim.dt:g} Hz policy / "
        f"{1 / sim.model.opt.timestep:g} Hz physics"
    )
    print(
        "[INFO] command "
        f"[{sim.command[0]:+.2f}, {sim.command[1]:+.2f}, {sim.command[2]:+.2f}] "
        "m/s, m/s, rad/s in the root frame. No motion reference."
    )
    controls = {
        "paused": False,
        "reset": False,
        "command": sim.command.copy(),
        "dirty": False,
    }
    lock = threading.Lock()
    step = np.array([0.1, 0.1, 0.2])

    def nudge(index, sign):
        with lock:
            controls["command"][index] += sign * step[index]
            controls["command"] = clamp_command(controls["command"], sim.limits)
            controls["dirty"] = True

    def key(code):
        if code in (87, 119):
            nudge(0, +1)
        elif code in (83, 115):
            nudge(0, -1)
        elif code in (65, 97):
            nudge(1, +1)
        elif code in (68, 100):
            nudge(1, -1)
        elif code in (81, 113):
            nudge(2, +1)
        elif code in (69, 101):
            nudge(2, -1)
        elif code in (88, 120):
            with lock:
                controls["command"][:] = 0
                controls["dirty"] = True
        elif code == 32:
            with lock:
                controls["paused"] = not controls["paused"]
        elif code in (82, 114):
            with lock:
                controls["reset"] = True

    viewer_context = contextlib.nullcontext(None)
    if args.render:
        import mujoco

        sim.model.geom_group[sim.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE] = 2
        viewer_context = native_viewer(sim.model, sim.data, key)
        print("[INFO] W/S forward | A/D left/right | Q/E yaw | X zero | Space pause | R reset.")
    with viewer_context as viewer:
        if viewer:
            viewer.opt.geomgroup[0] = 0
            viewer.opt.geomgroup[1] = 1
            viewer.opt.geomgroup[2] = 1
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -12
            viewer.cam.azimuth = 120
        while True:
            start = time.perf_counter()
            if viewer and not viewer.is_running():
                break
            with lock:
                paused = controls["paused"]
                reset = controls["reset"]
                command = controls["command"].copy()
                dirty = controls["dirty"]
                controls["reset"] = False
                controls["dirty"] = False
            if reset:
                sim.reset(command)
                print("[INFO] reset to the standing keyframe")
            elif dirty:
                text = ", ".join(f"{v:+.2f}" for v in command)
                print(f"[INFO] command [{text}]")
            if not paused and not sim.failed:
                sim.step(command)
                if sim.steps % max(1, round(1 / sim.dt)) == 0 or sim.failed:
                    _print_status(sim)
            if viewer and not sim.failed:
                with viewer.lock():
                    viewer.cam.lookat[:] = sim.data.xpos[sim.root_id]
                viewer.sync()
            limited = args.seconds is not None and sim.steps >= round(args.seconds / sim.dt)
            if sim.failed or limited:
                break
            if viewer and (paused or not args.no_realtime):
                time.sleep(max(0, sim.dt - (time.perf_counter() - start)))
    result = sim.report()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

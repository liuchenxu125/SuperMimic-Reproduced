"""Standalone SuperMimic actor in native MuJoCo (no mjlab or training imports).

This file is also copied into exported bundles and runs there by itself.
Only NumPy, PyTorch and MuJoCo are needed. State and quaternion conventions
are the SuperMimic contract, not the different ProtoMotions ONNX contract.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch


def rotation(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> rotation from local frame to world, including batches."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-10):
        raise ValueError("Invalid wxyz quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def yaw(q: np.ndarray) -> np.ndarray:
    r = rotation(q)
    return np.arctan2(r[..., 1, 0], r[..., 0, 0])


def observation(qpos, qvel, previous_action, q0) -> np.ndarray:
    """Single raw proprioception frame. Free-joint angular qvel is body-local."""
    n = len(q0)
    qp, qv = np.asarray(qpos), np.asarray(qvel)
    if qp.shape != (n + 7,) or qv.shape != (n + 6,) or np.shape(previous_action) != (n,):
        raise ValueError("State/action dimensions do not match the joint contract")
    gravity = -rotation(qp[3:7])[2, :]
    result = np.concatenate((qp[7:] - q0, qv[6:], qv[3:6], gravity, previous_action))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite robot observation")
    return result.astype(np.float32)


class History:
    def __init__(self, joints: int, length: int):
        if joints < 1 or length < 1:
            raise ValueError("History dimensions must be positive")
        self.sizes = (joints, joints, 3, 3, joints)
        self.frames = np.zeros((length, sum(self.sizes)), dtype=np.float32)

    def reset(self, frame):
        self.frames[:] = frame

    def append(self, frame):
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame

    def packed(self) -> np.ndarray:
        # Term-major, then chronological time, then joint/component.
        terms = np.split(self.frames, np.cumsum(self.sizes)[:-1], axis=1)
        return np.concatenate([x.reshape(-1) for x in terms]).astype(np.float32)


class Reference:
    def __init__(self, path: Path, joint_names: list[str], goal_frames: int, dt: float):
        with np.load(path, allow_pickle=False) as raw:
            self.joint_pos = raw["joint_pos"].astype(np.float64)
            self.joint_vel = raw["joint_vel"].astype(np.float64)
            self.root_pos = raw["root_pos_w"].astype(np.float64)
            self.root_quat = raw["root_quat_w"].astype(np.float64)
            self.root_lin = raw["root_lin_vel_w"].astype(np.float64)
            self.root_ang = raw["root_ang_vel_w"].astype(np.float64)
            self.features = raw["goal_features"].astype(np.float32)
            if raw["joint_names"].tolist() != joint_names:
                raise ValueError(f"{path}: reference joint order does not match actor")
            if not np.isclose(float(raw["fps"].item()), 1 / dt):
                raise ValueError(f"{path}: reference fps does not match control period")
        self.name, self.goal_frames = path.stem, goal_frames
        self.length = len(self.joint_pos)
        n = len(joint_names)
        for name, width in [
            ("joint_pos", n),
            ("joint_vel", n),
            ("root_pos", 3),
            ("root_quat", 4),
            ("root_lin", 3),
            ("root_ang", 3),
            ("features", 2 * n + 10),
        ]:
            a = getattr(self, name)
            if a.shape != (self.length, width) or not np.isfinite(a).all():
                raise ValueError(f"{path}: invalid {name}")
        if self.length < 2:
            raise ValueError("Reference must contain at least two frames")
        rotation(self.root_quat)  # Reject invalid orientations before simulation.

    def align(self, frame: int, start_xy=None, start_yaw=None):
        """A single SE(2) alignment at start; never follow the robot each step."""
        if not 0 <= frame < self.length:
            raise ValueError("start-frame is outside the reference")
        delta = 0.0 if start_yaw is None else float(start_yaw) - float(yaw(self.root_quat[frame]))
        dq = np.array([math.cos(delta / 2), 0, 0, math.sin(delta / 2)])
        r = rotation(dq)
        origin = self.root_pos[frame].copy()
        anchor = origin.copy()
        if start_xy is not None:
            anchor[:2] = start_xy
        self.root_pos = (self.root_pos - origin) @ r.T + anchor
        self.root_quat = quat_multiply(dq, self.root_quat)
        self.root_lin = self.root_lin @ r.T
        self.root_ang = self.root_ang @ r.T
        # Stored features are invariant under global XY/yaw transforms.

    def goal(self, frame: int, robot_pos, robot_quat, anchor_obs: bool) -> np.ndarray:
        idx = np.minimum(frame + np.arange(self.goal_frames), self.length - 1)
        result = self.features[idx].reshape(-1)
        if anchor_obs:
            r = rotation(robot_quat)
            pos = r.T @ (self.root_pos[frame] - robot_pos)
            ori = r.T @ rotation(self.root_quat[frame])
            result = np.concatenate((result, pos, ori[:, :2].reshape(-1)))
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite reference goal")
        return result.astype(np.float32)


def load_model(bundle: Path, metadata: dict):
    model = mujoco.MjModel.from_xml_path(str(bundle / metadata["model_file"]))
    for name, value in metadata["mujoco_options"].items():
        if isinstance(value, list):
            getattr(model.opt, name)[:] = value
        else:
            setattr(model.opt, name, value)
    for name in metadata["foot_geom_names"]:
        model.geom_friction[model.geom(name).id, 0] = 1.0
    names = metadata["joint_names"]
    if (model.nq, model.nv, model.nu) != (len(names) + 7, len(names) + 6, len(names)):
        raise ValueError("Model dimensions do not match the exported actor")
    actual = [model.joint(i).name for i in range(1, model.njnt)]
    if actual != metadata["model_joint_names"]:
        raise ValueError("MuJoCo joint order changed after export")
    jids = np.array([model.joint(n).id for n in actual])
    if not np.array_equal(model.jnt_qposadr[jids], np.arange(7, model.nq)):
        raise ValueError("Expected scalar joints immediately after the free joint")
    if not np.array_equal(model.jnt_dofadr[jids], np.arange(6, model.nv)):
        raise ValueError("Unexpected scalar joint velocity layout")
    if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("Only direct joint-transmission PD actuators are supported")
    to_joint = model.jnt_dofadr[model.actuator_trnid[:, 0]] - 6
    if sorted(to_joint.tolist()) != list(range(len(names))):
        raise ValueError("Actuators must map bijectively onto policy joints")
    if not np.allclose(model.actuator_gear, [1, 0, 0, 0, 0, 0]):
        raise ValueError("Expected unit actuator gearing")
    motors = metadata["motors"]
    for a, j in enumerate(to_joint):
        p = motors[j]
        if not np.isclose(model.actuator_gainprm[a, 0], p["kp"]):
            raise ValueError("Model PD gain does not match exported metadata")
        if not np.allclose(model.actuator_biasprm[a, :3], [0, -p["kp"], -p["kd"]]):
            raise ValueError("Expected implicit affine PD actuator")
        if not np.allclose(
            model.actuator_forcerange[a], [-p["effort_limit_nm"], p["effort_limit_nm"]]
        ):
            raise ValueError("Model actuator torque limits changed")
    return model, mujoco.MjData(model), to_joint


class Sim2Sim:
    def __init__(self, bundle: Path, threads: int = 2):
        self.bundle = Path(bundle).resolve()
        self.meta = json.loads((self.bundle / "manifest.json").read_text())
        if self.meta.get("format") != "supermimic-native-mujoco-v1":
            raise ValueError("Unsupported deployment bundle")
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.actor = torch.jit.load(
            str(self.bundle / self.meta["actor_file"]), map_location="cpu"
        ).eval()
        self.model, self.data, self.actuator_to_joint = load_model(self.bundle, self.meta)
        self.q0 = np.array([p["q0"] for p in self.meta["motors"]], dtype=np.float64)
        self.scale = np.array([p["scale"] for p in self.meta["motors"]], dtype=np.float64)
        self.history = History(len(self.q0), self.meta["history"])
        self.dt = self.meta["control_dt"]
        self.decimation = self.meta["decimation"]
        if not np.isclose(self.dt, self.model.opt.timestep * self.decimation):
            raise ValueError("Control timing does not match physics decimation")
        self.previous_action = np.zeros(len(self.q0), dtype=np.float32)
        self.root_id = self.model.body(self.meta["root_body_name"]).id

    def reset(self, clip: str, start_frame: int = 0, start_xy=None, start_yaw=None):
        if clip not in self.meta["motions"]:
            raise ValueError(f"Unknown clip {clip!r}; choices: {list(self.meta['motions'])}")
        self.reference = Reference(
            self.bundle / self.meta["motions"][clip],
            self.meta["joint_names"],
            self.meta["future_frames"],
            self.dt,
        )
        self.reference.align(start_frame, start_xy, start_yaw)
        ref = self.reference
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = ref.root_pos[start_frame]
        self.data.qpos[3:7] = ref.root_quat[start_frame]
        lo = np.array(self.meta["soft_joint_limits"])[:, 0]
        hi = np.array(self.meta["soft_joint_limits"])[:, 1]
        self.data.qpos[7:] = np.clip(ref.joint_pos[start_frame], lo, hi)
        self.data.qvel[:3] = ref.root_lin[start_frame]
        self.data.qvel[3:6] = rotation(ref.root_quat[start_frame]).T @ ref.root_ang[start_frame]
        self.data.qvel[6:] = ref.joint_vel[start_frame]
        self.data.ctrl[:] = self.q0[self.actuator_to_joint]
        mujoco.mj_forward(self.model, self.data)
        self.previous_action[:] = 0
        self.history.reset(
            observation(self.data.qpos, self.data.qvel, self.previous_action, self.q0)
        )
        self.start_frame, self.steps = start_frame, 0
        self.target_frame = min(start_frame + self.meta["reset_goal_advance"], ref.length - 1)
        self.finished = self.failed = self.completed = False
        self.metrics = []
        self.trajectory = []

    def inputs(self):
        return self.history.packed(), self.reference.goal(
            self.target_frame, self.data.qpos[:3], self.data.qpos[3:7], self.meta["anchor_obs"]
        )

    def position_targets(self, action):
        if np.shape(action) != (len(self.q0),) or not np.isfinite(action).all():
            raise FloatingPointError("Invalid policy action; refusing to advance simulation")
        return self.q0 + self.scale * action

    def step(self, record: bool = False):
        if self.finished:
            raise RuntimeError("Clip has finished; reset before stepping again")
        obs, goal = self.inputs()
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(obs[None]), torch.from_numpy(goal[None]))[
                0
            ].numpy()
        targets = self.position_targets(action)
        self.data.ctrl[:] = targets[self.actuator_to_joint]
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo state became nonfinite")
        ref, f = self.reference, self.target_frame
        # Training terminations read derived quantities just before its final
        # forward(), i.e. at the last substep's pre-integration kinematics.
        height_error = abs(self.data.xpos[self.root_id, 2] - ref.root_pos[f, 2])
        tilt_error = abs(
            rotation(self.data.xquat[self.root_id])[2, 2] - rotation(ref.root_quat[f])[2, 2]
        )
        self.failed = bool(
            height_error > self.meta["height_threshold"] or tilt_error > self.meta["tilt_threshold"]
        )
        mujoco.mj_forward(self.model, self.data)  # Fresh kinematics for input/visualization.
        self.steps += 1
        self.previous_action[:] = action
        root_error = np.linalg.norm(self.data.qpos[:3] - ref.root_pos[f])
        joint_rmse = np.sqrt(np.mean((self.data.qpos[7:] - ref.joint_pos[f]) ** 2))
        self.metrics.append(
            [height_error, tilt_error, root_error, joint_rmse, np.max(np.abs(action))]
        )
        if record:
            self.trajectory.append(
                np.concatenate((self.data.qpos, self.data.qvel, action, targets))
            )
        self.history.append(observation(self.data.qpos, self.data.qvel, action, self.q0))
        self.target_frame = min(f + 1, ref.length - 1)
        self.completed = self.steps >= ref.length - self.start_frame
        self.finished = self.failed or self.completed
        return self.metrics[-1]

    def report(self):
        values = np.asarray(self.metrics).reshape(-1, 5)
        mean = values.mean(0).tolist() if len(values) else [None] * 5
        return {
            "clip": self.reference.name,
            "start_frame": self.start_frame,
            "steps": self.steps,
            "simulated_seconds": self.steps * self.dt,
            "failed": self.failed,
            "completed_clip": bool(getattr(self, "completed", False)),
            "mean_height_error_m": mean[0],
            "mean_tilt_error": mean[1],
            "mean_root_position_error_m": mean[2],
            "mean_joint_rmse_rad": mean[3],
            "max_abs_action": float(values[:, 4].max()) if len(values) else None,
            "physics": "native MuJoCo CPU",
            "actor_runtime": "TorchScript CPU",
            "root_state_source": "MuJoCo ground truth; not a hardware state estimator",
            "reset_goal_advance_frames": self.meta["reset_goal_advance"],
        }


class ReferenceGhost:
    """Rendering-only reference robot; never added to the physical simulation."""

    def __init__(self, model):
        import copy

        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.model.geom_rgba[:] = [0.2, 0.65, 0.95, 0.3]
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[1] = 1
        self.options.sitegroup[:] = 0
        self.perturb = mujoco.MjvPerturb()

    def update(self, reference, frame, scene):
        self.data.qpos[:3] = reference.root_pos[frame]
        self.data.qpos[3:7] = reference.root_quat[frame]
        self.data.qpos[7:] = reference.joint_pos[frame]
        mujoco.mj_forward(self.model, self.data)
        scene.ngeom = 0
        mujoco.mjv_addGeoms(
            self.model,
            self.data,
            self.options,
            self.perturb,
            mujoco.mjtCatBit.mjCAT_DYNAMIC,
            scene,
        )


@contextlib.contextmanager
def native_viewer(model, data, key_callback):
    """Close and join MuJoCo's asynchronous render thread before Python exits.

    Handle.close() requests shutdown but does not wait. Exiting Python while
    the daemon render loop still uses GLFW can otherwise crash during atexit.
    Thread names here are Python's public names of MuJoCo's launch target.
    """
    import mujoco.viewer

    before = set(threading.enumerate())
    handle = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    render_threads = [
        t
        for t in threading.enumerate()
        if t not in before and t.name.endswith("(_launch_internal)")
    ]
    try:
        yield handle
    finally:
        handle.close()
        for thread in render_threads:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("MuJoCo viewer did not finish closing within 5 seconds")


def main():
    here = Path(__file__).resolve().parent
    default = here if (here / "manifest.json").exists() else Path("outputs/casbot02_sim2sim")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=default)
    p.add_argument("--clip", help="Clip name in the bundle; default is its first clip")
    p.add_argument("--list-clips", action="store_true")
    p.add_argument("--render", action="store_true", help="Native MuJoCo viewer")
    p.add_argument("--no-realtime", action="store_true")
    p.add_argument(
        "--seconds",
        type=float,
        help="Stop after this simulated duration; otherwise play the whole clip",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument(
        "--start-xy",
        type=float,
        nargs=2,
        help="Align reference and robot initial XY to this position",
    )
    p.add_argument(
        "--start-yaw",
        type=float,
        help="Initial heading in radians; rotates robot and reference together",
    )
    p.add_argument("--report", type=Path)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-ghost", action="store_true")
    a = p.parse_args()
    if a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds <= 0):
        p.error("--seconds must be positive and finite")
    sim = Sim2Sim(a.bundle, a.threads)
    clips = list(sim.meta["motions"])
    if a.list_clips:
        print("\n".join(clips))
        return
    index = clips.index(a.clip) if a.clip in clips else 0
    if a.clip is not None and a.clip not in clips:
        p.error(f"Unknown clip {a.clip!r}; choose from {clips}")
    sim.reset(clips[index], a.start_frame, a.start_xy, a.start_yaw)
    print(
        f"[INFO] {clips[index]} | {len(sim.q0)} joints | {1 / sim.dt:g} Hz policy / {1 / sim.model.opt.timestep:g} Hz physics"
    )
    print("[INFO] Native MuJoCo + exported actor; no mjlab, world model or GPU simulation.")
    controls = {"paused": False, "reset": False, "next": 0, "select": None}
    lock = threading.Lock()

    def key(code):
        with lock:
            if code == 32:
                controls["paused"] = not controls["paused"]
            elif code in (82, 114):
                controls["reset"] = True
            elif code in (78, 110):
                controls["next"] = 1
            elif code in (66, 98):
                controls["next"] = -1
            elif 49 <= code <= 57:
                controls["select"] = code - 49

    viewer_context = contextlib.nullcontext(None)
    if a.render:
        # Keep the floor visible while hiding collision meshes in group 0.
        sim.model.geom_group[sim.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE] = 2
        viewer_context = native_viewer(sim.model, sim.data, key)
        print(
            "[INFO] Space pause | R reset | N/B next/previous clip | 1–9 select (switching resets state)."
        )
    all_reports = []
    ghost = ReferenceGhost(sim.model) if a.render and not a.no_ghost else None
    with viewer_context as viewer:
        final_report_recorded = False
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
                switch = controls["next"] or controls["select"] is not None
                reset = controls["reset"]
                paused = controls["paused"]
                if switch or reset:
                    all_reports.append(sim.report())
                    index = (
                        (index + controls["next"]) % len(clips)
                        if controls["select"] is None
                        else min(controls["select"], len(clips) - 1)
                    )
                    sim.reset(clips[index], a.start_frame, a.start_xy, a.start_yaw)
                    controls.update(reset=False, next=0, select=None)
                    print(f"[INFO] reset -> {clips[index]}")
            if not paused:
                sim.step(record=a.trajectory is not None)
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = sim.data.xpos[sim.root_id]
                    if ghost is not None:
                        ghost.update(sim.reference, sim.target_frame, viewer.user_scn)
                viewer.sync()
            limited = a.seconds is not None and sim.steps >= round(a.seconds / sim.dt)
            if sim.finished or limited:
                all_reports.append(sim.report())
                final_report_recorded = True
                break
            if viewer and (paused or not a.no_realtime):
                time.sleep(max(0, sim.dt - (time.perf_counter() - start)))
    if not final_report_recorded:
        all_reports.append(sim.report())
    result = {"bundle": str(sim.bundle), "runs": all_reports}
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if a.trajectory:
        a.trajectory.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            a.trajectory,
            trajectory=np.asarray(sim.trajectory),
            dt=sim.dt,
            joint_names=np.array(sim.meta["joint_names"]),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

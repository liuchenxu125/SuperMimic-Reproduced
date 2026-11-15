#!/usr/bin/env python3
"""Independent live-reference MuJoCo viewer. It never opens ROS/motor topics."""

import argparse
import json
import socket
import time
from pathlib import Path

import mujoco
import numpy as np

from supermimic.deployment.sim2sim import Sim2Sim, rotation
from supermimic.teleop.reference import LatestWindow, active_sim_bundle, decode, load_contract
from supermimic.teleop.runtime import HoldOnTimeout, StreamingActor

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seconds", type=float, default=0.0, help="0 runs until interrupted")
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    bundle = (a.bundle or active_sim_bundle(ROOT)).resolve()
    meta = load_contract(bundle)
    sim = Sim2Sim(bundle, threads=1)
    actor = StreamingActor(sim.actor, meta)
    inbox = LatestWindow()
    hold = HoldOnTimeout(inbox)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((a.bind, a.port))
    sock.setblocking(False)
    viewer = None
    bad, accepted, steps, frames_hold = 0, 0, 0, 0
    start = due = None
    print(
        f"Waiting for matched references at udp://{a.bind}:{a.port}; checkpoint {meta['checkpoint_iteration']}; no motor publisher",
        flush=True,
    )

    def key(code):
        if code in (ord("r"), ord("R")):
            try:
                hold.resume()
                print("Reference resumed explicitly", flush=True)
            except ValueError as exc:
                print(exc, flush=True)

    try:
        while viewer is None or viewer.is_running():
            while True:
                try:
                    raw, _ = sock.recvfrom(8193)
                except BlockingIOError:
                    break
                try:
                    accepted += int(inbox.push(decode(raw, meta)))
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    bad += 1
                    if bad == 1:
                        print(f"Reference rejected: {exc}", flush=True)
            frames = hold.window()
            if frames is None:
                time.sleep(0.002)
                continue
            if start is None:
                f = frames[0]
                sim.data.qpos[:] = f[:34]
                sim.data.qvel[:3] = f[61:64]
                sim.data.qvel[3:6] = rotation(f[3:7]).T @ f[64:67]
                sim.data.qvel[6:] = f[34:61]
                mujoco.mj_forward(sim.model, sim.data)
                start = due = time.monotonic()
                if a.render:
                    from mujoco import viewer as mj_viewer

                    viewer = mj_viewer.launch_passive(sim.model, sim.data, key_callback=key)
                    viewer.cam.distance = 3.0
                    viewer.cam.elevation = -10
                print("Live policy active; R explicitly resumes after a stream timeout", flush=True)
            if time.monotonic() < due:
                time.sleep(min(0.002, due - time.monotonic()))
                continue
            _, target, _, _ = actor.act(sim.data.qpos, sim.data.qvel, frames)
            sim.data.ctrl[:] = target[sim.actuator_to_joint]
            for _ in range(sim.decimation):
                mujoco.mj_step(sim.model, sim.data)
            mujoco.mj_forward(sim.model, sim.data)
            steps += 1
            frames_hold += int(hold.latched or inbox.latest.mode == "HOLD")
            ref = actor.last_reference
            if (
                not np.isfinite(sim.data.qpos).all()
                or abs(sim.data.qpos[2] - ref[2]) > meta["height_threshold"]
                or abs(rotation(sim.data.qpos[3:7])[2, 2] - rotation(ref[3:7])[2, 2])
                > meta["tilt_threshold"]
            ):
                raise RuntimeError(
                    "Live sim tracking failed; simulator stopped, no robot commands were sent"
                )
            if viewer is not None:
                viewer.cam.lookat[:] = sim.data.qpos[:3]
                viewer.sync()
            due = max(due + 0.02, time.monotonic())
            if a.seconds and steps * 0.02 >= a.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if viewer is not None:
            viewer.close()
        result = {
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "steps": steps,
            "simulated_seconds": steps * 0.02,
            "reference_packets": accepted,
            "rejected_packets": bad,
            "held_steps": frames_hold,
            "hardware_commands_sent": False,
        }
        print(json.dumps(result, indent=2))
        if a.report:
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

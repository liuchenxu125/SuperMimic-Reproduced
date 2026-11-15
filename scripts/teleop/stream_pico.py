#!/usr/bin/env python3
"""PICO three-point or solved-body input -> calibrated IK -> 50Hz reference UDP.

No motor publisher or training imports. A calibrates/starts once, B pauses,
A resumes using the same calibration. Recalibrate by restarting the sender and
receiver while the robot is under its normal standing controller.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import time
from pathlib import Path

import numpy as np

from supermimic.teleop.pico import PicoSource, PicoUpperBodySource, Resampler
from supermimic.teleop.reference import PreviewBuilder, active_sim_bundle, encode, load_contract
from supermimic.teleop.retarget import CasbotRetargeter
from supermimic.teleop.upper_body import CasbotUpperBodyRetargeter

ROOT = Path(__file__).resolve().parents[2]


def stream_report(*, calibrated, active, last_received, now, frame, ik, sent_windows):
    """SDK polls between frames are idle, not a missing stream or lost calibration."""
    age = None if last_received is None else max(0.0, now - last_received)
    if age is None or age > 0.3:
        mode = "WAIT_INPUT" if not calibrated else "STREAM_GAP"
    elif not calibrated:
        mode = "WAIT_CALIBRATION"
    else:
        mode = "POSE" if active else "HOLD"
    return {
        "mode": mode,
        "calibrated": calibrated,
        "frame": frame,
        "source_age_ms": None if age is None else round(age * 1000, 1),
        "sent_windows": sent_windows,
        "ik": ik,
    }


def close_session(pico, sock, path, samples, timestamps, input_mode):
    """Commit the capture before SDK teardown; repeated Ctrl+C cannot discard it."""
    previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if path is not None and len(samples):
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".tmp.npz")
            payload = {
                "poses" if input_mode == "upper-body" else "body": samples,
                "input_mode": input_mode,
                "timestamp_ns": timestamps,
            }
            np.savez_compressed(temp, **payload)
            temp.replace(path)
            print(f"Recording saved: {path} ({len(samples)} samples)", flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        try:
            sock.close()
            if pico is not None:
                pico.close()
        except KeyboardInterrupt:
            print("SDK shutdown interrupted; exiting.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--source", choices=["pico", "recording", "bundle"], default="pico")
    parser.add_argument(
        "--input-mode",
        choices=["full-body", "upper-body"],
        default="full-body",
        help="upper-body reads head + two controllers, without Tracker; full-body preserves the existing input",
    )
    parser.add_argument(
        "--recording", type=Path, help="PICO NPZ: body[N,24,7] or poses[N,3,7], timestamp_ns[N]"
    )
    parser.add_argument("--clip", default="踏步", help="Headset-free reference transport test")
    parser.add_argument("--record", type=Path, help="Record raw PICO body samples for later replay")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--retargeter", choices=["simple", "gmr"], default="simple")
    parser.add_argument("--gmr-root", type=Path, help="GMR checkout (default ~/Desktop/GMR)")
    parser.add_argument("--gmr-config", type=Path, help="Full-body xrobot_to_casbot_casbot02.json")
    parser.add_argument("--human-height", type=float, help="Optional GMR human height in metres")
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=None,
        help="IK reference slew cap in rad/s (default: upper-body 2, full-body 6); does not change actuator limits",
    )
    parser.add_argument(
        "--max-hand-displacement",
        type=float,
        default=0.35,
        help="upper-body wrist reference workspace radius in metres",
    )
    args = parser.parse_args()
    upper = args.input_mode == "upper-body"
    if args.retargeter == "gmr" and upper:
        parser.error("GMR uses full-body 24-joint PICO tracking; select --input-mode full-body")
    if args.retargeter == "gmr" and args.scale is not None:
        parser.error("GMR uses its JSON human_scale_table; use --human-height instead of --scale")
    if args.retargeter != "gmr" and any(v is not None for v in (args.gmr_root, args.gmr_config, args.human_height)):
        parser.error("--gmr-root/--gmr-config/--human-height require --retargeter gmr")
    if args.max_joint_speed is None:
        args.max_joint_speed = 2.0 if upper else 6.0
    if not 0 < args.max_joint_speed <= 20:
        parser.error("Require 0 < --max-joint-speed <= 20")
    if upper and not 0.05 <= args.max_hand_displacement <= 0.8:
        parser.error("Require --max-hand-displacement in 0.05..0.8 metres")
    bundle = (args.bundle or active_sim_bundle(ROOT)).resolve()
    meta = load_contract(bundle)
    print(
        f"Reference only: {bundle}, checkpoint={meta['checkpoint_iteration']}, 930+690; input={args.input_mode}; real preview delay=180ms",
        flush=True,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = args.host, args.port
    preview = PreviewBuilder()
    if args.source == "bundle":
        with np.load(bundle / meta["motions"][args.clip], allow_pickle=False) as data:
            rows = np.c_[
                data["root_pos_w"],
                data["root_quat_w"],
                data["joint_pos"],
                data["joint_vel"],
                data["root_lin_vel_w"],
                data["root_ang_vel_w"],
            ]
        for f in range(len(rows)):
            indices = np.minimum(f + np.arange(10), len(rows) - 1)
            from supermimic.teleop.reference import Window

            frames = rows[indices]
            sock.sendto(encode(Window(preview.session, f, frames), meta), address)
            time.sleep(0.02)
        for f in range(100):
            from supermimic.teleop.reference import stationary

            sock.sendto(
                encode(
                    Window(preview.session, len(rows) + f, stationary(frames[-1]), "HOLD"), meta
                ),
                address,
            )
            time.sleep(0.02)
        return
    if args.source == "recording":
        if args.recording is None:
            parser.error("--recording is required")
        with np.load(args.recording, allow_pickle=False) as data:
            key = "poses" if upper else "body"
            if key not in data:
                parser.error(
                    f"Recording needs {key}; use the --input-mode matching its capture mode"
                )
            if "input_mode" in data and data["input_mode"].item() != args.input_mode:
                parser.error("Recording and --input-mode differ")
            body = data[key].copy()
            timestamps = data["timestamp_ns" if "timestamp_ns" in data else "t_ns"].copy()
        if body.shape[1:] != (3 if upper else 24, 7) or len(body) != len(timestamps):
            raise ValueError("Invalid PICO recording")
        source = iter(zip(timestamps, body))
        pico = None
    else:
        pico = PicoUpperBodySource() if upper else PicoSource()
    if args.retargeter == "gmr":
        from supermimic.teleop.gmr_retarget import CasbotGMRRetargeter

        retarget = CasbotGMRRetargeter(
            bundle, meta, gmr_root=args.gmr_root, config_path=args.gmr_config,
            human_height=args.human_height, max_joint_speed=args.max_joint_speed,
        )
        print(f"Actual GMR full-body: config={retarget.config_path}; "
              f"config_sha256={retarget.config_sha256}; solver_sha256={retarget.source_sha256}", flush=True)
    else:
        retarget_cls = CasbotUpperBodyRetargeter if upper else CasbotRetargeter
        retarget = retarget_cls(
            bundle, meta, scale=args.scale, max_joint_speed=args.max_joint_speed,
            **({"max_hand_displacement": args.max_hand_displacement} if upper else {}),
        )
    resampler = Resampler(num_poses=3 if upper else 24)
    recorded, recorded_times = [], []
    active = False
    a_was = b_was = False
    last_sample = None
    last_received = None
    received_samples = sent_windows = 0
    started = time.monotonic()
    report_due = started
    replay_start = replay_stamp = None
    reported_samples = 0
    report_time = started

    def report(now):
        nonlocal report_due, report_time, reported_samples
        if now < report_due:
            return
        value = stream_report(
            calibrated=retarget.calibrated,
            active=active,
            last_received=last_received,
            now=now,
            frame=preview.index,
            ik=retarget.metrics,
            sent_windows=sent_windows,
        )
        elapsed = now - report_time
        value["input_hz"] = (
            round((received_samples - reported_samples) / elapsed, 1) if elapsed >= 0.2 else None
        )
        if value["mode"] == "WAIT_INPUT" and pico is not None:
            value["reason"] = pico.waiting_message()
        elif value["mode"] == "STREAM_GAP":
            value["reason"] = (
                "No fresh PICO samples; check headset/Send/network. Explicit resume may be required."
            )
        elif value["mode"] == "WAIT_CALIBRATION":
            value["reason"] = "Stand relaxed and press A to calibrate."
        print(json.dumps(value, ensure_ascii=False), flush=True)
        report_due, report_time, reported_samples = now + 2.0, now, received_samples

    print(
        "Stand relaxed, matching the robot default pose. PICO A: calibrate/start; B: hold; A: resume.",
        flush=True,
    )
    if upper:
        print(
            "Upper-body: head is the hand-position anchor; root/legs/waist references stay at standing q0. No full-body inference.",
            flush=True,
        )
    try:
        while True:
            if pico is not None:
                sample = pico.read()
                if sample is None:
                    report(time.monotonic())
                    continue
                timestamp, raw, a, b = sample
            else:
                try:
                    timestamp, raw = next(source)
                except StopIteration:
                    break
                if replay_start is None:
                    replay_start, replay_stamp = time.monotonic(), timestamp
                # Absolute playback deadlines avoid accumulating IK compute
                # time into every recorded frame's interval.
                due = replay_start + (timestamp - replay_stamp) * 1e-9
                while due > time.monotonic():
                    time.sleep(max(0.0, min(0.02, due - time.monotonic())))
                a, b = not retarget.calibrated, False
            last_received = time.monotonic()
            received_samples += 1
            if args.record:
                recorded.append(raw.copy())
                recorded_times.append(timestamp)
            if a and not a_was:
                if not retarget.calibrated:
                    try:
                        retarget.calibrate(raw)
                    except ValueError as exc:
                        if pico is None:
                            raise
                        a_was, b_was = a, b
                        print(
                            f"Calibration rejected: {exc}; adjust pose and press A again",
                            flush=True,
                        )
                        continue
                    print(
                        f"Calibrated: human scale={retarget.scale:.3f}; no motor commands sent",
                        flush=True,
                    )
                active = True
            if b and not b_was:
                active = False
                print("PICO pause: stationary reference; policy feedback stays active", flush=True)
            if (
                retarget.calibrated
                and sent_windows > 0
                and last_sample is not None
                and timestamp - last_sample > 200_000_000
            ):
                active = False
                print("PICO device gap >200ms: HOLD; press A to resume", flush=True)
            a_was, b_was = a, b
            last_sample = timestamp
            if not retarget.calibrated:
                report(time.monotonic())
                continue
            for frame in resampler.push(timestamp, raw):
                if active:
                    try:
                        qpos = retarget.retarget(frame)
                    except ValueError as exc:
                        active = False
                        print(
                            f"Retarget rejected: {exc}; HOLD, check tracking before resuming",
                            flush=True,
                        )
                        qpos = retarget.previous.copy()
                else:
                    qpos = retarget.previous.copy()
                win = preview.push(qpos, "POSE" if active else "HOLD")
                if win is not None:
                    if not active:
                        retarget.previous = win.frames[0, :34].copy()
                    sock.sendto(encode(win, meta), address)
                    sent_windows += 1
            report(time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        close_session(pico, sock, args.record, recorded, recorded_times, args.input_mode)


if __name__ == "__main__":
    main()

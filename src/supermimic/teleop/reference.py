"""Versioned 50 Hz references with ten real, buffered preview frames."""

from __future__ import annotations

import json
import struct
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MAGIC = b"SMTEL001"
FPS, PREVIEW, WIDTH = 50, 10, 67
MAX_PACKET = 8192


def active_sim_bundle(root: Path) -> Path:
    """Follow the installed actor, without exporting/replacing any weight."""
    import yaml

    config = root / "rl_controller/config/supermimic_walk1_subject2_proprio.yaml"
    cfg = yaml.safe_load(config.read_text())
    meta = json.loads((config.parent / cfg["bundle"] / "manifest.json").read_text())
    records = sorted(
        (root / "outputs/casbot02_deployment").glob("*/deployment.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in records:
        record = json.loads(path.read_text())
        if (
            record.get("applied")
            and record.get("checkpoint_sha256") == meta["checkpoint_sha256"]
            and record.get("actor_sha256") == meta["actor_sha256"]
        ):
            bundle = Path(record["sim_bundle"])
            if (bundle / "manifest.json").is_file():
                return bundle
    raise ValueError("No export matches the installed actor; provide --bundle explicitly")


def load_contract(bundle: Path) -> dict:
    meta = json.loads((bundle / "manifest.json").read_text())
    if (
        meta.get("robot"),
        meta.get("obs_dim"),
        meta.get("goal_dim"),
        meta.get("goal_encoding"),
        meta.get("control_dt"),
    ) != ("casbot02", 930, 690, "reference_heading_v1", 0.02):
        raise ValueError("Teleop requires Casbot02 930+690, reference_heading_v1, 50 Hz")
    if meta.get("anchor_obs") or not meta.get("anchor_ori_obs"):
        raise ValueError("Expected per-history-frame pelvis rot6d without robot position")
    if "joint_limits" not in meta:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(bundle / meta["model_file"]))
        meta["joint_limits"] = model.jnt_range[1:].tolist()
    return meta


def validate_frames(frames, meta):
    frames = np.asarray(frames, dtype=np.float64)
    if frames.shape != (PREVIEW, WIDTH) or not np.isfinite(frames).all():
        raise ValueError("Expected ten finite 67-wide reference frames")
    if not np.allclose(np.linalg.norm(frames[:, 3:7], axis=1), 1.0, atol=1e-3):
        raise ValueError("Reference quaternions must be unit wxyz")
    limits = np.asarray(
        meta["joint_limits"] if "joint_limits" in meta else meta["soft_joint_limits"]
    )
    # Retargeted reference joints are checked, not silently reordered or clipped.
    if np.any(frames[:, 7:34] < limits[:, 0] - 1e-4) or np.any(
        frames[:, 7:34] > limits[:, 1] + 1e-4
    ):
        raise ValueError("Reference joints exceed the model limits")
    if np.abs(frames[:, 34:61]).max() > 100 or np.abs(frames[:, 61:]).max() > 100:
        raise ValueError("Unreasonable reference velocities")
    return frames.copy()


@dataclass
class Window:
    session: int
    frame: int
    frames: np.ndarray
    mode: str = "POSE"


def encode(window: Window, meta: dict) -> bytes:
    frames = validate_frames(window.frames, meta)
    if window.mode not in ("POSE", "HOLD") or window.session < 0 or window.frame < 0:
        raise ValueError("Invalid teleop session, index or mode")
    header = json.dumps(
        {
            "session": window.session,
            "frame": window.frame,
            "mode": window.mode,
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "joint_names": meta["joint_names"],
            "fps": FPS,
        },
        separators=(",", ":"),
    ).encode()
    packet = MAGIC + struct.pack("<I", len(header)) + header + frames.astype("<f4").tobytes()
    if len(packet) > MAX_PACKET:
        raise ValueError("Reference packet too large")
    return packet


def decode(packet: bytes, meta: dict) -> Window:
    if not 12 <= len(packet) <= MAX_PACKET or packet[:8] != MAGIC:
        raise ValueError("Invalid SMTEL001 packet")
    (size,) = struct.unpack("<I", packet[8:12])
    if size > 2048 or len(packet) != 12 + size + PREVIEW * WIDTH * 4:
        raise ValueError("Truncated or oversized reference packet")
    header = json.loads(packet[12 : 12 + size])
    if not isinstance(header, dict):
        raise ValueError("Invalid reference header object")  # noqa: TRY004 -- malformed wire value
    if (
        header.get("checkpoint_sha256") != meta["checkpoint_sha256"]
        or header.get("joint_names") != meta["joint_names"]
        or header.get("fps") != FPS
    ):
        raise ValueError("Stream/model checkpoint, joint order or rate mismatch")
    session, frame, mode = header["session"], header["frame"], header["mode"]
    if (
        not isinstance(session, int)
        or not isinstance(frame, int)
        or not 0 <= session < 2**53
        or not 0 <= frame < 2**53
        or mode not in ("POSE", "HOLD")
    ):
        raise ValueError("Invalid reference header")
    frames = np.frombuffer(packet[12 + size :], dtype="<f4").reshape(PREVIEW, WIDTH)
    return Window(session, frame, validate_frames(frames, meta), mode)


def stationary(frame):
    held = np.repeat(np.asarray(frame)[None], PREVIEW, axis=0)
    held[:, 34:] = 0.0
    return held


class PreviewBuilder:
    """Nine-frame lookahead delay. Derivatives come from the same causal poses."""

    def __init__(self, session: int | None = None):
        self.session = int(time.time_ns() % (2**52)) if session is None else session
        self.rows = deque(maxlen=PREVIEW)
        self.index = -1
        self.previous = None

    def push(self, qpos, mode="POSE") -> Window | None:
        qpos = np.asarray(qpos, dtype=float)
        if mode not in ("POSE", "HOLD"):
            raise ValueError("Invalid preview mode")
        if qpos.shape != (34,) or not np.isfinite(qpos).all():
            raise ValueError("Expected finite free-root + 27 joints")
        norm = np.linalg.norm(qpos[3:7])
        if abs(norm - 1) > 1e-3:
            raise ValueError("Invalid pose quaternion")
        if mode == "HOLD" and self.rows:
            # Freeze the reference currently being executed, not the newest
            # pose 180ms ahead in the preview queue.
            qpos = self.rows[0][:34].copy()
        row = np.zeros(WIDTH)
        row[:34] = qpos
        if self.previous is not None and mode == "POSE":
            row[34:61] = (qpos[7:] - self.previous[7:]) * FPS
            row[61:64] = (qpos[:3] - self.previous[:3]) * FPS
            a, b = (Rotation.from_quat(q[[4, 5, 6, 3]]) for q in (qpos, self.previous))
            row[64:67] = (a * b.inv()).as_rotvec() * FPS
        self.previous = qpos.copy()
        self.index += 1
        self.rows.append(row)
        if mode == "HOLD":
            # Pause immediately, without draining a queued moving preview.
            self.rows.clear()
            for _ in range(PREVIEW):
                self.rows.append(row.copy())
        if len(self.rows) < PREVIEW:
            return None
        return Window(self.session, self.index, np.asarray(self.rows).copy(), mode)


class LatestWindow:
    """Keep only fresh, monotonically indexed windows; timeout latches a hold."""

    def __init__(self, timeout=0.3, clock=time.monotonic):
        self.timeout, self.clock = timeout, clock
        self.latest = None
        self.received = None

    def push(self, window):
        if self.latest is not None:
            if window.session != self.latest.session:
                raise ValueError("New calibration session; restart/re-arm the receiver")
            if window.frame <= self.latest.frame:
                return False
        self.latest, self.received = window, self.clock()
        return True

    def fresh(self):
        return self.received is not None and self.clock() - self.received <= self.timeout

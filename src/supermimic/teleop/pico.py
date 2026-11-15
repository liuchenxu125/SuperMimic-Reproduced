"""XRoboToolkit body input and 50 Hz resampling; SDK imported only in PicoSource."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

JOINTS = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)
UNITY_TO_ZUP = Rotation.from_matrix([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def zup_poses(body, num_poses):
    """Unity xyz/xyzw -> z-up positions/world rotations, matching GMR's xrobot input.

    Anatomical local-frame offsets are established by neutral-pose calibration;
    raw PICO rotations are never passed straight to the robot IMU convention.
    """
    body = np.asarray(body, dtype=float)
    if body.shape != (num_poses, 7) or not np.isfinite(body).all():
        raise ValueError(f"PICO must provide finite {num_poses}x7 poses")
    norm = np.linalg.norm(body[:, 3:], axis=1)
    if np.any(norm < 0.5) or np.any(norm > 1.5):
        raise ValueError("Invalid PICO body orientations")
    positions = UNITY_TO_ZUP.apply(body[:, :3])
    rotations = UNITY_TO_ZUP * Rotation.from_quat(body[:, 3:] / norm[:, None])
    return positions, rotations


def zup_body(body):
    return zup_poses(body, 24)


class Resampler:
    def __init__(self, fps=50, max_gap=0.2, num_poses=24):
        self.fps, self.max_gap = fps, max_gap
        self.num_poses = num_poses
        self.previous = None
        self.origin = None
        self.next_index = 0

    def push(self, timestamp_ns, body):
        body = np.asarray(body, dtype=float)
        zup_poses(body, self.num_poses)
        t = timestamp_ns * 1e-9
        if self.previous is not None and t <= self.previous[0]:
            return []
        if self.previous is None or t - self.previous[0] > self.max_gap:
            self.previous, self.origin, self.next_index = (t, body.copy()), t, 1
            return [body.copy()]
        before, a = self.previous
        b = body.copy()
        b[:, 3:] *= np.where(np.sum(a[:, 3:] * b[:, 3:], axis=1, keepdims=True) < 0, -1.0, 1.0)
        out = []
        while self.origin + self.next_index / self.fps <= t + 1e-10:
            target = self.origin + self.next_index / self.fps
            alpha = np.clip((target - before) / (t - before), 0, 1)
            frame = (1 - alpha) * a + alpha * b
            frame[:, 3:] /= np.linalg.norm(frame[:, 3:], axis=1, keepdims=True)
            out.append(frame)
            self.next_index += 1
        self.previous = t, b
        return out


class PicoSource:
    def __init__(self):
        try:
            import xrobotoolkit_sdk as sdk
        except ImportError as exc:
            raise RuntimeError(
                "XRoboToolkit Python SDK is missing. Install it in the independent teleop environment; see docs/PICO_TELEOP.md"
            ) from exc
        self.sdk = sdk
        sdk.init()
        self.last_timestamp = None

    def status(self):
        """Read link diagnostics even when full-body data is unavailable."""
        return {
            "tracking_timestamp_ns": int(self.sdk.get_time_stamp_ns()),
            "body_available": bool(self.sdk.is_body_data_available()),
            "body_timestamp_ns": int(self.sdk.get_body_timestamp_ns()),
            "A": bool(self.sdk.get_A_button()),
            "B": bool(self.sdk.get_B_button()),
        }

    def waiting_message(self):
        status = self.status()
        if status["tracking_timestamp_ns"] <= 0:
            return "未收到头显数据：请启动 PC Service、在头显打开 XRoboToolkit 并连接电脑 IP，开启 Send。USB 连接本身不会发送骨架。"
        if not status["body_available"] or status["body_timestamp_ns"] <= 0:
            return "已收到头显/手柄数据，但没有全身骨架：当前重定向需 Full Body Tracking 与 PICO Tracker。仅头显和两手柄不能启动此全身接口。"
        return "已收到骨架：站姿校准后按右手柄 A；头显客户端的 Switch w/ A Button 应关闭，Send 保持开启。"

    def read(self):
        if not self.sdk.is_body_data_available():
            time.sleep(0.001)
            return None
        # Head/controller updates may continue while body tracking is stale.
        # Only the body's own timestamp may refresh a robot reference.
        timestamp = int(self.sdk.get_body_timestamp_ns())
        if timestamp <= 0 or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            time.sleep(0.001)
            return None
        body = np.asarray(self.sdk.get_body_joints_pose(), dtype=float).reshape(24, 7)
        zup_body(body)
        self.last_timestamp = timestamp
        return timestamp, body, bool(self.sdk.get_A_button()), bool(self.sdk.get_B_button())

    def close(self):
        self.sdk.close()


class PicoUpperBodySource(PicoSource):
    """Actual head/left-controller/right-controller poses, without body tracking."""

    def __init__(self):
        super().__init__()
        self.last_invalid_reason = ""

    def waiting_message(self):
        if self.status()["tracking_timestamp_ns"] <= 0:
            return "未收到 PICO 数据：检查 PC Service、客户端连接和 Send。"
        if self.last_invalid_reason:
            return "头显/双手柄位姿无效：请同时开启 Head 和 Controller。" + self.last_invalid_reason
        return "三点输入已连接，无需 Tracker。站直、双手自然放在两侧，按 A 校准；B 暂停。"

    def read(self):
        timestamp = int(self.sdk.get_time_stamp_ns())
        if timestamp <= 0 or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            time.sleep(0.001)
            return None
        poses = np.asarray(
            [
                self.sdk.get_headset_pose(),
                self.sdk.get_left_controller_pose(),
                self.sdk.get_right_controller_pose(),
            ],
            dtype=float,
        )
        try:
            zup_poses(poses, 3)
        except ValueError as exc:
            self.last_invalid_reason = str(exc)
            time.sleep(0.001)
            return None
        self.last_invalid_reason = ""
        self.last_timestamp = timestamp
        return timestamp, poses, bool(self.sdk.get_A_button()), bool(self.sdk.get_B_button())

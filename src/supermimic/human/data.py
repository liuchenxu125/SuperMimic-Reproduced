"""Strict human/robot time pairs with lazy bounded CPU storage, separate from training data."""

from __future__ import annotations

import csv
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .constants import HUMAN_JOINTS, HUMAN_ROLL_OFFSET, HUMAN_YAW_OFFSET

REST = Rotation.from_euler("z", HUMAN_YAW_OFFSET) * Rotation.from_euler("x", HUMAN_ROLL_OFFSET)


@dataclass(frozen=True)
class Pair:
    name: str
    robot_npz: Path
    human_npz: Path
    robot_start_frame: int = 0
    human_start_frame: int = 0
    length: int | None = None


def read_pairs(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pairs, seen = [], set()
    for row in rows:
        if row.get("keep", "true").strip().lower() in ("0", "false", "no"):
            continue
        name = row["name"]
        if not name or name in seen:
            raise ValueError(f"Duplicate/empty pair name: {name!r}")
        seen.add(name)
        pair = Pair(
            name,
            (path.parent / row["robot_npz"]).resolve(),
            (path.parent / row["human_npz"]).resolve(),
            int(row.get("robot_start_frame") or 0),
            int(row.get("human_start_frame") or 0),
            int(row["length"]) if row.get("length") else None,
        )
        if min(pair.robot_start_frame, pair.human_start_frame) < 0 or (
            pair.length is not None and pair.length < 1
        ):
            raise ValueError(f"Invalid time alignment for {name}")
        if not pair.robot_npz.is_file() or not pair.human_npz.is_file():
            raise FileNotFoundError(f"Missing pair file: {name}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("No human/robot pairs; robot motion alone is not human training data")
    return pairs


def _scalar(data, key):
    return float(np.asarray(data[key]).reshape(-1)[0])


def _quats(q, name):
    q = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(np.abs(norms - 1) > 0.02):
        raise ValueError(f"Invalid unit wxyz quaternions: {name}")
    return q / norms


def load_pair(pair: Pair, meta):
    with np.load(pair.robot_npz, allow_pickle=False) as data:
        if not np.isclose(_scalar(data, "fps"), 50):
            raise ValueError(f"{pair.name}: robot fps must be 50")
        if tuple(data["joint_names"]) != tuple(meta["joint_names"]):
            raise ValueError(f"{pair.name}: robot joint order differs from the teacher")
        names = [str(n).removeprefix("robot/") for n in data["body_names"]]
        root = names.index(meta["root_body_name"].removeprefix("robot/"))
        root_pos = data["body_pos_w"][:, root]
        root_quat = _quats(data["body_quat_w"][:, root], pair.name)
        robot = np.concatenate(
            [
                root_pos,
                root_quat,
                data["joint_pos"],
                data["joint_vel"],
                data["body_lin_vel_w"][:, root],
                data["body_ang_vel_w"][:, root],
            ],
            axis=-1,
        ).astype(np.float32)
    with np.load(pair.human_npz, allow_pickle=False) as data:
        if not np.isclose(_scalar(data, "fps"), 50):
            raise ValueError(f"{pair.name}: human fps must be 50")
        if tuple(data["human_joint_names"]) != HUMAN_JOINTS:
            raise ValueError(f"{pair.name}: human joint order must be the 24 SMPL slots")
        pos = data["human_pos_w"].astype(np.float32)
        quat = _quats(data["human_quat_w"], pair.name)
    if pos.ndim != 3 or pos.shape[1:] != (24, 3) or quat.shape != (*pos.shape[:2], 4):
        raise ValueError(f"{pair.name}: human poses must be [T,24,3/4]")
    if (
        robot.ndim != 2
        or robot.shape[1] != 67
        or not np.isfinite(robot).all()
        or not np.isfinite(pos).all()
    ):
        raise ValueError(f"{pair.name}: nonfinite/invalid robot or human motion")
    rn, hn = len(robot) - pair.robot_start_frame, len(pos) - pair.human_start_frame
    if rn <= 0 or hn <= 0:
        raise ValueError(f"{pair.name}: start offset is outside the clip")
    if pair.length is None:
        if rn != hn:
            raise ValueError(
                f"{pair.name}: frame mismatch robot={rn}, human={hn}; supply verified start offsets/length, no automatic truncation"
            )
        length = rn
    else:
        length = pair.length
        if length > min(rn, hn):
            raise ValueError(f"{pair.name}: requested pair length exceeds source")
    robot = robot[pair.robot_start_frame : pair.robot_start_frame + length].copy()
    pos = pos[pair.human_start_frame : pair.human_start_frame + length].copy()
    quat = quat[pair.human_start_frame : pair.human_start_frame + length].copy()
    # Retargeters may recenter/turn each robot clip. Match only the fixed start
    # heading; preserve the human's root height, proportions and temporal motion.
    human_heading = (Rotation.from_quat(quat[0, 0], scalar_first=True) * REST.inv()).as_matrix()
    robot_heading = Rotation.from_quat(robot[0, 3:7], scalar_first=True).as_matrix()
    yaw = np.arctan2(robot_heading[1, 0], robot_heading[0, 0]) - np.arctan2(
        human_heading[1, 0], human_heading[0, 0]
    )
    turn = Rotation.from_euler("z", yaw)
    pos = turn.apply(pos.reshape(-1, 3)).reshape(pos.shape)
    quat = (
        (turn * Rotation.from_quat(quat.reshape(-1, 4), scalar_first=True))
        .as_quat(scalar_first=True)
        .reshape(quat.shape)
    )
    return robot, np.concatenate([pos, quat], axis=-1).astype(np.float32)


class PairedLibrary:
    def __init__(self, pairs, meta, window, cache_clips=4):
        if window < 10 or cache_clips < 0:
            raise ValueError("Invalid paired-library window/cache size")
        self.pairs, self.meta, self.window = pairs, meta, window
        self.cache_clips, self.cache = cache_clips, OrderedDict()
        self.lengths = []
        for i in range(len(pairs)):
            robot, _ = self.get(i)
            if len(robot) < window:
                raise ValueError(f"{pairs[i].name}: {len(robot)} frames < required window {window}")
            self.lengths.append(len(robot))
        self.weights = torch.tensor([n - window + 1 for n in self.lengths], dtype=torch.float)

    def get(self, i):
        if i not in self.cache:
            self.cache[i] = load_pair(self.pairs[i], self.meta)
            while self.cache_clips and len(self.cache) > self.cache_clips:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def sample(self, batch_size, generator, device):
        selected = torch.multinomial(
            self.weights, batch_size, replacement=True, generator=generator
        )
        starts = torch.rand(batch_size, generator=generator)
        robot = np.empty((batch_size, self.window, 67), dtype=np.float32)
        human = np.empty((batch_size, self.window, 24, 7), dtype=np.float32)
        ids, uniforms = selected.numpy(), starts.numpy()
        # Group by clip, so a large batch never repeatedly decompresses the same
        # clip as the bounded LRU cache cycles between random row selections.
        for i in np.unique(ids):
            rows = np.flatnonzero(ids == i)
            start = (uniforms[rows] * (self.lengths[i] - self.window + 1)).astype(np.int64)
            indices = start[:, None] + np.arange(self.window)[None, :]
            r, h = self.get(i)
            robot[rows], human[rows] = r[indices], h[indices]
        return torch.from_numpy(robot).to(device), torch.from_numpy(human).to(device)


def split_pairs(pairs, validation_fraction):
    if not 0 <= validation_fraction < 0.5:
        raise ValueError("Require 0 <= validation_fraction < 0.5")
    groups = {}
    for pair in pairs:
        groups.setdefault(pair.name.removesuffix("_M"), []).append(pair)
    if not validation_fraction or len(groups) < 2:
        return pairs, []
    count = min(len(groups) - 1, max(1, int(len(groups) * validation_fraction)))
    order = sorted(groups, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    val = set(order[:count])
    return (
        [p for p in pairs if p.name.removesuffix("_M") not in val],
        [p for p in pairs if p.name.removesuffix("_M") in val],
    )

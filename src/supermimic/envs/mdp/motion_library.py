"""Multi-clip motion library with GPU or CPU-backed compact reference storage.

All clips are concatenated along a single global frame axis.  A per-frame
``clip_id`` and per-clip ``start`` / ``length`` tables let a batch of
environments index arbitrary clips and frames with plain gathers.

Two storage modes:

* **full** (small sets, e.g. LAFAN1): per frame we keep the reference joints, the pelvis
  state and the world-frame link states of the *tracked* bodies. ~1 KB/frame.
* **compact** (BONES-SEED, 47.5M frames): per frame only the 71-d reference state
  ``[root_pos(3), root_quat(4), joint_pos(29), root_lin_vel_w(3), root_ang_vel_w(3),
  joint_vel(29)]`` in fp16/fp32 (284 B/frame in fp32); tracked-body states and goal
  features are computed on the fly with the differentiable FK (plan 1.3).

Goal features are assembled for a window, not stored per frame. Pelvis
displacement and orientation use the heading of the current reference frame.
Each frame then holds:
  [joint_pos - q0, joint_vel,
   pelvis displacement in that heading (3), pelvis rot6d in that heading (6),
   pelvis linear velocity in the pelvis frame (3),
   pelvis angular velocity in the pelvis frame (3)]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from supermimic.kinematics.rotations import (
    heading_matrix,
    matrix_from_quat,
    quat_apply_inverse,
)
from supermimic.robots import RobotLayout, get_robot, split_state

REF_DIM = 71


@dataclass
class ClipInfo:
    name: str
    start: int
    length: int

    @property
    def end(self) -> int:  # exclusive
        return self.start + self.length


@dataclass
class RefState:
    root_pos_w: torch.Tensor  # (N,3)
    root_quat_w: torch.Tensor  # (N,4) wxyz
    joint_pos: torch.Tensor  # (N,29)
    root_lin_vel_w: torch.Tensor  # (N,3)
    root_ang_vel_w: torch.Tensor  # (N,3)
    joint_vel: torch.Tensor  # (N,29)


@dataclass
class RefBodies:
    pos_w: torch.Tensor  # (N,B,3)
    quat_w: torch.Tensor  # (N,B,4)
    lin_vel_w: torch.Tensor  # (N,B,3)
    ang_vel_w: torch.Tensor  # (N,B,3)


_REF_FIELDS = (
    "root_pos_w",
    "root_quat_w",
    "joint_pos",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "joint_vel",
)


def _ref_from(parts: tuple[torch.Tensor, ...]) -> RefState:
    return RefState(*parts)


def _parts(state: RefState) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(state, name) for name in _REF_FIELDS)


def goal_features(
    anchor: RefState, future: RefState, default_joint_pos: torch.Tensor
) -> torch.Tensor:
    """Window features relative to the current reference root.

    ``anchor`` is that root, shape ``(N, ...)``. ``future`` holds the goal frames,
    shape ``(N, K, ...)``, including the current frame as the first one. Displacement
    and rot6d use only its heading. Linear and angular velocity use each future
    frame's own pelvis frame.
    """
    heading = heading_matrix(anchor.root_quat_w)
    heading_inverse = heading.transpose(-1, -2)[:, None]
    delta = future.root_pos_w - anchor.root_pos_w[:, None, :]
    displacement = torch.matmul(heading_inverse, delta.unsqueeze(-1)).squeeze(-1)
    relative = torch.matmul(heading_inverse, matrix_from_quat(future.root_quat_w))
    orientation = relative[..., :, :2].reshape(*relative.shape[:-2], 6)
    linear = quat_apply_inverse(future.root_quat_w, future.root_lin_vel_w)
    angular = quat_apply_inverse(future.root_quat_w, future.root_ang_vel_w)
    return torch.cat(
        [
            future.joint_pos - default_joint_pos,
            future.joint_vel,
            displacement,
            orientation,
            linear,
            angular,
        ],
        dim=-1,
    )


class MotionLibrary:
    def __init__(
        self,
        npz_files: list[str | Path],
        body_names: tuple[str, ...],
        default_joint_pos: torch.Tensor | np.ndarray,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        bin_frames: int = 50,
        compact: bool = False,
        store_dtype: torch.dtype = torch.float16,
        root_body_name: str = "pelvis",
        robot_name: str = "g1",
        joint_names: tuple[str, ...] | None = None,
        library_device: Literal["device", "cpu"] = "device",
        library_transfer: Literal["sync", "pinned"] = "pinned",
    ):
        if library_device not in ("device", "cpu"):
            raise ValueError(f"Unknown library_device: {library_device!r}")
        if library_transfer not in ("sync", "pinned"):
            raise ValueError(f"Unknown library_transfer: {library_transfer!r}")
        if library_device == "cpu" and not compact:
            raise ValueError("CPU reference storage requires compact_library")
        self.device = torch.device(device)
        self.storage_device = torch.device("cpu") if library_device == "cpu" else self.device
        self.library_device = library_device
        self.library_transfer = library_transfer
        self.dtype = dtype
        self.body_names = tuple(body_names)
        self.bin_frames = int(bin_frames)
        self.compact = compact
        if len(npz_files) == 0:
            raise ValueError("MotionLibrary needs at least one clip")
        if isinstance(default_joint_pos, torch.Tensor):
            q0 = default_joint_pos.detach().to(device=self.device, dtype=dtype).reshape(-1)
        else:
            q0 = torch.as_tensor(
                np.asarray(default_joint_pos), dtype=dtype, device=self.device
            ).reshape(-1)
        self.default_joint_pos = q0
        self.layout = RobotLayout(q0.numel())
        self.goal_feat_dim = self.layout.goal_dim

        clips: list[ClipInfo] = []
        start = 0
        body_idx = root_idx = None
        ref_chunks: list[np.ndarray] = []
        body_chunks = {"pos": [], "quat": [], "lin": [], "ang": []}
        # A single packed cache (build_library.py) can stand in for the npz list (compact mode only).
        if len(npz_files) == 1 and str(npz_files[0]).endswith(".pt"):
            assert compact, "packed .pt libraries are compact-only"
            pack = torch.load(npz_files[0], map_location="cpu", weights_only=False)
            self._ref = pack["ref"].to(device=self.storage_device, dtype=store_dtype)
            clips = [
                ClipInfo(n, int(st), int(ln))
                for n, st, ln in zip(pack["names"], pack["starts"], pack["lengths"])
            ]
            start = int(self._ref.shape[0])
            ref_chunks = None
        for file_index, f in enumerate(npz_files if ref_chunks is not None else [], 1):
            with np.load(f) as d:
                if d["joint_pos"].shape[-1] != self.layout.num_joints:
                    raise ValueError(f"{f}: joint count does not match {robot_name}")
                if joint_names is not None and tuple(d["joint_names"]) != tuple(joint_names):
                    raise ValueError(f"{f}: joint order does not match the simulator")
                if not np.isclose(float(d["fps"].reshape(-1)[0]), 50):
                    raise ValueError(f"{f}: convert motion to 50 Hz before training")
                if body_idx is None:
                    names = [str(n) for n in d["body_names"]]
                    body_idx = np.array([names.index(n) for n in self.body_names])
                    root_idx = names.index(root_body_name)
                n = int(d["joint_pos"].shape[0])
                bp, bq, blv, bav = (
                    d["body_pos_w"],
                    d["body_quat_w"],
                    d["body_lin_vel_w"],
                    d["body_ang_vel_w"],
                )
                ref = np.concatenate(
                    [
                        bp[:, root_idx],
                        bq[:, root_idx],
                        d["joint_pos"],
                        blv[:, root_idx],
                        bav[:, root_idx],
                        d["joint_vel"],
                    ],
                    axis=1,
                ).astype(np.float32)
                ref_chunks.append(ref)
                if not compact:
                    body_chunks["pos"].append(bp[:, body_idx])
                    body_chunks["quat"].append(bq[:, body_idx])
                    body_chunks["lin"].append(blv[:, body_idx])
                    body_chunks["ang"].append(bav[:, body_idx])
            clips.append(ClipInfo(Path(f).stem, start, n))
            start += n
            if len(npz_files) >= 1000 and (file_index % 5000 == 0 or file_index == len(npz_files)):
                print(f"[INFO] Loaded {file_index}/{len(npz_files)} reference clips; "
                      f"{start} frames", flush=True)
        self.clips = clips
        self.num_clips = len(clips)
        self.num_frames = start

        if ref_chunks is not None:
            ref_np = np.concatenate(ref_chunks, 0)
            assert ref_np.shape[1] == self.layout.state_dim
            self._ref = torch.as_tensor(
                ref_np, device=self.storage_device, dtype=store_dtype if compact else dtype
            )
            del ref_np, ref_chunks
        if not compact:
            t = lambda arrs: torch.as_tensor(
                np.concatenate(arrs, 0), dtype=dtype, device=self.device
            )
            self.body_pos_w = t(body_chunks["pos"])
            self.body_quat_w = t(body_chunks["quat"])
            self.body_lin_vel_w = t(body_chunks["lin"])
            self.body_ang_vel_w = t(body_chunks["ang"])
            del body_chunks
        else:
            from supermimic.kinematics.g1_fk import G1Kinematics

            self._fk = G1Kinematics(get_robot(robot_name).model(), device=self.device, dtype=dtype)
            self._fk_fn = self._fk
            self._fk_compiled_mode = None
            self._tracked_idx = self._fk.body_indices(self.body_names)

        self.clip_start = torch.tensor(
            [c.start for c in clips], device=self.device, dtype=torch.long
        )
        self.clip_len = torch.tensor(
            [c.length for c in clips], device=self.device, dtype=torch.long
        )
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(
            torch.arange(self.num_clips, device=self.device), self.clip_len
        )
        # Bins of `bin_frames` frames inside each clip (plan D14), numbered globally.
        bins_per_clip = (self.clip_len + self.bin_frames - 1) // self.bin_frames
        self.clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.num_bins = int(bins_per_clip.sum().item())
        bin_clip = torch.repeat_interleave(
            torch.arange(self.num_clips, device=self.device), bins_per_clip
        )
        local_bin = torch.arange(self.num_bins, device=self.device) - self.clip_bin_offset[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local_bin * self.bin_frames
        self.bin_end = torch.minimum(
            self.bin_start + self.bin_frames, self.clip_start[bin_clip] + self.clip_len[bin_clip]
        )
        self.bin_clip = bin_clip

    # ------------------------------------------------------------------ per-frame access

    @staticmethod
    def _split(r: torch.Tensor) -> RefState:
        return RefState(*split_state(r))

    def ref(self, frame: torch.Tensor) -> RefState:
        return self._split(self.ref71(frame))

    def ref71(self, frame: torch.Tensor) -> torch.Tensor:
        if self.storage_device.type == "cpu" and self.device.type != "cpu":
            # Copy only selected rows. Default to cpu4's pinned + asynchronous
            # path; blocking copies remain available via library_transfer="sync".
            indices = frame.detach().to(device="cpu")
            rows = self._ref[indices]
            asynchronous = self.library_transfer == "pinned" and self.device.type == "cuda"
            if asynchronous:
                rows = rows.pin_memory()
            return rows.to(device=self.device, dtype=self.dtype, non_blocking=asynchronous)
        return self._ref[frame].to(self.dtype)

    def bodies(self, frame: torch.Tensor) -> RefBodies:
        """Tracked-body link states (world) at the given frames."""
        if not self.compact:
            return RefBodies(
                self.body_pos_w[frame],
                self.body_quat_w[frame],
                self.body_lin_vel_w[frame],
                self.body_ang_vel_w[frame],
            )
        qpos, qvel = self.ref_qpos_qvel(frame)
        return self.bodies_from_qpos_qvel(qpos, qvel)

    # Convenience columns live on the storage device. Prefer ref(frame): it
    # returns only requested rows on the compute device for either backend.
    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref[:, 0:3].to(self.dtype)

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref[:, 3:7].to(self.dtype)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._split(self._ref).joint_pos.to(self.dtype)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._split(self._ref).root_lin_vel_w.to(self.dtype)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._split(self._ref).root_ang_vel_w.to(self.dtype)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._split(self._ref).joint_vel.to(self.dtype)

    # ------------------------------------------------------------------ queries

    def bin_of_frame(self, frame: torch.Tensor) -> torch.Tensor:
        clip = self.frame_clip[frame]
        return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return self.clip_last[self.frame_clip[frame]]

    def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
        """Goal for frames ``frame .. frame+k-1``, clamped at the clip end.

        Pelvis displacement and orientation are relative to the heading of
        ``frame``. The result is ``(N, k * goal_dim)``.
        """
        if k < 1:
            raise ValueError("Goal window must contain at least one frame")
        last = self.last_frame_of(frame)
        offsets = torch.arange(k, device=self.device)
        index = torch.minimum(frame[:, None] + offsets, last[:, None])
        features = goal_features(self.ref(frame), self.ref(index), self.default_joint_pos)
        return features.reshape(frame.shape[0], k * self.goal_feat_dim)

    def ref_qpos_qvel(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MuJoCo-layout reference state: qpos (N,36) and qvel (N,35; root ang vel in body frame)."""
        return self.qpos_qvel_from_ref(self.ref(frame))

    def qpos_qvel_from_ref(self, state: RefState) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = torch.cat([state.root_pos_w, state.root_quat_w, state.joint_pos], dim=-1)
        ang_b = quat_apply_inverse(state.root_quat_w, state.root_ang_vel_w)
        qvel = torch.cat([state.root_lin_vel_w, ang_b, state.joint_vel], dim=-1)
        return qpos, qvel

    def bodies_from_qpos_qvel(self, qpos: torch.Tensor, qvel: torch.Tensor) -> RefBodies:
        if not self.compact:
            raise RuntimeError("Body states are stored explicitly when compact_library is off")
        with torch.no_grad():
            out = self._fk_fn(qpos, qvel)
        i = self._tracked_idx
        return RefBodies(
            out["pos"][..., i, :],
            out["quat"][..., i, :],
            out["lin_vel"][..., i, :],
            out["ang_vel"][..., i, :],
        )

    def bodies_from_ref(self, state: RefState) -> RefBodies:
        """FK tracked bodies from an already gathered reference, without another library read."""
        return self.bodies_from_qpos_qvel(*self.qpos_qvel_from_ref(state))

    def compile_reference_fk(self, mode: str | None) -> None:
        """Compile the reference FK used by compact libraries. Eager when mode is unset."""
        if not self.compact or not mode or self._fk_compiled_mode == mode:
            return
        compiled_mode = None if mode == "default" else mode
        self._fk_fn = torch.compile(self._fk.forward, mode=compiled_mode, dynamic=False)
        self._fk_compiled_mode = mode

    def reference_span(self, frame: torch.Tensor, length: int) -> RefState:
        """States at ``frame + 0 .. length-1``, clamped per clip. Shape ``(length, N)``."""
        if length < 1:
            raise ValueError("reference span must contain at least one frame")
        last = self.last_frame_of(frame)
        offsets = torch.arange(length, device=frame.device)
        index = torch.minimum(frame.unsqueeze(0) + offsets[:, None], last.unsqueeze(0))
        state = self.ref(index.reshape(-1))
        n = frame.shape[0]
        return _ref_from(tuple(part.reshape(length, n, *part.shape[1:]) for part in _parts(state)))

    def goals_from_span(self, span: RefState, horizon: int, future_frames: int) -> torch.Tensor:
        """Goal rows for each step of a ``reference_span`` of length ``horizon + future_frames``."""
        if future_frames < 1:
            raise ValueError("Goal window must contain at least one frame")
        n = span.root_pos_w.shape[1]
        rows = []
        for t in range(horizon + 1):
            anchor = _ref_from(tuple(part[t] for part in _parts(span)))
            future = _ref_from(
                tuple(part[t : t + future_frames].movedim(0, 1) for part in _parts(span))
            )
            rows.append(goal_features(anchor, future, self.default_joint_pos).reshape(n, -1))
        return torch.stack(rows)

    def summary(self) -> str:
        hours = self.num_frames / 50.0 / 3600.0
        mem = self._ref.numel() * self._ref.element_size() / 1e9
        if not self.compact:
            mem += (
                sum(
                    t.numel() * t.element_size()
                    for t in (
                        self.body_pos_w,
                        self.body_quat_w,
                        self.body_lin_vel_w,
                        self.body_ang_vel_w,
                    )
                )
                / 1e9
            )
        storage = str(self.storage_device)
        transfer = f"; batches -> {self.device} ({self.library_transfer})" if self.library_device == "cpu" else ""
        return f"MotionLibrary({'compact' if self.compact else 'full'}): {self.num_clips} clips, {self.num_frames} frames ({hours:.3f} h @50Hz), {self.num_bins} bins, {mem:.2f} GB on {storage}{transfer}"


def discover_npz(
    directory: str | Path,
    limit: int | None = None,
    names: list[str] | None = None,
    manifest: str | Path | None = None,
) -> list[Path]:
    """List npz clips; ``manifest`` (csv with name,keep columns) restricts to keep==1."""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if manifest is not None:
        import csv

        with open(manifest) as f:
            keep = {
                row["name"]
                for row in csv.DictReader(f)
                if row.get("keep", "1") in ("1", "True", "true")
            }
        files = [f for f in files if f.stem in keep]
    if names is not None:
        wanted = set(names)
        files = [f for f in files if f.stem in wanted]
    if limit is not None:
        files = files[:limit]
    return files

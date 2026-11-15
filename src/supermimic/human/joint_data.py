"""Time-aligned human windows indexed by the native replay's global frame IDs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import PairedLibrary, load_pair


def stage_robot_pairs(pairs, directory, meta):
    """Stage only training pairs; honor explicit, verified time offsets/lengths.

    Complete clips are symlinks. Crops are new NPZs, never edits to the source.
    The native library sorts these filenames; HumanWindows maps by name, not order.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pairs, 1):
        target = directory / f"{pair.name}.npz"
        if Path(pair.name).name != pair.name or pair.name in (".", ".."):
            raise ValueError(f"Unsafe pair name: {pair.name}")
        with np.load(pair.robot_npz, allow_pickle=False) as data:
            total = len(data["joint_pos"])
            length = pair.length if pair.length is not None else total-pair.robot_start_frame
            if length < 1 or pair.robot_start_frame+length > total:
                raise ValueError(f"Invalid verified robot crop: {pair.name}")
            if tuple(data["joint_names"]) != tuple(meta["joint_names"]):
                raise ValueError(f"Robot joint order changed: {pair.name}")
            if pair.robot_start_frame == 0 and length == total:
                if target.is_symlink() and target.resolve() == pair.robot_npz.resolve():
                    continue
                if target.exists() or target.is_symlink():
                    raise FileExistsError(target)
                target.symlink_to(pair.robot_npz.resolve())
            else:
                arrays = {key: data[key] for key in data.files}
                for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                            "body_lin_vel_w", "body_ang_vel_w"):
                    arrays[key] = arrays[key][pair.robot_start_frame:pair.robot_start_frame + length]
                if target.exists():
                    with np.load(target, allow_pickle=False) as existing:
                        if all(np.array_equal(existing[k], v) for k, v in arrays.items()):
                            continue
                    raise ValueError(f"Staged crop changed: {target}")
                temporary = target.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez(stream, **arrays)
                temporary.replace(target)
        if len(pairs)>=1000 and (index%5000==0 or index==len(pairs)):
            print(f"[JOINT] staged robot pairs {index}/{len(pairs)}",flush=True)
    expected = {p.name for p in pairs}
    if {p.stem for p in directory.glob("*.npz")} != expected:
        raise ValueError("Staging directory has unexpected clips; use a new joint run")


class HumanWindows:
    """Bounded CPU cache, grouped gather, clip-end clamping identical to motion."""

    def __init__(self, pairs, meta, library, cache_clips=0):
        self.library = library
        self._packed = None
        by_name = {p.name: i for i, p in enumerate(pairs)}
        self.pair_indices = []
        if cache_clips == 0:
            # One contiguous CPU human tensor in *native library order*. Unlike
            # the post-training cache, do not retain a duplicate robot state.
            # Global replay frame IDs can now gather directly, without looping
            # over hundreds of clips or reopening NPZs on every collection step.
            chunks = []
            for index, clip in enumerate(library.clips, 1):
                if clip.name not in by_name:
                    raise ValueError(f"No human pair for {clip.name}")
                _, poses = load_pair(pairs[by_name[clip.name]], meta)
                if len(poses) != clip.length:
                    raise ValueError(f"Human/motion frame count mismatch: {clip.name}")
                chunks.append(torch.from_numpy(poses))
                if len(library.clips)>=1000 and (index%5000==0 or index==len(library.clips)):
                    print(f"[JOINT] loaded human CPU pairs {index}/{len(library.clips)}",flush=True)
            self._packed = torch.cat(chunks)
            self.data = None
            return
        self.data = PairedLibrary(pairs, meta, window=10, cache_clips=cache_clips)
        for clip in library.clips:
            if clip.name not in by_name:
                raise ValueError(f"No human pair for motion library clip {clip.name}")
            index = by_name[clip.name]
            if self.data.lengths[index] != clip.length:
                raise ValueError(f"Human/motion frame count mismatch: {clip.name}")
            self.pair_indices.append(index)

    @torch.no_grad()
    def future(self, frame, length):
        if length < 1 or frame.ndim != 1:
            raise ValueError("Expected 1-D global frames and positive preview length")
        lib = self.library
        if self._packed is not None:
            indices = torch.minimum(frame[:, None] + torch.arange(length, device=frame.device),
                                    lib.last_frame_of(frame)[:, None]).detach().cpu()
            output = self._packed[indices]
            if frame.device.type == "cuda" and lib.library_transfer == "pinned":
                output = output.pin_memory()
            return output.to(frame.device, non_blocking=output.is_pinned())
        clip = lib.frame_clip[frame].detach().cpu()
        local = (frame - lib.clip_start[lib.frame_clip[frame]]).detach().cpu()
        output = torch.empty((len(frame), length, 24, 7), dtype=torch.float32)
        for c in clip.unique().tolist():
            rows = (clip == c).nonzero().flatten()
            _, poses = self.data.get(self.pair_indices[c])
            index = (local[rows, None] + torch.arange(length)).clamp_max(len(poses) - 1)
            output[rows] = torch.from_numpy(poses[index.numpy()])
        if frame.device.type == "cuda" and lib.library_transfer == "pinned":
            output = output.pin_memory()
        return output.to(frame.device, non_blocking=output.is_pinned())

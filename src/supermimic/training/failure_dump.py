"""Save the actual failing minibatch; never suppress or replace the training error."""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch


def dump_world_model_failure(directory, model, library, trainer_cfg, batch, error):
    """Best-effort diagnostics only. These files are not resumable checkpoints."""
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now(UTC).strftime("wm_failure_%Y%m%dT%H%M%S_%f")
        path = directory / f"{name}_{os.getpid()}.pt"
        inputs = {key: getattr(batch, key).detach().cpu() for key in batch.__dataclass_fields__}
        ranges = {}
        for key, value in inputs.items():
            finite = torch.isfinite(value)
            clean = value[finite]
            ranges[key] = {
                "shape": list(value.shape),
                "nonfinite": int((~finite).sum()),
                "max_abs_finite": float(clean.to(torch.float64).abs().max()) if clean.numel() else None,
            }
        starts = [clip.start for clip in library.clips]
        frame_clips = []
        for frame in torch.unique(inputs["frame"]).tolist():
            index = bisect.bisect_right(starts, frame) - 1
            if index < 0 or frame >= library.clips[index].end:
                frame_clips.append({"global_frame": frame, "invalid": True})
            else:
                clip = library.clips[index]
                frame_clips.append({"global_frame": frame, "clip": clip.name,
                                    "local_frame": frame - clip.start})
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        report = {
            "diagnostic_only": True,
            "error": f"{type(error).__name__}: {error}",
            "inputs": ranges,
            "all_model_tensors_finite": all(bool(torch.isfinite(v).all()) for v in state.values()),
            "trainer_config": asdict(trainer_cfg),
            "model_config": asdict(model.cfg) if is_dataclass(model.cfg) else None,
            "frames": frame_clips,
        }
        torch.save({"report": report, "batch": inputs, "world_model": state}, path)
        path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[ERROR] World model failure batch and weights saved: {path}", flush=True)
    except Exception as dump_error:  # noqa: BLE001 - diagnostics must not mask the original error
        # A sticky CUDA failure can prevent even a diagnostic device-to-host copy.
        # Preserve the original exception rather than replacing it with this one.
        print(f"[ERROR] Could not save world model diagnostics: {dump_error}", flush=True)

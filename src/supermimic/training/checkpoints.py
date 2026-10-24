"""Interpret checkpoint counters before allocating a simulator or changing run files."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProgress:
    iteration: int = 0
    env_steps: int = 0
    policy_updates: int = 0
    collection_accumulator: float = 0.0
    train_seconds: float = 0.0
    migrated_legacy: bool = False


def checkpoint_progress(checkpoint: dict, *, allow_legacy: bool = False) -> TrainingProgress:
    unit = checkpoint.get("iteration_unit")
    if unit is None:
        if not allow_legacy:
            raise ValueError(
                "Legacy checkpoint counts iterations as env steps. Use a new run from scratch "
                "for paper-aligned training, or --legacy-resume with a NEW --run-name to "
                "explicitly migrate counters and adopt the new LR / sum-loss / precision settings."
            )
        if "num_updates" not in checkpoint:
            raise ValueError("Cannot migrate a legacy checkpoint without its recorded num_updates")
        outer = int(checkpoint["num_updates"])
        per_outer = checkpoint["config"]["schedule"].get("policy_updates_per_wm_update", 1)
        result = TrainingProgress(
            iteration=outer,
            env_steps=int(checkpoint["iteration"]),
            policy_updates=outer * int(per_outer),
            migrated_legacy=True,
        )
    elif unit == "update":
        result = TrainingProgress(
            iteration=int(checkpoint["iteration"]),
            env_steps=int(checkpoint["env_steps"]),
            policy_updates=int(checkpoint["num_updates"]),
            collection_accumulator=float(checkpoint["collection_accumulator"]),
            train_seconds=float(checkpoint.get("train_seconds", 0.0)),
        )
    else:
        raise ValueError(f"Unknown checkpoint iteration_unit: {unit!r}")
    if (
        min(result.iteration, result.env_steps, result.policy_updates) < 0
        or not 0 <= result.collection_accumulator < 1
        or not math.isfinite(result.train_seconds)
        or result.train_seconds < 0
    ):
        raise ValueError("Invalid checkpoint progress counters")
    return result

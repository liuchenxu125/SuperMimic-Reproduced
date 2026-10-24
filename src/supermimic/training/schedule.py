"""Paper Table A.5: update-count LR schedule and vectorised environment collection."""

from __future__ import annotations

import math


def cosine_lr(
    update: int, base_lr: float, final_ratio: float = 0.03, decay_iterations: int = 250_000
) -> float:
    """Decay over a fixed number of outer updates, then hold the final LR."""
    if update < 0 or decay_iterations <= 0 or not 0 <= final_ratio <= 1 or base_lr <= 0:
        raise ValueError("Invalid learning-rate schedule")
    phase = min(update / decay_iterations, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cosine)


def collection_steps(accumulator: float, updates_per_env_step: float) -> tuple[int, float]:
    """UTD=0.4 collects 2,3,2,3,... vectorised steps per outer update.

    The fractional accumulator is checkpointed so an odd update count does not
    change collection volume on resume. Replay refill is counted separately.
    """
    if not math.isfinite(updates_per_env_step) or updates_per_env_step <= 0:
        raise ValueError("updates_per_env_step must be finite and positive")
    if not math.isfinite(accumulator) or not 0 <= accumulator < 1:
        raise ValueError("Collection accumulator must be in [0,1)")
    total = accumulator + 1.0 / updates_per_env_step
    steps = math.floor(total + 1e-12)
    return steps, max(0.0, total - steps)

# Adapted from MorphMimic, Apache-2.0; see NOTICE.
from __future__ import annotations

import torch
from torch import Tensor, nn


class TermScales(nn.Module):
    """Running scale of each loss term `names`: an exponential average (`decay`) of the
    term's absolute value, started at the first value seen and floored at `floor`. A
    term not yet seen reads 1.0, so the first loss is the raw term. Buffers `scale` and
    `seen`, so the scales move with `.to` and save with `state_dict`."""

    def __init__(self, names: tuple[str, ...], decay: float = 0.99, floor: float = 1e-6) -> None:
        super().__init__()
        self.names = names
        self.decay = decay
        self.floor = floor
        self.register_buffer("scale", torch.zeros(len(names)))
        self.register_buffer("seen", torch.zeros(len(names), dtype=torch.bool))

    def scales(self) -> dict[str, Tensor]:
        """The current scale of every term (detached)."""
        value = torch.where(self.seen, self.scale.clamp(min=self.floor), 1.0)
        return {name: value[i] for i, name in enumerate(self.names)}

    @torch.no_grad()
    def update(self, terms: dict[str, Tensor]) -> None:
        """Folds the absolute value of each of `terms` into its scale. A non-finite value
        is skipped (it would poison the scale, and so every later loss)."""
        for name, term in terms.items():
            i = self.names.index(name)
            size = term.detach().abs().to(self.scale.dtype)
            mixed = self.decay * self.scale[i] + (1.0 - self.decay) * size
            finite = torch.isfinite(size)
            self.scale[i] = torch.where(
                finite, torch.where(self.seen[i], mixed, size), self.scale[i]
            )
            self.seen[i] = self.seen[i] | finite

"""Fail before an optimizer update can write nonfinite gradients into weights."""

from __future__ import annotations

import torch
import torch.distributed as dist

from supermimic.training.distributed import allreduce_grads, world_size


def require_finite(value: torch.Tensor, name: str) -> None:
    """All ranks stop together before backward if a loss/target is invalid."""
    finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if world_size() > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite):
        raise FloatingPointError(f"{name} contains NaN/Inf; stopped before optimizer.step()")


def backward_step(
    loss: torch.Tensor,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    max_norm: float,
    max_value: float | None = None,
) -> torch.Tensor:
    require_finite(loss, f"{name} loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    allreduce_grads(module)
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            module.parameters(), max_norm, error_if_nonfinite=True
        )
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        bad = [
            key
            for key, parameter in module.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if bad:
            detail = ", ".join(bad[:3])
            raise FloatingPointError(
                f"{name} gradient norm is NaN/Inf ({detail}); stopped before optimizer.step()"
            ) from error
        # A float32 sum of squares can overflow even when every gradient is
        # finite. Preserve the same global-norm clipping rule using float64,
        # including the multiplication, so a tiny coefficient cannot underflow
        # to zero in float32. This fallback runs only on norm overflow.
        gradients = [
            parameter.grad for parameter in module.parameters() if parameter.grad is not None
        ]
        norm = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(gradient, dtype=torch.float64) for gradient in gradients]
            )
        )
        require_finite(norm, f"{name} float64 gradient norm")
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        with torch.no_grad():
            for gradient in gradients:
                gradient.copy_((gradient.to(torch.float64) * coefficient).to(gradient.dtype))
    if max_value is not None:
        torch.nn.utils.clip_grad_value_(module.parameters(), max_value)
    optimizer.step()
    return norm

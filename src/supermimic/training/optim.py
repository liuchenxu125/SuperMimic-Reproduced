"""Adam and Muon construction for one captured training update."""

from __future__ import annotations

import torch
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz


def make_adam(params, *, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Same hyperparameters as ``torch.optim.Adam``.

    CUDA parameters use the fused capturable kernel. The learning rate is a
    device tensor so a captured update can change it with ``fill_``.
    """
    params = list(params)
    if params and all(parameter.device.type == "cuda" for parameter in params):
        lr = torch.tensor(float(lr), device=params[0].device)
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=True,
            capturable=True,
        )
    return torch.optim.Adam(
        params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )


def hidden_muon_weights(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Weights of every Linear between the first and last Linear of a Sequential."""
    chosen: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for seq in module.modules():
        if not isinstance(seq, torch.nn.Sequential):
            continue
        linears = [child for child in seq if isinstance(child, torch.nn.Linear)]
        for linear in linears[1:-1]:
            if id(linear.weight) in seen:
                continue
            seen.add(id(linear.weight))
            chosen.append(linear.weight)
    return chosen


def _muon_step(opt) -> None:
    """Muon update that stays inside a CUDA graph when the learning rate is a tensor.

    ``Muon.step`` applies the orthogonalized update with ``add_(alpha=lr)``.
    A tensor alpha is copied to the host, which invalidates graph capture.
    A Python float would stay at the value seen during capture, so the cosine
    schedule could not move it. This form matches ``Muon.step`` and reads the
    learning-rate tensor on replay.
    """
    with torch.no_grad():
        for group in opt.param_groups:
            lr = group["lr"]
            if not torch.is_tensor(lr):
                lr = torch.tensor(float(lr), device=group["params"][0].device)
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = opt.state[param]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if group["nesterov"] else buf
                update = _zeropower_via_newtonschulz(
                    update, group["ns_coefficients"], group["ns_steps"], group["eps"]
                )
                adjusted = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                param.mul_(1 - lr * weight_decay)
                param.add_(update.to(dtype=param.dtype) * (-adjusted))


class Optimizers:
    """Adam on biases, first and last linears, and Muon on the hidden linears."""

    def __init__(self, members: list):
        self.members = members

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.members:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.members:
            if isinstance(opt, torch.optim.Muon):
                _muon_step(opt)
            else:
                opt.step()

    @property
    def param_groups(self):
        return [group for opt in self.members for group in opt.param_groups]

    @property
    def state(self):
        merged = {}
        for opt in self.members:
            merged.update(opt.state)
        return merged

    def state_dict(self):
        return {type(opt).__name__.lower(): opt.state_dict() for opt in self.members}

    def load_state_dict(self, state) -> None:
        if "param_groups" in state:
            if any(isinstance(opt, torch.optim.Muon) for opt in self.members):
                raise ValueError(
                    "Checkpoint optimizer is Adam-only. Muon training needs a new run."
                )
            self.members[0].load_state_dict(state)
            return
        for opt in self.members:
            key = type(opt).__name__.lower()
            if key not in state:
                raise ValueError(f"Checkpoint is missing the {key} optimizer state")
            opt.load_state_dict(state[key])


def make_optimizers(module, *, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Split a module the way the official learner does.

    Hidden Linear weights use Muon with ``adjust_lr_fn="match_rms_adamw"``,
    weight decay 0.1, momentum 0.95 and Nesterov momentum. Everything else,
    including biases and the first and last Linear of each Sequential, keeps
    this function's Adam betas, epsilon and weight decay. A one-hidden-layer
    Sequential has no matrix between the first and last Linear, so it stays Adam.
    """
    params = list(module.parameters())
    hidden_ids = {id(param) for param in hidden_muon_weights(module)}
    hidden = [param for param in params if id(param) in hidden_ids]
    rest = [param for param in params if id(param) not in hidden_ids]
    if len(hidden) + len(rest) != len(params):
        raise RuntimeError("Muon split did not cover every parameter once")
    members = []
    if rest:
        members.append(
            make_adam(rest, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )
    if hidden:
        muon_lr = lr
        if hidden[0].device.type == "cuda":
            muon_lr = torch.tensor(float(lr), device=hidden[0].device)
        members.append(
            torch.optim.Muon(
                hidden,
                lr=muon_lr,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=True,
                adjust_lr_fn="match_rms_adamw",
            )
        )
    if not members:
        raise ValueError("make_optimizers received a module with no parameters")
    return Optimizers(members)


def parameter_counts(opt) -> tuple[int, int]:
    """Return ``(adam elements, muon elements)`` for a split or a plain Adam."""
    members = getattr(opt, "members", None)
    if members is None:
        count = sum(param.numel() for group in opt.param_groups for param in group["params"])
        return count, 0
    adam = muon = 0
    for member in members:
        count = sum(param.numel() for group in member.param_groups for param in group["params"])
        if isinstance(member, torch.optim.Muon):
            muon += count
        else:
            adam += count
    return adam, muon


def assign_lr(optimizer, lr: float) -> None:
    """Write ``lr`` into every group. Device learning rates stay at the same address."""
    for group in optimizer.param_groups:
        current = group["lr"]
        if torch.is_tensor(current):
            current.fill_(float(lr))
        else:
            group["lr"] = float(lr)

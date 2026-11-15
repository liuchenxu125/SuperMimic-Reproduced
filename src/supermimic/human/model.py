"""Human goal encoder, sharing an immutable native motion policy's decoder."""

from __future__ import annotations

import torch
from torch import nn

from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.dynamics import MLP

from .constants import HUMAN_FEAT_DIM, HUMAN_GOAL_DIM, HUMAN_HEADING_DIM
from .features import heading_features, human_features, mask_human_goal


class RunningNormalizer(nn.Module):
    """MorphMimic's single-process population moments (Apache-2.0, see NOTICE)."""

    def __init__(self, dim: int, eps: float = 0.01):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, x):
        if not self.training:
            return
        x = x.detach().reshape(-1, self.dim).to(self.mean.dtype)
        n = x.shape[0]
        if not n:
            return
        var, mean = torch.var_mean(x, dim=0, unbiased=False)
        self.count += n
        rate = (n / self.count.double()).to(self.mean.dtype)
        delta = mean - self.mean
        self.mean.add_(rate * delta)
        self.var.add_(rate * (var - self.var + delta * (mean - self.mean)))
        self.var.clamp_(min=0)

    def forward(self, x):
        return (x - self.mean) / (self.var.sqrt() + self.eps)


class HumanEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.mlp = MLP(HUMAN_GOAL_DIM, cfg.encoder_hidden, cfg.latent_dim)
        self.feature_norm = RunningNormalizer(HUMAN_FEAT_DIM)
        self.heading_norm = RunningNormalizer(HUMAN_HEADING_DIM)

    def pack(self, pos, quat, robot_anchor_quat, points, frames, heading, *, update=False):
        feats = human_features(pos, quat, points)
        rel = heading_features(robot_anchor_quat, quat, points)
        if update:
            self.feature_norm.update(feats)
            self.heading_norm.update(rel)
        return mask_human_goal(
            self.feature_norm(feats), points, frames, self.heading_norm(rel), heading
        )

    def forward(self, packed):
        return torch.tanh(self.mlp(packed))


class HumanPolicy(nn.Module):
    """Additional interface; `motion` remains exactly the frozen teacher path.

    Proprio history is supplied in the native term-major 870/930 layout. Human
    targets are separate 2254-wide goals, not an extra proprio observation term.
    """

    def __init__(self, teacher: Policy, human: HumanEncoder):
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.human = human

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, obs_hist, goal, interface="human"):
        if interface == "motion":
            return self.teacher(obs_hist, goal)
        if interface != "human":
            raise ValueError(f"Unknown interface: {interface}")
        return self.teacher.decoder(self.human(goal), obs_hist)

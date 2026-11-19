"""Three-value encoder sharing the native decoder, without reference leakage."""

import torch
from torch import nn

from supermimic.kinematics.rotations import cached_constant
from supermimic.robots.layout import reference_pelvis_ori_enabled
from supermimic.world_model.dynamics import MLP


class CommandNormalizer(nn.Module):
    """Population moments updated on real velocity-slot collection only."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("var", torch.ones(3))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, command):
        n = len(command)
        if not n:
            return
        var, mean = torch.var_mean(command, dim=0, unbiased=False)
        self.count += n
        ratio = (n / self.count.double()).to(command.dtype)
        delta = mean - self.mean
        self.mean.add_(ratio * delta)
        self.var.add_(ratio * (var - self.var + delta * (mean - self.mean))).clamp_(min=0)

    def forward(self, command):
        return (command - self.mean) / (self.var.sqrt() + 0.01)


class VelocityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.normalizer = CommandNormalizer()
        self.mlp = MLP(3, cfg.encoder_hidden, cfg.latent_dim)

    def forward(self, command):
        return torch.tanh(self.mlp(self.normalizer(command)))


def velocity_history(history, cfg):
    """Keep native motion input intact; velocity gets proprio + constant padding.

    The extra term-major 6D reference orientation is unavailable under a pure
    velocity command. Replace it with identity, not with a hidden motion target.
    A deployed velocity actor only needs the preceding 870 proprio values.
    """
    if not reference_pelvis_ori_enabled(cfg.obs_term_scale):
        return history
    proprio = cfg.history * (3 * cfg.num_joints + 6)
    identity = cached_constant((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), history).repeat(cfg.history)
    return torch.cat([history[..., :proprio], identity.expand(*history.shape[:-1], -1)], -1)


def mixed_actions(policy, history, motion_goal, command, motion_count):
    z_motion = policy.encoders["motion"](motion_goal[:motion_count])
    z_velocity = policy.encoders["velocity"](command[motion_count:])
    obs = torch.cat([history[:motion_count], velocity_history(history[motion_count:], policy.cfg)])
    return policy.decoder(torch.cat([z_motion, z_velocity]), obs)


def latent_bc(policy, command, motion_goal):
    """BC only trains the velocity encoder; not motion, decoder or goal/state."""
    with torch.no_grad():
        teacher = policy.encoders["motion"](motion_goal.detach())
    z = policy.encoders["velocity"](command.detach())
    return (z - teacher).square().mean(-1)


class VelocityActor(nn.Module):
    """Exported forward(proprio_history, [vx_b,vy_b,wz_b]) -> actions."""

    def __init__(self, policy):
        super().__init__()
        self.encoder, self.decoder = policy.encoders["velocity"], policy.decoder
        cfg = policy.cfg
        proprio = cfg.history * (3 * cfg.num_joints + 6)
        self.proprio_dim = proprio
        count = cfg.history * 6 if reference_pelvis_ori_enabled(cfg.obs_term_scale) else 0
        self.register_buffer(
            "padding", torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).repeat(count // 6)
        )

    def forward(self, history, command):
        obs = torch.cat([history, self.padding.expand(history.shape[0], -1)], -1)
        return self.decoder(self.encoder(command), obs)

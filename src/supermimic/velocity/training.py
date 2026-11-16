"""Native BPTT/critic with mixed motion/velocity slots and official latent BC.

Uses the original imagine(), rewards, termination, history, objective and GAE.
Only actor routing is changed; the original PolicyTrainer is not edited.
"""

import torch

from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.imagination.rollout import imagine
from supermimic.training.numerics import backward_step, require_finite

from .policy import latent_bc, mixed_actions


def motion_count(batch, fraction):
    if batch < 2:
        raise ValueError("Mixed batch needs at least two rows")
    return max(1, min(batch - 1, round(batch * (1 - fraction))))


class ImaginedActor:
    def __init__(self, policy, commands, cfg, start, horizon):
        self.policy, self.commands, self.cfg = policy, commands, cfg
        self.state, self.t = start.state, 0
        last = commands.lib.last_frame_of(start.frame)
        self.frames = torch.minimum(
            start.frame[None] + torch.arange(horizon, device=start.frame.device)[:, None],
            last[None],
        )
        self.world = commands.world_velocity(self.frames)
        self.count = motion_count(len(start.state), cfg.velocity_fraction)
        self.bc = []

    def __call__(self, history, goal):
        cmd = self.commands.project(self.world[self.t], self.state[:, 3:7])
        action = mixed_actions(self.policy, history, goal, cmd, self.count)
        self.bc.append(
            latent_bc(self.policy, cmd[self.count :], goal[self.count :]).sum() / len(cmd)
        )
        return action

    def step(self, native_step):
        def run(*args):
            result = native_step(*args)
            self.state, self.t = result[0], self.t + 1
            return result

        return run


class VelocityPolicyTrainer(PolicyTrainer):
    def __init__(self, wm, policy, critic, library, cfg, commands, velocity_cfg):
        super().__init__(wm, policy, critic, library, cfg)
        self.commands, self.velocity_cfg = commands, velocity_cfg
        self.latest = {}

    def _fit_critic(self, ro, feat, e_root, xi_norm):
        cfg = self.cfg
        with torch.no_grad():
            feat, goals, e_root = feat.detach(), ro.goals.detach(), e_root.detach()
            h1, n, _ = feat.shape
            xi = xi_norm[None].expand(h1, n, -1)
            values = self.critic(
                feat.reshape(h1 * n, -1),
                goals.reshape(h1 * n, -1),
                xi.reshape(h1 * n, -1),
                e_root.reshape(h1 * n, -1),
            ).reshape(h1, n)
            targets = gae_targets(
                ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda
            )
            inputs = [x[:-1].reshape(-1, x.shape[-1]) for x in (feat, goals, xi, e_root)]
            y = targets.reshape(-1)
        require_finite(y, "Mixed critic targets")
        size, mb = len(y), len(y) // cfg.critic_minibatches
        last = y.new_zeros(())
        max_norm = y.new_zeros((), dtype=torch.float64)
        for _ in range(cfg.critic_epochs):
            permutation = torch.randperm(size, device=y.device)
            for i in range(cfg.critic_minibatches):
                ids = permutation[i * mb : (i + 1) * mb]
                loss = (self.critic(*(x[ids] for x in inputs)) - y[ids]).square().mean()
                norm = backward_step(
                    loss,
                    self.critic,
                    self.critic_opt,
                    name="Mixed critic",
                    max_norm=cfg.critic_grad_clip,
                )
                last, max_norm = loss.detach(), torch.maximum(max_norm, norm.double())
        return {
            "critic_loss": float(last),
            "critic_grad_norm_max": float(max_norm),
            "critic_value_mean": float(values.mean()),
            "critic_target_mean": float(y.mean()),
        }

    def update(self, buffer, xi_norm_fn=None):
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)
        xi_norm = self.wm.spec.normalize_xi(start.xi)
        actor = ImaginedActor(self.policy, self.commands, self.velocity_cfg, start, cfg.horizon)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            ro = imagine(
                self.wm,
                actor,
                self.lib,
                start,
                cfg.horizon,
                cfg.future_frames,
                obs_noise=cfg.obs_noise,
                step_fn=actor.step(self.step_fn),
                anchor_obs=cfg.anchor_obs,
                anchor_ori_obs=cfg.anchor_ori_obs,
            )
            feat, goals, e_root = self._critic_inputs(ro)
            bootstrap = (
                self.critic(feat[-1], goals[-1], xi_norm, e_root[-1]) if cfg.use_critic else None
            )
            objective = policy_objective(ro, bootstrap, cfg.gamma)
            bc = torch.stack(actor.bc).sum()
            loss = -objective + self.velocity_cfg.latent_bc_weight * bc
            grad = backward_step(
                loss,
                self.policy,
                self.opt,
                name="Motion+velocity actor",
                max_norm=cfg.grad_clip_norm,
                max_value=cfg.grad_clip_value,
            )
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        count = actor.count
        result = {
            "imag_objective": float(objective.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "policy_grad_norm": float(grad),
            "velocity_latent_bc": float(bc.detach()),
            "motion_imag_return": float(ro.rewards[:, :count].sum(0).mean().detach()),
            "velocity_imag_return": float(ro.rewards[:, count:].sum(0).mean().detach()),
            "velocity_imag_survival": float(ro.survival[-1, count:].mean().detach()),
        }
        result.update({f"imag_r/{k}": v for k, v in ro.reward_terms.items()})
        if cfg.use_critic:
            result.update(self._fit_critic(ro, feat, e_root, xi_norm))
        self.latest = result
        return result

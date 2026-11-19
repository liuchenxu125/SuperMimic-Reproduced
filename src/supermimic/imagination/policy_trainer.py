"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.policy.nets import Critic, Policy
from supermimic.training.cuda_graph import CapturedUpdate, graph_state, optimize_step
from supermimic.training.distributed import world_size
from supermimic.training.optim import assign_lr, make_optimizers
from supermimic.world_model.world_model import WorldModel


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False
    anchor_ori_obs: bool = False


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        lib.compile_reference_fk(cfg.compile)
        self.opt = make_optimizers(policy, lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = make_optimizers(critic, lr=cfg.critic_lr)
        compiled_mode = None if cfg.compile in (None, "default") else cfg.compile
        self.compiled_policy = (
            torch.compile(policy, mode=compiled_mode, dynamic=False) if cfg.compile else policy
        )
        from supermimic.imagination.compiled import ImagStep

        self.step_fn = ImagStep(
            wm, lib.default_joint_pos, compile_mode=cfg.compile,
            reference_pelvis_ori=cfg.anchor_ori_obs and not cfg.anchor_obs,
        )
        self.capture = True
        self.latest: dict[str, float] = {}
        self._captured = CapturedUpdate(self._update_from, self._graph_state)

    def set_lr(self, lr: float) -> None:
        assign_lr(self.opt, lr)
        assign_lr(self.critic_opt, lr)

    def _graph_state(self) -> list[torch.Tensor]:
        state = graph_state([self.wm, self.policy, self.critic], [self.opt, self.critic_opt])
        ref = getattr(self.lib, "_ref", None)
        if torch.is_tensor(ref):
            state.append(ref)
        return state

    def _graph_enabled(self) -> bool:
        if not self.capture or world_size() > 1:
            return False
        params = list(self.policy.parameters())
        if not params or params[0].device.type != "cuda":
            return False
        storage = getattr(self.lib, "storage_device", None)
        return storage is None or torch.device(storage).type == "cuda"

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def _start_from(self, packed: dict[str, torch.Tensor]) -> StartBatch:
        return StartBatch(
            state=packed["state"],
            xi=packed["xi"],
            frame=packed["frame"],
            prev_action=packed["prev_action"],
            hist_states=packed["hist_states"],
            hist_actions=packed["hist_actions"],
            hist_frames=packed.get("hist_frames"),
        )

    def _update_from(self, packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        start = self._start_from(packed)
        xi_norm = self.wm.spec.normalize_xi(start.xi)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            ro = imagine(
                self.wm, self.compiled_policy, self.lib, start, cfg.horizon, cfg.future_frames,
                obs_noise=cfg.obs_noise, step_fn=self.step_fn,
                anchor_obs=cfg.anchor_obs, anchor_ori_obs=cfg.anchor_ori_obs,
            )
            feat, goals, e_root = self._critic_inputs(ro)
            bootstrap = (
                self.critic(feat[-1], goals[-1], xi_norm, e_root[-1]) if cfg.use_critic else None
            )
            objective = policy_objective(ro, bootstrap, cfg.gamma)
            # Eq. 2 already averages over rollouts; sum over time, no extra 1/H.
            loss = -objective
            grad, finite = optimize_step(
                self.opt, loss, self.policy,
                max_norm=cfg.grad_clip_norm, max_value=cfg.grad_clip_value,
            )
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
        metrics = {
            "imag_objective": objective.detach(),
            "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean().detach(),
            "imag_reward_mean": ro.rewards.mean().detach(),
            "imag_survival_H": ro.survival[-1].mean().detach(),
            "imag_dhat_mean": ro.dhat[1:].mean().detach(),
            "policy_grad_norm": grad.detach(),
            "action_abs_mean": ro.actions.abs().mean().detach(),
            "finite": finite.detach(),
        }
        for key, value in ro.reward_terms.items():
            metrics[f"imag_r/{key}"] = value.detach()
        if cfg.use_critic:
            critic_metrics = self._fit_critic(ro, feat, e_root, xi_norm)
            finite = finite & critic_metrics.pop("finite")
            metrics.update(critic_metrics)
        metrics["finite"] = finite.detach()
        return metrics

    def _fit_critic(self, ro, feat, e_root, xi_norm) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        with torch.no_grad():
            feat_d, goals_d, e_root_d = feat.detach(), ro.goals.detach(), e_root.detach()
            h1, n, _ = feat_d.shape
            xi_rep = xi_norm[None].expand(h1, n, -1)
            values = self.critic(
                feat_d.reshape(h1 * n, -1), goals_d.reshape(h1 * n, -1),
                xi_rep.reshape(h1 * n, -1), e_root_d.reshape(h1 * n, -1),
            ).reshape(h1, n)
            targets = gae_targets(
                ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda
            )
            x_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
            x_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
            x_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
            x_er = e_root_d[:-1].reshape(-1, 4)
            y = targets.reshape(-1)
        count = y.shape[0]
        minibatch = count // cfg.critic_minibatches
        last = y.new_zeros(())
        grad_max = y.new_zeros((), dtype=torch.float64)
        finite = torch.ones((), device=y.device, dtype=torch.bool)
        for _ in range(cfg.critic_epochs):
            permutation = torch.randperm(count, device=y.device)
            for i in range(cfg.critic_minibatches):
                idx = permutation[i * minibatch : (i + 1) * minibatch]
                pred = self.critic(x_feat[idx], x_goal[idx], x_xi[idx], x_er[idx])
                closs = torch.nn.functional.mse_loss(pred, y[idx])
                grad, ok = optimize_step(
                    self.critic_opt, closs, self.critic, max_norm=cfg.critic_grad_clip
                )
                finite = finite & ok
                grad_max = torch.maximum(grad_max, grad.detach().to(torch.float64))
                last = closs.detach()
        return {
            "critic_loss": last,
            "critic_grad_norm_max": grad_max,
            "critic_value_mean": values.mean().detach(),
            "critic_target_mean": targets.mean().detach(),
            "finite": finite,
        }

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        del xi_norm_fn
        start = buffer.sample_starts(self.cfg.batch_size, self.cfg.history)
        packed = {
            "state": start.state,
            "xi": start.xi,
            "frame": start.frame,
            "prev_action": start.prev_action,
            "hist_states": start.hist_states,
            "hist_actions": start.hist_actions,
        }
        if start.hist_frames is not None:
            packed["hist_frames"] = start.hist_frames
        outputs = self._captured(packed) if self._graph_enabled() else self._update_from(packed)
        if not bool(outputs["finite"]):
            raise FloatingPointError(
                "Policy loss contains NaN/Inf; stopped before optimizer.step()"
            )
        metrics = {key: float(value) for key, value in outputs.items() if key != "finite"}
        self.latest = metrics
        return metrics

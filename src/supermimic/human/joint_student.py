"""MorphMimic copy step, starting from real mixed-interface replay.

Only the human encoder is optimized here. The motion step separately trains the
shared decoder, and the WM step separately trains the model of both interfaces.
"""
from __future__ import annotations

from contextlib import contextmanager

import mujoco
import torch

from supermimic.kinematics.rotations import quat_from_angle_axis, quat_mul
from supermimic.training.numerics import backward_step
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
from .features import sample_human_masks
from .loss_scales import TermScales


@contextmanager
def frozen(*modules):
    params = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


class TorsoFrame:
    """Casbot02 waist/torso orientation from root IMU + actual joint axes."""
    def __init__(self, model, device):
        chain = []
        body = model.body("waist_yaw_link").id
        while body:
            count = int(model.body_jntnum[body])
            joint = int(model.body_jntadr[body]) if count else None
            if count > 1:
                raise ValueError("Unsupported torso body with multiple joints")
            if joint is not None and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                if model.jnt_qposadr[joint] != 0:
                    raise ValueError("Free root must be at qpos address zero")
                break
            fixed = torch.tensor(model.body_quat[body].copy(), dtype=torch.float32, device=device)
            axis = None if joint is None else torch.tensor(model.jnt_axis[joint].copy(), dtype=torch.float32, device=device)
            adr = None if joint is None else int(model.jnt_qposadr[joint])
            rest = 0. if joint is None else float(model.qpos0[adr])
            chain.append((fixed, axis, adr, rest))
            body = int(model.body_parentid[body])
        if not body:
            raise ValueError("Torso must descend from the free root")
        self.chain = list(reversed(chain))

    def __call__(self, state):
        q = state[..., 3:7]
        for fixed, axis, adr, rest in self.chain:
            q = quat_mul(q, fixed.expand_as(q))
            if axis is not None:
                q = quat_mul(q, quat_from_angle_axis(state[..., adr] - rest, axis.expand(*state.shape[:-1], 3)))
        return q


class JointCopyStep:
    def __init__(self, wm, policy, human, library, windows, torso, *, batch_size=512,
                 horizon=32, masking="mixed", lr=2e-4, obs_noise=True):
        self.wm, self.policy, self.human = wm, policy, human
        self.lib, self.windows, self.torso = library, windows, torso
        self.batch_size, self.horizon, self.masking = batch_size, horizon, masking
        self.obs_noise = obs_noise
        self.opt = torch.optim.Adam(human.parameters(), lr=lr, betas=(.9, .95))
        self.scales = TermScales(("copy", "align")).to(library.device)
        self.noise_half = torch.tensor([.01]*27 + [.5]*27 + [.2]*3 + [.05]*3 + [0.]*27 + [.05]*6, device=library.device)

    def noise(self, obs):
        return obs + (torch.rand_like(obs)*2-1)*self.noise_half if self.obs_noise else obs

    def loss(self, start, *, update_norms=True):
        n, h = len(start.state), len(start.hist_states)
        if start.hist_frames is None or h != 10:
            raise ValueError("Joint copy needs the native 10-frame reference-pelvis history")
        quats = self.lib.ref(start.hist_frames.reshape(-1)).root_quat_w.reshape(h, n, 4)
        hist = [self.noise(analytic_observation(start.hist_states[i], start.hist_actions[i],
                    start.xi, self.lib.default_joint_pos, quats[i])) for i in range(h)]
        last = self.lib.last_frame_of(start.frame)
        frames = torch.minimum(start.frame[:, None] + torch.arange(self.horizon+10, device=start.state.device), last[:, None])
        goal_all = self.lib.goal_window(frames[:, :self.horizon].reshape(-1), 10).reshape(n, self.horizon, -1)
        refs = self.lib.ref(frames.reshape(-1)).root_quat_w.reshape(n, -1, 4)
        poses = self.windows.future(start.frame, self.horizon+10)
        points, times, heading = sample_human_masks(n, self.masking, start.state.device,
                                                   heading_frac=1., preview_mix=False)
        state = start.state.detach()
        action_error, latent_error = [], []
        for t in range(self.horizon):
            history = flatten_history_term_major(torch.stack(hist), num_joints=27).detach()
            with torch.no_grad():
                teacher_z = self.policy.encoders["motion"](goal_all[:, t])
                target = self.policy.decoder(teacher_z, history)
                anchor = self.torso(state)
            window = poses[:, t:t+10]
            packed = self.human.pack(window[..., :3], window[..., 3:], anchor,
                                     points, times, heading, update=update_norms)
            z = self.human(packed)
            action = self.policy.decoder(z, history)
            action_error.append((action-target).square().mean(-1))
            latent_error.append((z-teacher_z).square().mean(-1))
            with torch.no_grad():
                state = self.wm.step(state.detach(), action.detach(), start.xi)
                obs = analytic_observation(state, action.detach(), start.xi,
                                            self.lib.default_joint_pos, refs[:, t+1])
                hist = hist[1:] + [self.noise(obs)]
        # Match upstream copy's rewarded-step mask: do not train on padding beyond
        # the source clip end. No critic or return term in this student.
        mask = torch.arange(self.horizon, device=state.device)[:, None] <= (last-start.frame)[None]
        denom = mask.sum().clamp_min(1)
        return {"copy": (torch.stack(action_error)*mask).sum()/denom,
                "align": (torch.stack(latent_error)*mask).sum()/denom}

    def update(self, replay):
        self.human.train()
        start = replay.sample_starts(self.batch_size, 10)
        versions = [p._version for m in (self.wm, self.policy) for p in m.parameters()]
        with frozen(self.wm, self.policy):
            terms = self.loss(start)
            scales = self.scales.scales()
            loss = terms["copy"]/scales["copy"] + terms["align"]/scales["align"]
            gn = backward_step(loss, self.human, self.opt, name="Joint human copy",
                               max_norm=1., max_value=.1)
        if versions != [p._version for m in (self.wm, self.policy) for p in m.parameters()]:
            raise RuntimeError("Human copy step modified the motion policy/world model")
        self.scales.update(terms)
        return {"action_mse": float(terms["copy"].detach()), "latent_mse": float(terms["align"].detach()),
                "loss": float(loss.detach()), "grad_norm": float(gn)}


class JointPolicyUpdates:
    """Compose the native motion/critic update with the extra human update."""
    def __init__(self, native, human_step):
        self.native, self.human_step = native, human_step
        self.opt, self.critic_opt, self.cfg = native.opt, native.critic_opt, native.cfg
        self.motion_updates = self.human_updates = 0
        self.latest = {}

    def set_lr(self, lr):
        self.native.set_lr(lr)
        for group in self.human_step.opt.param_groups:
            group["lr"] = lr

    def update(self, replay):
        result = self.native.update(replay)
        self.motion_updates += 1
        self.latest = self.human_step.update(replay)
        self.human_updates += 1
        return result | {f"human/{k}": v for k, v in self.latest.items()}

"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary, RefState
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.kinematics.rotations import cached_constant
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import (
    ReferenceFrame,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, goal)
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
    anchor_obs: bool = False,
    anchor_ori_obs: bool = False,
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    # History rot6d only for the orientation task. Full anchor mode keeps its goal suffix.
    pelvis = anchor_ori_obs and not anchor_obs
    step_fn = step_fn or ImagStep(wm, q0, reference_pelvis_ori=pelvis)
    nj = wm.spec.layout.num_joints
    # Uniform half-widths: joint position, joint velocity, angular velocity, gravity, action.
    noise_values = (0.01,) * nj + (0.5,) * nj + (0.2,) * 3 + (0.05,) * 3 + (0.0,) * nj
    if pelvis:
        noise_values += (0.05,) * 6
    noise_half = cached_constant(noise_values, start.state)
    anchor_values = (0.25,) * 3 + (0.05,) * 6 if anchor_obs else (0.05,) * 6
    anchor_noise_half = cached_constant(anchor_values, start.state)
    ai = wm.heads.anchor_index

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    def make_goal(step, ref_t, anchor_pose):
        g = goal_rows[step]
        if not anchor_obs:
            return g
        rel = anchor_relative_obs(
            anchor_pose[..., 0:3], anchor_pose[..., 3:7],
            ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :],
        )
        if obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * anchor_noise_half
        return torch.cat([g, rel], dim=-1)

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    hist_len = start.hist_states.shape[0]
    hist_quat = None
    if pelvis:
        if start.hist_frames is None:
            raise ValueError("The orientation task needs the reference frame of each history row")
        hist_ref = lib.ref(start.hist_frames.reshape(-1))
        hist_quat = hist_ref.root_quat_w.reshape(hist_len, n, 4)
    hist = [
        add_noise(
            analytic_observation(
                start.hist_states[i],
                start.hist_actions[i],
                start.xi,
                q0,
                None if hist_quat is None else hist_quat[i],
            )
        )
        for i in range(hist_len)
    ]
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # The whole rollout's reference frames are known before the first actor call.
    # Read that span once, build every goal window from it, and FK only the
    # frames the reward actually consumes.
    last = lib.last_frame_of(frame)
    frames_all = torch.minimum(
        frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :]
    )  # (H+1, n)
    span = lib.reference_span(frame, horizon + future_frames)
    goal_rows = lib.goals_from_span(span, horizon, future_frames)
    if lib.compact:
        step_ref = RefState(*(part[: horizon + 1].reshape(-1, *part.shape[2:]) for part in (
            span.root_pos_w, span.root_quat_w, span.joint_pos,
            span.root_lin_vel_w, span.root_ang_vel_w, span.joint_vel,
        )))
        bodies = lib.bodies_from_ref(step_ref)
        ref_all = ReferenceFrame(
            bodies.pos_w, bodies.quat_w, bodies.lin_vel_w, bodies.ang_vel_w,
            step_ref.root_pos_w, step_ref.root_quat_w,
        )
    else:
        ref_all = reference_at(lib, frames_all.reshape(-1))

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(
            ref_all.body_pos_w[sl],
            ref_all.body_quat_w[sl],
            ref_all.body_lin_vel_w[sl],
            ref_all.body_ang_vel_w[sl],
            ref_all.root_pos_w[sl],
            ref_all.root_quat_w[sl],
        )

    ref = ref_at(0)
    if anchor_obs:
        b0 = wm.heads.robot_bodies(s)
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    goals = [make_goal(0, ref, anchor_pose)]
    e_roots = [wm.heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)

    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0), num_joints=nj)
        a = policy(obs_flat, goals[-1])
        next_root_quat = ref_at(t + 1).root_quat_w
        s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(
            s, a, a_prev, start.xi, ref, next_root_quat
        )
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)

        frame = frames_all[t + 1]
        ref = ref_at(t + 1)
        hist = hist[1:] + [add_noise(obs_clean)]
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(make_goal(t + 1, ref, anchor_pose))
        e_roots.append(wm.heads.root_error_features(s, ref))
        frames.append(frame)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        reward_terms={
            name: (term_sum[index] / horizon).detach()
            for index, name in enumerate(REWARD_TERM_NAMES)
        },
    )

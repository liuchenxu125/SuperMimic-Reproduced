"""Streaming actor with exactly the existing 930/690 observation conventions."""

from __future__ import annotations

import numpy as np
import torch

from supermimic.deployment.sim2sim import (
    History,
    observation,
    reference_heading_goal,
    rotation,
    yaw,
)

from .reference import stationary, validate_frames


class StreamingActor:
    def __init__(self, actor, meta):
        self.actor, self.meta = actor, meta
        self.q0 = np.array([m["q0"] for m in meta["motors"]])
        self.scale = np.array([m["scale"] for m in meta["motors"]])
        self.history = History(27, 10, reference_pelvis_ori=True)
        self.previous = np.zeros(27, dtype=np.float32)
        self.turn = None
        self.last_reference = None
        self.first = True

    def align(self, qpos, first):
        angle = float(yaw(qpos[3:7]) - yaw(first[3:7]))
        self.turn = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        self.origin = first[:3].copy()

    def aligned(self, frames):
        from supermimic.deployment.sim2sim import quat_multiply

        r = rotation(self.turn)
        out = frames.copy()
        out[:, :3] = (frames[:, :3] - self.origin) @ r.T + self.origin
        out[:, 3:7] = quat_multiply(self.turn, frames[:, 3:7])
        out[:, 61:64] = frames[:, 61:64] @ r.T
        out[:, 64:67] = frames[:, 64:67] @ r.T
        return out

    def inputs(self, qpos, qvel, frames):
        frames = validate_frames(frames, self.meta)
        if self.turn is None:
            self.align(qpos, frames[0])
        frames = self.aligned(frames)
        obs = observation(qpos, qvel, self.previous, self.q0, frames[0, 3:7])
        if self.first:
            self.history.reset(obs)
            self.first = False
        else:
            self.history.append(obs)
        goal = reference_heading_goal(
            frames[:, :3],
            frames[:, 3:7],
            frames[:, 61:64],
            frames[:, 64:67],
            frames[:, 7:34],
            frames[:, 34:61],
            self.q0,
            0,
            10,
        )
        self.last_reference = frames[0].copy()
        return self.history.packed(), goal

    def act(self, qpos, qvel, frames):
        obs, goal = self.inputs(qpos, qvel, frames)
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(obs[None]), torch.from_numpy(goal[None]))[
                0
            ].numpy()
        if action.shape != (27,) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/invalid actor output")
        self.previous[:] = action
        return action, self.q0 + self.scale * action, obs, goal


class HoldOnTimeout:
    """Stream loss holds a stationary reference with feedback, never repeats motion.

    Resumption after timeout is explicit. This is not a balance guarantee: the
    safe recovery/standing policy remains a separate validation requirement.
    """

    def __init__(self, receiver):
        self.receiver = receiver
        self.last = None
        self.latched = False

    def resume(self):
        if not self.receiver.fresh():
            raise ValueError("Cannot resume without a fresh reference")
        self.latched = False

    def window(self):
        if self.receiver.latest is None:
            return None
        if not self.receiver.fresh():
            self.latched = True
        if self.latched:
            return (
                stationary(self.last[0])
                if self.last is not None
                else stationary(self.receiver.latest.frames[0])
            )
        win = self.receiver.latest
        self.last = stationary(win.frames[0]) if win.mode == "HOLD" else win.frames.copy()
        return self.last.copy()

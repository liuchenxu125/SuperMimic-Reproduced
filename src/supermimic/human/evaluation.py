"""Deterministic full-clip real-physics evaluation of both joint interfaces.

All modes share the same environment/reward/termination and reference clips.
Human goal features use the real robot torso, matching joint collection. No
world-model rollout or teacher action is used to score the human interfaces.
"""

import torch

from .features import fixed_human_masks


class JointEvaluationActor:
    def __init__(self, policy, human, windows, torso, state_io, command, interfaces):
        self.policy, self.human = policy.eval(), human.eval()
        self.windows, self.torso, self.io, self.command = windows, torso, state_io, command
        self.interfaces = interfaces

    @torch.no_grad()
    def __call__(self, obs, modes):
        actions = torch.empty((len(modes), self.policy.cfg.num_joints), device=obs["policy"].device)
        human_rows = []
        for mode, name in enumerate(self.interfaces):
            ids = (modes == mode).nonzero().flatten()
            if not len(ids):
                continue
            if name == "motion":
                actions[ids] = self.policy(obs["policy"][ids], obs["goal"][ids])
            else:
                human_rows.append((name, ids))
        if human_rows:
            # One CPU gather and one feature/decoder batch for both human masks.
            ids = torch.cat([row for _, row in human_rows])
            poses = self.windows.future(self.command.frame[ids], 10)
            masks = [
                fixed_human_masks(len(row), name.removeprefix("human_"), 10, actions.device)
                for name, row in human_rows
            ]
            points, times, heading = [torch.cat([m[i] for m in masks]) for i in range(3)]
            state = self.io.read()[ids]
            goal = self.human.pack(
                poses[..., :3],
                poses[..., 3:],
                self.torso(state),
                points,
                times,
                heading,
                update=False,
            )
            actions[ids] = self.policy.decoder(self.human(goal), obs["policy"][ids])
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite evaluation actions")
        return actions

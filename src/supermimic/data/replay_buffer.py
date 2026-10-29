"""Episode-safe device replay, reconstructed for this project's training interfaces.

Each environment owns a ring of state rows. A normal row holds s_t and its
outgoing transition; a terminal row holds the true pre-reset s_{t+1} only.
Sequence numbers distinguish chronological neighbours from wrapped storage.
Timeouts end a sequence but are not positive labels for physical termination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.robots.layout import RobotLayout


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k+1, B, 71)
    a_clean: torch.Tensor  # (k, B, 29); legacy name: env input (already noisy in collection mode)
    a_exec: torch.Tensor  # (k, B, 29)
    xi: torch.Tensor  # (B, 33), constant within an episode
    done: torch.Tensor  # (k, B), physical termination, excluding timeouts
    frame: torch.Tensor  # (k, B), target used for each transition


@dataclass
class StartBatch:
    state: torch.Tensor
    xi: torch.Tensor
    frame: torch.Tensor
    prev_action: torch.Tensor
    hist_states: torch.Tensor  # (hist, B, 71), oldest first
    hist_actions: torch.Tensor  # previous env-input action at each historical state


class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        capacity: int,
        device: str | torch.device = "cuda",
        num_joints: int = 29,
    ):
        if num_envs < 1 or capacity < 2:
            raise ValueError("Replay needs num_envs >= 1 and capacity >= 2")
        self.N, self.T = num_envs, capacity
        self.device = torch.device(device)
        layout = RobotLayout(num_joints)
        shape = (num_envs, capacity)

        def floats(dim):
            return torch.zeros((*shape, dim), device=self.device)

        self.states = floats(layout.state_dim)
        self.a_clean = floats(num_joints)
        self.a_exec = floats(num_joints)
        self.prev_actions = floats(num_joints)
        self.xi = floats(layout.xi_dim)
        self.reward = torch.zeros(shape, device=self.device)
        self.done = torch.zeros(shape, device=self.device)
        self.frame = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.clip = torch.zeros_like(self.frame)
        self.valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.terminal = torch.zeros_like(self.valid)
        self.ep_start = torch.zeros_like(self.valid)
        self._sequence = torch.full_like(self.frame, -1)
        self._episode = torch.full_like(self.frame, -1)
        self._step = torch.zeros_like(self.frame)
        self._envs = torch.arange(num_envs, device=self.device)
        self._ptr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._count = torch.zeros_like(self._ptr)
        self._next_sequence = torch.zeros_like(self._ptr)
        self._episode_id = torch.zeros_like(self._ptr)
        self._episode_step = torch.zeros_like(self._ptr)
        self._last_action = torch.zeros((num_envs, num_joints), device=self.device)

    @property
    def size(self) -> int:
        """Number of stored state rows, including terminal rows."""
        return int(self._count.sum().item())

    def _commit(self, ids: torch.Tensor) -> None:
        rows = self._ptr[ids]
        self.valid[ids, rows] = True
        self._sequence[ids, rows] = self._next_sequence[ids]
        self._episode[ids, rows] = self._episode_id[ids]
        self._step[ids, rows] = self._episode_step[ids]
        self._next_sequence[ids] += 1
        self._ptr[ids] = (rows + 1) % self.T
        self._count[ids] = (self._count[ids] + 1).clamp_max(self.T)

    @torch.no_grad()
    def add_step(self, s, a, a_exec, xi, r, d, clip, frame, ep_start, env_ids=None) -> None:
        """Append outgoing transitions; all inputs correspond to env_ids in order.

        ``s`` is pre-action state, ``frame`` is its target before command advancement,
        ``d`` is physical termination (not time limit), and ``ep_start`` marks the
        first transition after any reset. Inputs are copied into owned storage.
        """
        ids = self._envs if env_ids is None else env_ids.to(self.device, dtype=torch.long)
        rows = self._ptr[ids]
        first = torch.as_tensor(ep_start, device=self.device, dtype=torch.bool).expand(len(ids))
        self._episode_id[ids] += first.long()
        self._episode_step[ids] = torch.where(first, 0, self._episode_step[ids])
        self.states[ids, rows] = s
        self.a_clean[ids, rows] = a
        self.a_exec[ids, rows] = a_exec
        self.prev_actions[ids, rows] = torch.where(first[:, None], 0, self._last_action[ids])
        self.xi[ids, rows] = xi
        self.reward[ids, rows] = r
        self.done[ids, rows] = d.to(dtype=self.done.dtype)
        self.clip[ids, rows] = clip
        self.frame[ids, rows] = frame
        self.ep_start[ids, rows] = first
        self.terminal[ids, rows] = False
        self._commit(ids)
        self._last_action[ids] = a
        self._episode_step[ids] += 1

    @torch.no_grad()
    def add_terminal(self, env_ids: torch.Tensor, s_next: torch.Tensor) -> None:
        """Append true end states before the simulator overwrites them on reset."""
        ids = env_ids.to(self.device, dtype=torch.long)
        rows = self._ptr[ids]
        previous = (rows - 1) % self.T
        if not bool((self.valid[ids, previous] & ~self.terminal[ids, previous]).all()):
            raise ValueError("A terminal state must follow an outgoing transition")
        self.states[ids, rows] = s_next
        self.xi[ids, rows] = self.xi[ids, previous]
        self.frame[ids, rows] = self.frame[ids, previous]
        self.clip[ids, rows] = self.clip[ids, previous]
        self.prev_actions[ids, rows] = self._last_action[ids]
        self.a_clean[ids, rows] = 0
        self.a_exec[ids, rows] = 0
        self.reward[ids, rows] = 0
        self.done[ids, rows] = self.done[ids, previous]
        self.ep_start[ids, rows] = False
        self.terminal[ids, rows] = True
        self._commit(ids)

    def _sample_rows(self, n: int, k: int = 0):
        if n < 1:
            raise ValueError("Batch size must be positive")
        if not bool(self._count.sum()):
            raise RuntimeError("Replay is empty; collect simulator transitions first")
        found_env, found_row = [], []
        remaining = n
        # Rejection sampling only over occupied rows. Endpoint sequence/episode
        # checks reject overwritten data, future rows and reset crossings.
        for _ in range(64):
            count = max(2 * remaining, 64)
            ids = torch.multinomial(self._count.float(), count, replacement=True)
            age = (torch.rand(count, device=self.device) * self._count[ids]).long()
            rows = (self._ptr[ids] - self._count[ids] + age) % self.T
            mask = ~self.terminal[ids, rows]
            if k:
                end = (rows + k) % self.T
                mask &= self.valid[ids, end]
                mask &= self._sequence[ids, end] == self._sequence[ids, rows] + k
                mask &= self._episode[ids, end] == self._episode[ids, rows]
                path = (rows[:, None] + torch.arange(k, device=self.device)) % self.T
                mask &= ~self.terminal[ids[:, None], path].any(-1)
            ids, rows = ids[mask][:remaining], rows[mask][:remaining]
            found_env.append(ids)
            found_row.append(rows)
            remaining -= ids.numel()
            if remaining == 0:
                return torch.cat(found_env), torch.cat(found_row)
        raise RuntimeError(
            f"Not enough contiguous replay data for {k}-step windows; "
            "collect longer episodes or reduce the window"
        )

    @torch.no_grad()
    def sample_windows(self, n: int, k: int) -> WindowBatch:
        if not 1 <= k < self.T:
            raise ValueError("Window length must be between 1 and capacity - 1")
        ids, start = self._sample_rows(n, k)
        rows = (start[None] + torch.arange(k + 1, device=self.device)[:, None]) % self.T
        return WindowBatch(
            states=self.states[ids[None], rows],
            a_clean=self.a_clean[ids[None], rows[:-1]],
            a_exec=self.a_exec[ids[None], rows[:-1]],
            xi=self.xi[ids, start],
            done=self.done[ids[None], rows[:-1]],
            frame=self.frame[ids[None], rows[:-1]],
        )

    @torch.no_grad()
    def sample_starts(self, n: int, hist: int = 10) -> StartBatch:
        if hist < 1:
            raise ValueError("History length must be positive")
        ids, start = self._sample_rows(n)
        # Repeat the earliest available state at episode/buffer boundaries,
        # matching mjlab's observation history padding convention.
        available = self._sequence[ids, start] - (self._next_sequence[ids] - self._count[ids])
        available = torch.minimum(available, self._step[ids, start])
        back = torch.minimum(
            torch.arange(hist - 1, -1, -1, device=self.device)[:, None], available[None]
        )
        rows = (start[None] - back) % self.T
        return StartBatch(
            state=self.states[ids, start],
            xi=self.xi[ids, start],
            frame=self.frame[ids, start],
            prev_action=self.prev_actions[ids, start],
            hist_states=self.states[ids[None], rows],
            hist_actions=self.prev_actions[ids[None], rows],
        )

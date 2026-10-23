"""mjlab recorder bridging pre-reset simulator states to the training replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.robots.layout import RobotLayout


class ReplayRecorder(RecorderTerm):
    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        # Deferred imports avoid envs.__init__ -> tracking_env_cfg -> recorder cycles.
        from supermimic.envs.sim_context import SimContext
        from supermimic.envs.state_io import StateIO

        self.io = StateIO(env)
        self.ctx = SimContext(
            env, cfg.params["foot_geom_regex"], cfg.params.get("com_body_name", "torso_link")
        )
        layout = RobotLayout(env.action_manager.action.shape[-1])
        self.buffer = ReplayBuffer(
            env.num_envs, int(cfg.params["capacity"]), env.device, num_joints=layout.num_joints
        )
        self.command = env.command_manager.get_term("motion")
        self.action = env.action_manager.get_term("joint_pos")
        self._state = torch.zeros((env.num_envs, layout.state_dim), device=env.device)
        self._xi = torch.zeros((env.num_envs, layout.xi_dim), device=env.device)
        self._frame = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._first = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._env_ids = torch.arange(env.num_envs, device=env.device)

    def _cache(self, ids):
        self._state[ids] = self.io.read()[ids]
        self._xi[ids] = self.ctx.read()[ids]
        self._frame[ids] = self.command.frame[ids]

    def _record(self, ids):
        env = self._env
        self.buffer.add_step(
            self._state[ids],
            env.action_manager.action[ids],
            self.action.executed_raw[ids],
            self._xi[ids],
            env.reward_buf[ids],
            env.reset_terminated[ids],
            self.command.library.frame_clip[self._frame[ids]],
            self._frame[ids],
            self._first[ids],
            env_ids=ids,
        )
        self._first[ids] = False

    @torch.no_grad()
    def record_pre_reset(self, env_ids):
        # Actions, xi and generalized coordinates still describe the old episode.
        # Read qpos/qvel directly: derived body kinematics are one substep stale here.
        self._record(env_ids)
        self.buffer.add_terminal(env_ids, self.io.read()[env_ids])

    @torch.no_grad()
    def record_post_reset(self, env_ids):
        self._cache(env_ids)
        self._first[env_ids] = True

    @torch.no_grad()
    def record_post_step(self):
        env = self._env
        if env.cfg.auto_reset:
            # Reset envs were recorded in pre_reset and cached in post_reset.
            ids = self._env_ids[~env.reset_buf]
        else:
            ids = self._env_ids
        self._record(ids)
        if not env.cfg.auto_reset:
            done_ids = self._env_ids[env.reset_buf]
            if len(done_ids):
                self.buffer.add_terminal(done_ids, self.io.read()[done_ids])
        self._cache(ids)


@dataclass
class ReplayRecorderCfg(RecorderTermCfg):
    func: type = ReplayRecorder

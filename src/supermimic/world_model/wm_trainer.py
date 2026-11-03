"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.training.numerics import backward_step
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg,
                 diagnostic_dir: Path | None = None):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.diagnostic_dir = diagnostic_dir
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            try:
                loss, m = self.loss_on(w)
                gn = backward_step(
                    loss, self.wm, self.opt, name="World model", max_norm=self.cfg.grad_clip
                )
            except (FloatingPointError, RuntimeError) as error:
                if self.diagnostic_dir is not None:
                    from supermimic.training.failure_dump import dump_world_model_failure

                    dump_world_model_failure(self.diagnostic_dir, self.wm, self.lib, self.cfg, w, error)
                raise
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)

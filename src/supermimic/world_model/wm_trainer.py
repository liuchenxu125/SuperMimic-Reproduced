"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.training.cuda_graph import CapturedUpdate, graph_state, optimize_step
from supermimic.training.distributed import world_size
from supermimic.training.optim import assign_lr, make_optimizers
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


def _batch_to_dump(batches: list[WindowBatch]) -> WindowBatch:
    for batch in batches:
        for key in batch.__dataclass_fields__:
            value = getattr(batch, key)
            if value.is_floating_point() and not torch.isfinite(value).all():
                return batch
    return batches[-1]


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
        compile_fk = getattr(lib, "compile_reference_fk", None)
        if compile_fk is not None:
            compile_fk(cfg.compile)
        self.opt = make_optimizers(wm, lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)
        self.capture = True
        self._captured = CapturedUpdate(self._update_from, self._graph_state)

    def set_lr(self, lr: float) -> None:
        assign_lr(self.opt, lr)

    def _graph_state(self) -> list[torch.Tensor]:
        state = graph_state([self.wm], [self.opt])
        ref = getattr(self.lib, "_ref", None)
        if torch.is_tensor(ref):
            state.append(ref)
        return state

    def _graph_enabled(self) -> bool:
        if not self.capture or world_size() > 1:
            return False
        params = list(self.wm.parameters())
        if not params or params[0].device.type != "cuda":
            return False
        storage = getattr(self.lib, "storage_device", None)
        return storage is None or torch.device(storage).type == "cuda"

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def _window_from(self, packed: dict[str, torch.Tensor], index: int) -> WindowBatch:
        return WindowBatch(
            states=packed["states"][index],
            a_clean=packed["a_clean"][index],
            a_exec=packed["a_exec"][index],
            xi=packed["xi"][index],
            done=packed["done"][index],
            frame=packed["frame"][index],
        )

    def _update_from(self, packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        count = packed["states"].shape[0]
        agg: dict[str, torch.Tensor] = {}
        finite = torch.ones((), device=packed["states"].device, dtype=torch.bool)
        for index in range(count):
            loss, metrics = self.loss_on(self._window_from(packed, index))
            norm, ok = optimize_step(
                self.opt, loss, self.wm, max_norm=self.cfg.grad_clip
            )
            finite = finite & ok
            metrics = dict(metrics)
            metrics["loss"] = loss.detach()
            metrics["grad_norm"] = norm.detach()
            for key, value in metrics.items():
                agg[key] = value if key not in agg else agg[key] + value
        out = {key: value / count for key, value in agg.items()}
        out["finite"] = finite
        return out

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        batches = [
            buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            for _ in range(self.cfg.batches_per_update)
        ]
        packed = {
            key: torch.stack([getattr(batch, key) for batch in batches])
            for key in batches[0].__dataclass_fields__
        }
        try:
            outputs = self._captured(packed) if self._graph_enabled() else self._update_from(packed)
            if not bool(outputs["finite"]):
                raise FloatingPointError(
                    "World model loss contains NaN/Inf; stopped before optimizer.step()"
                )
        except (FloatingPointError, RuntimeError) as error:
            if self.diagnostic_dir is not None:
                from supermimic.training.failure_dump import dump_world_model_failure

                dump_world_model_failure(
                    self.diagnostic_dir, self.wm, self.lib, self.cfg, _batch_to_dump(batches), error
                )
            raise
        return {key: float(value) for key, value in outputs.items() if key != "finite"}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)

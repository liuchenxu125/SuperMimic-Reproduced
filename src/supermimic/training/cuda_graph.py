"""Record a whole training update the way the official learners do.

Sampling stays outside the graph. The captured region is the forward, backward,
gradient clip and optimizer step. Every captured update on one device shares one
stream and one memory pool, and replays in the order it was recorded.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from supermimic.training.distributed import allreduce_grads


def graph_state(modules, optimizers) -> list[torch.Tensor]:
    """Parameters, buffers, optimizer state and learning-rate tensors.

    These are the storages Python can rebind. A replay refuses to run if one
    of them has moved.
    """
    state: list[torch.Tensor] = []
    for module in modules:
        state += [*module.parameters(), *module.buffers()]
    for optimizer in optimizers:
        state += [tensor for slot in optimizer.state.values() for tensor in slot.values()]
        state += [
            group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])
        ]
    return state


def _identity(state: list[torch.Tensor]) -> list[int]:
    return [tensor.data_ptr() for tensor in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
    return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
    """Eager for the first ``warmup`` calls, then one CUDA graph.

    The capture call itself does not launch the recorded kernels, so this
    replays once before returning that call's outputs. Later calls only replay.
    CPU tensors take the eager path. A learning rate that changes must already
    be a device tensor updated with ``fill_``.
    """

    def __init__(self, fn: Callable, state: Callable[[], list[torch.Tensor]], warmup: int = 3):
        self._fn = fn
        self._state = state
        self._warmup = warmup
        self._inputs: dict[str, torch.Tensor] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._identity: list[int] = []
        self._outputs: dict[str, torch.Tensor] = {}

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = self._state()
        if not state or not state[0].is_cuda:
            return self._fn(inputs)
        with torch.cuda.device(state[0].device):
            stream, pool = _capture_stream_and_pool(state[0].device)
            if self._inputs is None:
                self._inputs = {key: value.detach().clone() for key, value in inputs.items()}
            else:
                for key, value in inputs.items():
                    self._inputs[key].copy_(value)
            if self._warmup > 0:
                self._warmup -= 1
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs = self._fn(self._inputs)
                torch.cuda.current_stream().wait_stream(stream)
                return outputs
            if self._graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._outputs = self._fn(self._inputs)
                self._graph = graph
                self._identity = _identity(self._state())
                graph.replay()
            elif _identity(state) != self._identity:
                raise RuntimeError("a captured tensor moved")
            else:
                self._graph.replay()
            torch.cuda.current_stream().synchronize()
            return self._outputs


def optimize_step(optimizer, loss, module, *, max_norm: float, max_value: float | None = None):
    """One optimizer step without a host sync.

    A non-finite loss or gradient is replaced by zero before ``step``, so the
    weights do not absorb NaN. The caller checks the returned flag after the
    graph has replayed; a branch inside the graph cannot be captured.
    """
    optimizer.zero_grad(set_to_none=True)
    finite = torch.isfinite(loss.detach()).all()
    safe = torch.where(finite, loss, loss.new_zeros(()))
    safe.backward()
    allreduce_grads(module)
    grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if grads:
        usable = finite & torch.stack([torch.isfinite(grad).all() for grad in grads]).all()
        for grad in grads:
            grad.mul_(usable)
        finite = usable
    norm = torch.nn.utils.clip_grad_norm_(
        module.parameters(), max_norm, error_if_nonfinite=False
    )
    if max_value is not None:
        torch.nn.utils.clip_grad_value_(module.parameters(), max_value)
    optimizer.step()
    return norm.detach(), finite.detach()

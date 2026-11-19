"""Fused Adam must follow the same update as the eager kernel and reload its state."""

import copy

import torch

from supermimic.training.optim import make_adam


def _step(model, optimizer, inputs, target):
    optimizer.zero_grad(set_to_none=True)
    (model(inputs) - target).square().mean().backward()
    optimizer.step()


def test_cpu_adam_stays_eager():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = make_adam([parameter], lr=1e-3, betas=(0.9, 0.95))
    assert not optimizer.defaults.get("fused")
    assert optimizer.defaults["betas"] == (0.9, 0.95)


def test_fused_adam_matches_eager_and_reloads():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    fused_model = torch.nn.Linear(4, 3, device="cuda")
    eager_model = torch.nn.Linear(4, 3, device="cuda")
    eager_model.load_state_dict(fused_model.state_dict())
    fused = make_adam(fused_model.parameters(), lr=2e-4, betas=(0.9, 0.95))
    eager = torch.optim.Adam(eager_model.parameters(), lr=2e-4, betas=(0.9, 0.95))
    assert fused.defaults.get("fused") is True
    inputs = torch.randn(5, 4, device="cuda")
    target = torch.randn(5, 3, device="cuda")
    for _ in range(3):
        _step(fused_model, fused, inputs, target)
        _step(eager_model, eager, inputs, target)
    for left, right in zip(fused_model.parameters(), eager_model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    # Checkpoint tensors are cloned the same way torch.save does. A live
    # state_dict keeps the fused step tensor, so it must not be shared.
    restored = torch.nn.Linear(4, 3, device="cuda")
    restored_opt = make_adam(restored.parameters(), lr=2e-4, betas=(0.9, 0.95))
    restored.load_state_dict(copy.deepcopy(fused_model.state_dict()))
    restored_opt.load_state_dict(copy.deepcopy(fused.state_dict()))
    _step(fused_model, fused, inputs, target)
    _step(restored, restored_opt, inputs, target)
    for left, right in zip(fused_model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    resumed = torch.nn.Linear(4, 3, device="cuda")
    resumed_opt = make_adam(resumed.parameters(), lr=2e-4, betas=(0.9, 0.95))
    resumed.load_state_dict(copy.deepcopy(eager_model.state_dict()))
    resumed_opt.load_state_dict(copy.deepcopy(eager.state_dict()))
    _step(eager_model, eager, inputs, target)
    _step(resumed, resumed_opt, inputs, target)
    for left, right in zip(eager_model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

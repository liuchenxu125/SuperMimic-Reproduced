"""Regression checks for zero-angle BPTT and rejection before optimizer writes."""

import pytest
import torch
from mjlab.utils.lab_api.math import quat_box_minus

from supermimic.envs.mdp.rewards import _sq_quat_err
from supermimic.kinematics.rotations import quat_from_rotvec, quat_integrate
from supermimic.training.numerics import backward_step


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_matching_orientations_have_zero_finite_gradient(sign, compiled):
    error = (
        torch.compile(_sq_quat_err, backend="aot_eager", fullgraph=True)
        if compiled
        else _sq_quat_err
    )
    q = torch.tensor([[sign, 0.0, 0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = error(q, target).sum()
    value.backward()
    assert value == 0
    torch.testing.assert_close(q.grad, torch.zeros_like(q))


def test_orientation_reward_value_matches_previous_definition():
    rng = torch.Generator().manual_seed(91)
    q1, q2 = [
        torch.nn.functional.normalize(torch.randn(200, 4, generator=rng), dim=-1) for _ in range(2)
    ]
    expected = quat_box_minus(q1, q2).square().sum(-1)
    torch.testing.assert_close(_sq_quat_err(q1, q2), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 1e-4, 0.3])
def test_orientation_error_gradcheck_near_zero(angle):
    vector = torch.tensor([[angle, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda v: _sq_quat_err(quat_from_rotvec(v), target), (vector,))


def test_rotation_bptt_through_zero_increments():
    increments = torch.zeros(32, 2, 3, requires_grad=True)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
    target = quat_from_rotvec(torch.tensor([[0.2, 0.0, 0.0]])).expand(2, -1)
    objective = 0
    for step in increments:
        quaternion = quat_integrate(quaternion, step)
        objective = objective + torch.exp(-_sq_quat_err(quaternion, target)).sum()
    objective.backward()
    assert torch.isfinite(increments.grad).all()
    assert increments.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("source", ["loss", "gradient"])
def test_invalid_update_preserves_weights_and_adam_state(bad, source):
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    backward_step(module.weight.square().sum(), module, optimizer, name="Warmup", max_norm=1.0)
    weight = module.weight.detach().clone()
    state = {key: value.clone() for key, value in optimizer.state[module.weight].items()}
    if source == "gradient":
        module.weight.register_hook(lambda grad: torch.full_like(grad, bad))
    loss = module.weight.square().sum()
    if source == "loss":
        loss = loss * bad
    with pytest.raises(FloatingPointError, match=f"Test {source}.*NaN/Inf"):
        backward_step(loss, module, optimizer, name="Test", max_norm=1.0, max_value=0.1)
    torch.testing.assert_close(module.weight, weight, atol=0, rtol=0)
    for key, value in state.items():
        torch.testing.assert_close(optimizer.state[module.weight][key], value, atol=0, rtol=0)


def test_finite_update_preserves_clipping_order():
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[3.0, 4.0]]))
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
    norm = backward_step(
        module.weight.square().sum() / 2,
        module,
        optimizer,
        name="Test",
        max_norm=1.0,
        max_value=0.1,
    )
    assert norm == 5
    torch.testing.assert_close(module.weight, torch.tensor([[2.9, 3.9]]))


@pytest.mark.parametrize("scale", [1e20, 1e37])
def test_finite_large_gradients_are_clipped_without_norm_overflow(scale):
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    module.weight.register_hook(lambda grad: torch.tensor([[3 * scale, 4 * scale]]))
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
    norm = backward_step(module.weight.sum(), module, optimizer, name="Test", max_norm=1.0)
    assert torch.isfinite(norm)
    assert float(norm) == pytest.approx(5 * scale, rel=1e-6)
    torch.testing.assert_close(module.weight, torch.tensor([[-0.6, -0.8]]))

"""A captured update replays the same Adam step and can change its learning rate in place."""

import torch

from supermimic.training.cuda_graph import CapturedUpdate, graph_state, optimize_step
from supermimic.training.optim import assign_lr, make_adam


def _run(capture, batches, lrs):
    torch.manual_seed(1)
    model = torch.nn.Linear(4, 3, device="cuda")
    opt = make_adam(model.parameters(), lr=2e-4, betas=(0.9, 0.95))
    address = opt.param_groups[0]["lr"].data_ptr()

    def step(inputs):
        loss = (model(inputs["x"]) - inputs["y"]).square().mean()
        norm, finite = optimize_step(opt, loss, model, max_norm=1.0, max_value=0.1)
        return {"loss": loss.detach(), "norm": norm.detach(), "finite": finite}

    call = step
    captured = None
    if capture:
        captured = CapturedUpdate(step, lambda: graph_state([model], [opt]), warmup=3)
        call = captured
    weights = []
    for (x, y), lr in zip(batches, lrs, strict=True):
        assign_lr(opt, lr)
        out = call({"x": x, "y": y})
        assert bool(out["finite"])
        weights.append(model.weight.detach().clone())
    assert opt.param_groups[0]["lr"].data_ptr() == address
    return weights, captured


def test_captured_update_matches_eager_adam():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    batches = [
        (torch.randn(8, 4, device="cuda"), torch.randn(8, 3, device="cuda")) for _ in range(6)
    ]
    lrs = [2e-4, 2e-4, 2e-4, 2e-4, 1e-3, 1e-3]
    eager, _ = _run(False, batches, lrs)
    graphed, captured = _run(True, batches, lrs)
    assert captured._graph is not None
    for left, right in zip(eager, graphed, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_rollout_constants_survive_cuda_graph_capture():
    if not torch.cuda.is_available():
        return
    from supermimic.kinematics.rotations import cached_constant, projected_gravity
    from supermimic.policy.nets import PolicyCfg
    from supermimic.velocity.policy import velocity_history
    from supermimic.world_model.state_spec import quat_apply_inverse_quat

    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    history = torch.zeros(2, 930, device="cuda")
    cfg = PolicyCfg(num_joints=27, obs_term_scale=(1.0, 5.0, 2.0, 1.0, 1.0, 1.0))
    noise = (0.01,) * 27 + (0.5,) * 27 + (0.2,) * 3 + (0.05,) * 3 + (0.0,) * 27 + (0.05,) * 6

    def step():
        return (
            cached_constant(noise, q),
            projected_gravity(q),
            quat_apply_inverse_quat(q),
            velocity_history(history, cfg),
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()
    graph.replay()
    assert all(torch.isfinite(value).all() for value in outputs)

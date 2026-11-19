"""Hidden Linear weights use Muon. The captured step still follows the cosine learning rate."""

import torch
from torch import nn

from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.training.cuda_graph import CapturedUpdate, graph_state, optimize_step
from supermimic.training.optim import assign_lr, make_optimizers
from supermimic.world_model.dynamics import MLP


def _ids(opt):
    return {id(param) for group in opt.param_groups for param in group["params"]}


def _split(module, **kwargs):
    opt = make_optimizers(module, **kwargs)
    muon = next(member for member in opt.members if isinstance(member, torch.optim.Muon))
    adam = next(member for member in opt.members if isinstance(member, torch.optim.Adam))
    return opt, _ids(muon), _ids(adam)


def test_hidden_linears_use_muon_and_endpoints_stay_adam():
    mlp = MLP(4, (8, 8, 8), 2, zero_last=True).cuda() if torch.cuda.is_available() else MLP(
        4, (8, 8, 8), 2, zero_last=True
    )
    opt, muon_ids, adam_ids = _split(mlp, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0)
    assert muon_ids.isdisjoint(adam_ids)
    assert muon_ids | adam_ids == {id(param) for param in mlp.parameters()}
    linears = [child for child in mlp.net if isinstance(child, nn.Linear)]
    assert [id(linear.weight) in muon_ids for linear in linears] == [False, True, True, False]
    for linear in linears:
        assert id(linear.bias) in adam_ids
    assert id(linears[0].weight) in adam_ids and id(linears[-1].weight) in adam_ids
    muon = opt.members[-1]
    assert muon.defaults["adjust_lr_fn"] == "match_rms_adamw"
    assert muon.defaults["weight_decay"] == 0.1
    assert muon.defaults["momentum"] == 0.95
    assert muon.defaults["nesterov"] is True
    assert opt.members[0].defaults["betas"] == (0.9, 0.95)
    assert opt.members[0].defaults["weight_decay"] == 0.0
    assert all(torch.is_tensor(group["lr"]) == linears[0].weight.is_cuda for group in opt.param_groups)


def test_one_hidden_layer_stays_adam():
    mlp = MLP(4, (8,), 2)
    opt = make_optimizers(mlp, lr=1e-3)
    assert len(opt.members) == 1
    assert isinstance(opt.members[0], torch.optim.Adam)


def test_policy_and_critic_follow_the_same_slice():
    cfg = PolicyCfg(
        num_joints=27,
        goal_dim=690,
        encoder_hidden=(8, 8),
        decoder_hidden=(8, 8),
        obs_term_scale=(1.0, 5.0, 2.0, 1.0, 1.0, 1.0),
    )
    policy = Policy(cfg)
    policy.encoders["velocity"] = type(policy.encoders["motion"])(cfg, goal_dim=3)
    _, muon_ids, adam_ids = _split(policy, lr=2e-4, betas=(0.9, 0.95))
    assert muon_ids | adam_ids == {id(param) for param in policy.parameters()}
    for seq in policy.modules():
        if not isinstance(seq, nn.Sequential):
            continue
        linears = [child for child in seq if isinstance(child, nn.Linear)]
        for linear in linears[1:-1]:
            assert id(linear.weight) in muon_ids
            assert id(linear.bias) in adam_ids
        assert id(linears[0].weight) in adam_ids
        assert id(linears[-1].weight) in adam_ids
    critic = Critic(CriticCfg(hidden=(8, 8)), pcfg=cfg)
    _, critic_muon, critic_adam = _split(critic, lr=2e-4)
    assert critic_muon
    assert id(critic.mlp.net[2].weight) in critic_muon
    assert id(critic.mlp.net[0].weight) in critic_adam
    assert id(critic.mlp.net[-1].weight) in critic_adam


def _match(device):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 4)
    ).to(device)
    opt = make_optimizers(model, lr=2e-4, betas=(0.9, 0.95))
    middle = model[2].weight
    official_weight = nn.Parameter(middle.detach().clone())
    official = torch.optim.Muon([official_weight], lr=2e-4, adjust_lr_fn="match_rms_adamw")
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        loss = (model(torch.randn(4, 8, device=device)) - torch.randn(4, 4, device=device))
        loss.square().mean().backward()
        official_weight.grad = middle.grad.detach().clone()
        official.step()
        opt.step()
        torch.testing.assert_close(middle, official_weight, atol=0, rtol=0, msg=f"step {step}")


def test_muon_step_matches_torch_muon_on_cpu():
    _match("cpu")


def test_muon_step_matches_torch_muon_on_cuda():
    if not torch.cuda.is_available():
        return
    _match("cuda")


def test_adam_checkpoint_does_not_load_into_muon_and_new_state_roundtrips():
    torch.manual_seed(1)
    model = MLP(4, (8, 8), 2)
    opt = make_optimizers(model, lr=2e-4)
    legacy = torch.optim.Adam(model.parameters(), lr=2e-4)
    try:
        opt.load_state_dict(legacy.state_dict())
    except ValueError as error:
        assert "Adam-only" in str(error)
    else:
        raise AssertionError("Adam checkpoint loaded into a Muon split")
    opt.zero_grad(set_to_none=True)
    model(torch.randn(2, 4)).square().mean().backward()
    opt.step()
    saved = opt.state_dict()
    again = make_optimizers(model, lr=2e-4)
    again.load_state_dict(saved)
    assert set(saved) == {"adam", "muon"}


def test_captured_muon_matches_eager_and_follows_the_learning_rate():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    batches = [
        (torch.randn(8, 4, device="cuda"), torch.randn(8, 3, device="cuda")) for _ in range(6)
    ]
    lrs = [2e-4, 2e-4, 2e-4, 2e-4, 1e-3, 1e-3]

    def run(capture):
        torch.manual_seed(3)
        model = nn.Sequential(
            nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 3)
        ).cuda()
        opt = make_optimizers(model, lr=2e-4, betas=(0.9, 0.95))
        addresses = [group["lr"].data_ptr() for group in opt.param_groups]

        def step(inputs):
            loss = (model(inputs["x"]) - inputs["y"]).square().mean()
            norm, finite = optimize_step(opt, loss, model, max_norm=1.0, max_value=0.1)
            return {"loss": loss.detach(), "norm": norm, "finite": finite}

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
            weights.append(torch.cat([param.detach().flatten() for param in model.parameters()]))
        assert [group["lr"].data_ptr() for group in opt.param_groups] == addresses
        return weights, captured

    eager, _ = run(False)
    graphed, captured = run(True)
    assert captured._graph is not None
    for left, right in zip(eager, graphed, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

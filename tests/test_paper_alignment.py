"""Paper protocol, update-clock resume, actor gradient scale and precision regressions."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig
from supermimic.eval.benchmark import clip_result, save_results, summarize
from supermimic.training.checkpoints import checkpoint_progress
from supermimic.training.precision import configure_precision
from supermimic.training.schedule import collection_steps, cosine_lr


def test_failed_clip_is_scored_over_full_length(tmp_path):
    failed = clip_result("failed", 1000, 100, 8.0, True)
    complete = clip_result("complete", 50, 50, 2.5, False)
    assert failed.norm_return == pytest.approx(0.08)
    assert failed.norm_return_survived == pytest.approx(0.8)
    result = summarize([failed, complete])
    assert result["mean_norm_return"] == pytest.approx(0.29)
    # Total earned reward 10.5 / total possible reward 105, not survived length.
    assert result["frame_weighted_norm_return"] == pytest.approx(0.1)
    assert result["failed"] == 1
    path = tmp_path / "benchmark.json"
    save_results([failed, complete], path, meta={"iteration_unit": "update"})
    import json

    saved = json.loads(path.read_text())
    assert saved["normalization"] == "full_clip"
    assert saved["checkpoint"]["iteration_unit"] == "update"


@pytest.mark.parametrize("steps,failed", [(0, True), (1001, True), (999, False)])
def test_invalid_or_early_success_does_not_hide_clock_errors(steps, failed):
    with pytest.raises(ValueError, match="evaluation length"):
        clip_result("bad", 1000, steps, 1.0, failed)


def test_lr_decay_uses_fixed_update_budget():
    assert cosine_lr(0, 2e-4) == pytest.approx(2e-4)
    assert cosine_lr(125000, 2e-4) == pytest.approx(1.03e-4)
    assert cosine_lr(250000, 2e-4) == pytest.approx(6e-6)
    assert cosine_lr(320000, 2e-4) == pytest.approx(6e-6)
    cfg = TrainConfig()
    cfg.schedule.total_iterations = 100
    assert cfg.schedule.lr_decay_iterations == 250000
    assert cosine_lr(100, 2e-4) > 1.99e-4


@pytest.mark.parametrize("utd", [0.4, 0.3, 2.0])
def test_fractional_collection_and_resume(utd):
    accumulator = 0.0
    total = 0
    for _ in range(4000):
        n, accumulator = collection_steps(accumulator, utd)
        total += n
    assert total == int(4000 / utd)
    assert 0 <= accumulator < 1


def test_legacy_units_require_explicit_migration():
    old = {"iteration": 208000, "num_updates": 83184, "config": {"schedule": {}}}
    with pytest.raises(ValueError, match="legacy-resume"):
        checkpoint_progress(old)
    progress = checkpoint_progress(old, allow_legacy=True)
    assert progress.iteration == 83184
    assert progress.env_steps == 208000
    assert progress.migrated_legacy


def test_resume_rejection_preserves_existing_run_before_simulator_init(tmp_path):
    from dataclasses import asdict

    from supermimic.training.loop import Trainer

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "config.json"
    marker.write_text("original experiment\n")
    old_path = existing / "old.pt"
    torch.save({"iteration": 1000, "num_updates": 384}, old_path)
    cfg = TrainConfig(log_root=tmp_path, run_name="existing", resume=old_path)
    with pytest.raises(ValueError, match="legacy-resume"):
        Trainer(cfg)
    assert marker.read_text() == "original experiment\n"

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    new_path = foreign / "new.pt"
    torch.save(
        {
            "iteration_unit": "update",
            "iteration": 10,
            "env_steps": 65,
            "num_updates": 10,
            "collection_accumulator": 0.0,
            "config": asdict(TrainConfig()),
        },
        new_path,
    )
    cfg.resume = new_path
    with pytest.raises(FileExistsError, match="resume its own checkpoint"):
        Trainer(cfg)
    assert marker.read_text() == "original experiment\n"


def test_precision_is_explicit_without_autocast(monkeypatch):
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    configure_precision(False)  # Recover precision state left by mjlab script imports.
    original = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        for tf32 in (True, False):
            info = configure_precision(tf32)
            assert info["cuda_matmul_allow_tf32"] is tf32
            assert info["cudnn_allow_tf32"] is tf32
            assert not info["autocast"]
            assert torch.nn.Linear(2, 2).weight.dtype == torch.float32
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
        with pytest.raises(ValueError, match="disables requested TF32"):
            configure_precision(True)
    finally:
        torch.set_float32_matmul_precision(original)
        torch.backends.cudnn.allow_tf32 = cudnn


@pytest.mark.parametrize("tf32", [False, True])
def test_environment_setup_respects_requested_precision(monkeypatch, tf32):
    import supermimic.envs.tracking_env_cfg as module

    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    configure_precision(False)
    original = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    monkeypatch.setattr(module, "make_env_cfg", lambda args: None)
    monkeypatch.setattr(module, "ManagerBasedRlEnv", lambda *args, **kwargs: object())
    try:
        # Begin with the opposite choice to catch an unconditional TF32 reset.
        configure_precision(not tf32)
        module.make_env(None, device="cpu", tf32=tf32)
        assert torch.backends.cuda.matmul.allow_tf32 is tf32
        assert torch.backends.cudnn.allow_tf32 is tf32
        assert torch.get_float32_matmul_precision() == ("high" if tf32 else "highest")
    finally:
        torch.set_float32_matmul_precision(original)
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


@pytest.mark.parametrize("tf32", [False, True])
def test_precision_recovers_from_mjlab_backend_overrides(monkeypatch, tf32):
    from mjlab.utils.torch import configure_torch_backends

    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    configure_torch_backends(allow_tf32=not tf32)
    info = configure_precision(tf32)
    assert info["float32_matmul_precision"] == ("high" if tf32 else "highest")
    assert info["cuda_matmul_allow_tf32"] is tf32
    assert info["cudnn_allow_tf32"] is tf32
    assert not info["autocast"]


def test_actor_uses_time_sum_before_gradient_clipping(monkeypatch):
    import supermimic.imagination.policy_trainer as module

    trainer = module.PolicyTrainer.__new__(module.PolicyTrainer)
    trainer.cfg = module.PolicyTrainerCfg(
        horizon=4, batch_size=1, use_critic=False, grad_clip_norm=100, grad_clip_value=100
    )
    trainer.wm = torch.nn.Linear(1, 1)
    trainer.wm.spec = SimpleNamespace(normalize_xi=lambda x: x)
    trainer.policy = torch.nn.Linear(1, 1, bias=False)
    trainer.critic = torch.nn.Linear(1, 1)
    trainer.opt = torch.optim.SGD(trainer.policy.parameters(), lr=0.0)
    trainer.step_fn, trainer.lib = None, None
    trainer._critic_inputs = lambda ro: (None, None, None)
    weight = trainer.policy.weight.reshape(())
    rollout = SimpleNamespace(
        rewards=weight.expand(4, 1),
        survival=torch.ones(5, 1),
        dhat=torch.zeros(5, 1),
        actions=weight.expand(4, 1, 1),
        reward_terms={},
    )
    monkeypatch.setattr(module, "imagine", lambda *args, **kwargs: rollout)
    buffer = SimpleNamespace(sample_starts=lambda *args: SimpleNamespace(xi=torch.zeros(1, 1)))
    trainer.update(buffer)
    expected = -sum(trainer.cfg.gamma**t for t in range(4))
    assert trainer.policy.weight.grad.item() == pytest.approx(expected)
    assert all(p.requires_grad for p in trainer.wm.parameters())


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = (step, float(value))

    def flush(self):
        pass


class Environment:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return {}, {}

    def step(self, action):
        self.steps += 1
        return (
            {},
            torch.full((2,), 0.05),
            torch.zeros(2, dtype=torch.bool),
            torch.full((2,), self.steps % 4 == 0, dtype=torch.bool),
            {},
        )


class Updater:
    def __init__(self, model, **metrics):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        self.metrics = metrics

    def set_lr(self, lr):
        self.opt.param_groups[0]["lr"] = lr

    def update(self, buffer):
        self.opt.zero_grad()
        sum(p.square().sum() for p in self.model.parameters()).backward()
        self.opt.step()
        return self.metrics

    def online_error(self, *args, **kwargs):
        return torch.zeros(8)


def cpu_trainer(path, iterations, policy_per_outer=1):
    """Real orchestration + Adam/checkpoint I/O; deterministic fake physics only."""
    from supermimic.training.loop import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig()
    trainer.cfg.schedule.total_iterations = iterations
    trainer.cfg.schedule.warmup_iterations = 2
    trainer.cfg.schedule.checkpoint_interval = 5
    trainer.cfg.schedule.log_interval = 1
    trainer.cfg.schedule.policy_updates_per_wm_update = policy_per_outer
    trainer.log_dir = path
    path.mkdir()
    trainer.device, trainer.is_main, trainer.world = "cpu", True, 1
    trainer.env, trainer.writer = Environment(), Writer()
    trainer.buffer = SimpleNamespace(size=4096)
    torch.manual_seed(13)
    trainer.wm, trainer.policy, trainer.critic = [torch.nn.Linear(1, 1) for _ in range(3)]
    trainer.wm_trainer = Updater(trainer.wm, nse_1step=0.0)
    trainer.policy_trainer = Updater(trainer.policy, imag_return_undisc=1.0, imag_survival_H=1.0)
    trainer.policy_trainer.critic_opt = torch.optim.Adam(trainer.critic.parameters())
    trainer.command = SimpleNamespace(
        bin_return=torch.ones(1),
        bin_memory=torch.ones(1),
        metrics={"sampling_entropy": torch.zeros(1)},
    )
    trainer.ep_reward_sum, trainer.ep_len = torch.zeros(2), torch.zeros(2)
    trainer.max_step_reward = 0.1
    trainer.iteration = trainer.num_updates = trainer.env_steps = (
        trainer.replay_refill_env_steps
    ) = 0
    trainer._collection_acc = trainer._time_offset = trainer._train_seconds = 0.0
    trainer.precision, trainer.resume_migration = {}, None
    trainer.act = lambda obs: torch.zeros(2, 1)
    return trainer


def test_update_clock_checkpoint_and_resume_preserve_fraction_and_lr(tmp_path):
    baseline = cpu_trainer(tmp_path / "baseline", 7)
    baseline.run()
    partial = cpu_trainer(tmp_path / "partial", 5)
    partial.run()
    checkpoint = torch.load(partial.log_dir / "ckpt_latest.pt", weights_only=False)
    assert checkpoint["iteration_unit"] == "update"
    assert checkpoint["iteration"] == checkpoint["num_updates"] == 5
    assert checkpoint["env_steps"] == 14  # 2 refill + 12 data-collection steps
    assert checkpoint["collection_accumulator"] == 0.5
    resumed = cpu_trainer(tmp_path / "resumed", 7)
    resumed.load(partial.log_dir / "ckpt_latest.pt")
    resumed.run()
    assert resumed.iteration == resumed.num_updates == 7
    assert resumed.env_steps == baseline.env_steps + 2  # explicit additional replay refill
    assert resumed._collection_acc == baseline._collection_acc
    assert (
        resumed.wm_trainer.opt.param_groups[0]["lr"]
        == baseline.wm_trainer.opt.param_groups[0]["lr"]
    )
    for a, b in zip(resumed.policy.parameters(), baseline.policy.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not list(resumed.log_dir.glob("*.tmp"))


def test_policy_update_count_and_logging_with_multiple_updates(tmp_path):
    trainer = cpu_trainer(tmp_path / "multiple", 3, policy_per_outer=2)
    trainer.run()
    assert trainer.iteration == 3
    assert trainer.num_updates == 6
    assert trainer.writer.values["policy/imag_return_undisc"] == (3, 1.0)

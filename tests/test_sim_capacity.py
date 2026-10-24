"""A substep overflow must remain visible even when the following reset clears it."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from mujoco_warp import OverflowType

spec = importlib.util.spec_from_file_location(
    "validate_sim_capacity",
    Path(__file__).resolve().parents[1] / "scripts/validate_sim_capacity.py",
)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def test_capacity_monitor_retains_transient_overflow_and_separates_solver_limits(monkeypatch):
    monitor = capacity.CapacityMonitor.__new__(capacity.CapacityMonitor)
    monitor.data = SimpleNamespace(nworld=2, naconmax=192, njmax=320, njmax_pad=320)
    monitor.env = SimpleNamespace(sim=SimpleNamespace(wp_model=SimpleNamespace(is_sparse=True)))
    monitor.nefc = torch.tensor([320, 319])
    monitor.nacon = torch.tensor([192])
    monitor.ncollision = torch.tensor([192])
    monitor.overflow = torch.tensor([int(OverflowType.ITERATIONS), int(OverflowType.LS_ITERATIONS)])
    monitor.bits = list(OverflowType)
    monitor.bit_values = torch.tensor([int(bit) for bit in monitor.bits])
    monitor.flags = torch.zeros(2, dtype=torch.int32)
    monitor.bit_world_samples = torch.zeros(len(monitor.bits), dtype=torch.int64)
    monitor.peaks = torch.zeros(3, dtype=torch.int64)
    monitor.capacity_exceed_samples = torch.zeros(3, dtype=torch.int64)
    monitor.calls = {"step": 0, "forward": 0}
    monitor.stream = None
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    monitor.sample("step")
    assert monitor.report()["capacity_passed"]
    monitor.nefc[0] = 321
    monitor.nacon[0] = 193
    monitor.overflow[0] |= int(OverflowType.NEFC | OverflowType.NARROWPHASE)
    monitor.sample("step")

    # Mimic a reset and subsequent forward replacing the transient counters.
    for value in (monitor.nefc, monitor.nacon, monitor.ncollision, monitor.overflow):
        value.zero_()
    monitor.sample("forward")
    report = monitor.report()
    assert not report["capacity_passed"]
    assert report["max_constraints_per_world"] == 321
    assert report["max_contacts_shared"] == 193
    assert report["raw_exceed_samples"] == {
        "contact_pool": 1,
        "broadphase_pool": 0,
        "constraints": 1,
    }
    assert report["overflow_worlds_by_bit"]["NEFC"] == 1

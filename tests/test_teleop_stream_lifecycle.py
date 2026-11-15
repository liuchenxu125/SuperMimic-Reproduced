"""Regression checks for idle polls, misleading calibration status and interrupted capture exit."""

import importlib.util
import signal
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/teleop/stream_pico.py"
    spec = importlib.util.spec_from_file_location("teleop_stream_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_healthy_idle_poll_preserves_calibrated_status(script):
    common = {
        "calibrated": True,
        "last_received": 10.0,
        "now": 10.01,
        "frame": 100,
        "ik": {"max_position_error_m": 0.01},
        "sent_windows": 91,
    }
    assert script.stream_report(active=True, **common)["mode"] == "POSE"
    assert script.stream_report(active=False, **common)["mode"] == "HOLD"
    assert script.stream_report(active=True, **common)["calibrated"]
    assert script.stream_report(active=True, **{**common, "now": 10.5})["mode"] == "STREAM_GAP"
    assert (
        script.stream_report(active=True, **{**common, "calibrated": False})["mode"]
        == "WAIT_CALIBRATION"
    )


def test_capture_committed_before_sdk_close_interrupt(script, tmp_path):
    path = tmp_path / "record.npz"
    samples = np.zeros((2, 3, 7))
    samples[:, :, 6] = 1.0
    events = []

    class Sock:
        def close(self):
            events.append("socket")

    class SDK:
        def close(self):
            assert path.is_file()  # reproduces the failing old shutdown order
            events.append("sdk")
            raise KeyboardInterrupt

    script.close_session(SDK(), Sock(), path, samples, [1, 2], "upper-body")
    assert events == ["socket", "sdk"]
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["poses"], samples)
        assert data["input_mode"].item() == "upper-body"
    assert not path.with_name(path.name + ".tmp.npz").exists()


def test_repeated_ctrl_c_during_atomic_save_is_ignored_and_handler_restored(
    script, tmp_path, monkeypatch
):
    original_handler = signal.getsignal(signal.SIGINT)
    save = np.savez_compressed

    def interrupting_save(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        return save(*args, **kwargs)

    monkeypatch.setattr(script.np, "savez_compressed", interrupting_save)

    class Sock:
        def close(self):
            pass

    path = tmp_path / "full.npz"
    script.close_session(None, Sock(), path, np.zeros((2, 24, 7)), [1, 2], "full-body")
    assert signal.getsignal(signal.SIGINT) == original_handler
    with np.load(path, allow_pickle=False) as data:
        assert data["body"].shape == (2, 24, 7)

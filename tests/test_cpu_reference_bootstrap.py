"""The CPU CLI must configure native thread pools before importing Torch."""
import json
import os
import subprocess
import sys

import pytest

from supermimic.training.cpu_runtime import CPU4_THREAD_ENV, configure_cpu_reference_environment


@pytest.mark.parametrize("argv", [["--env.library-device", "cpu"], ["--env.library-device=cpu"]])
def test_cpu_reference_overrides_all_three_environment_limits(argv):
    environment = {key: "24" for key in CPU4_THREAD_ENV}
    environment["UNCHANGED"] = "value"
    assert configure_cpu_reference_environment(argv, environment)
    assert {key: environment[key] for key in CPU4_THREAD_ENV} == {key: "1" for key in CPU4_THREAD_ENV}
    assert environment["UNCHANGED"] == "value"


@pytest.mark.parametrize("argv", [[], ["--env.library-device", "device"], ["--data.npz-dir", "cpu"]])
def test_other_reference_jobs_keep_existing_environment(argv):
    environment = {key: "7" for key in CPU4_THREAD_ENV}
    before = environment.copy()
    assert not configure_cpu_reference_environment(argv, environment)
    assert environment == before


def test_cli_bootstraps_before_real_torch_import():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update({key: "24" for key in CPU4_THREAD_ENV})
    code = r'''
import json, os, runpy, sys
assert "torch" not in sys.modules
sys.argv = ["sm-train", "--env.library-device=cpu", "--help"]
try:
    runpy.run_module("supermimic.scripts.train", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("Expected --help to exit without starting training")
import torch
from supermimic.config import TrainConfig
print("BOOTSTRAP_RESULT=" + json.dumps({
    "env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    "cpu_threads": torch.get_num_threads(),
    "transfer": TrainConfig().env.library_transfer,
}))
'''
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("BOOTSTRAP_RESULT="))
    actual = json.loads(line.split("=", 1)[1])
    assert actual["env"] == {key: "1" for key in CPU4_THREAD_ENV}
    assert actual["cpu_threads"] == 1
    assert actual["transfer"] == "pinned"

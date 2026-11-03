import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supermimic.data.geometry_cleaning import (
    collision_depths,
    collision_model_with_floor,
    compatible_spans,
)
from supermimic.data.temporal_cleaning import rebuild_span
from supermimic.robots import get_robot


def test_geometry_cuts_never_bridge_invalid_reference_frames():
    ground, own = np.zeros(170), np.zeros(170)
    ground[:5] = 0.18
    own[70:80] = 0.07
    assert compatible_spans(ground, own) == [(5, 70), (80, 170)]
    assert compatible_spans(ground, own, trim_prefix=25) == [(80, 170)]
    assert compatible_spans(np.ones(60), np.zeros(60)) == []
    assert compatible_spans(np.zeros(51), np.zeros(51), trim_prefix=5) == []


def test_compatible_spans_reject_nonfinite_diagnostics():
    with pytest.raises(ValueError, match="Nonfinite"):
        compatible_spans(np.array([0.0, np.nan, 0.0]), np.zeros(3), min_frames=3)


def motion_arrays():
    profile = get_robot("casbot02")
    model = collision_model_with_floor()
    n = 65
    joints = np.tile(
        [
            profile.motor_parameters()["default_joint_pos"][model.joint(i).name]
            for i in range(1, model.njnt)
        ],
        (n, 1),
    ).astype(np.float32)
    root = np.tile([0.0, 0.0, 1.1], (n, 1)).astype(np.float32)
    root[:5, 2] = 0.3  # Clearly underground under this model, then a valid stationary pose.
    arrays = {
        "fps": np.array([50.0], dtype=np.float32),
        "joint_pos": joints,
        "joint_vel": np.zeros_like(joints),
        "body_names": np.array(["base_link"]),
        "body_pos_w": root[:, None],
        "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1, 1)).astype(np.float32),
        "body_lin_vel_w": np.zeros((n, 1, 3), np.float32),
        "body_ang_vel_w": np.zeros((n, 1, 3), np.float32),
        "joint_names": np.array([model.joint(i).name for i in range(1, model.njnt)]),
    }
    return model, arrays


def test_native_geometry_and_rebuild_preserve_compact_target_poses():
    model, arrays = motion_arrays()
    ground, own = collision_depths(arrays, model)
    assert ground[:5].min() > 0.03
    spans = compatible_spans(ground, own)
    assert spans == [(5, 65)]
    repaired = rebuild_span(arrays, model, *spans[0])
    np.testing.assert_array_equal(repaired["joint_pos"], arrays["joint_pos"][5:])
    np.testing.assert_allclose(repaired["body_lin_vel_w"], 0, atol=1e-6)
    new_ground, new_own = collision_depths(repaired, model)
    assert max(new_ground.max(), new_own.max()) <= 0.03


def test_variant_install_keeps_parent_data_and_manifests_unchanged(tmp_path):
    from supermimic.data.motion_convert import write_motion_npz

    model, arrays = motion_arrays()
    source = tmp_path / "source.csv"
    source.write_text("original CSV is preserved\n")
    arrays["source"] = np.array(str(source))
    dataset, audit = tmp_path / "parent", tmp_path / "audit"
    dataset.mkdir()
    (audit / "frame_geometry").mkdir(parents=True)
    write_motion_npz(dataset / "bad.npz", arrays)
    parent_bytes = (dataset / "bad.npz").read_bytes()
    healthy = rebuild_span(arrays, model, 5, 65)
    write_motion_npz(dataset / "healthy.npz", healthy)
    manifest = dataset / "manifest.csv"
    rows = [
        {
            "name": name,
            "frames": frames,
            "keep": 1,
            "reason": "",
            "source": str(source),
            "source_bytes": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
            "npz_bytes": 0,
        }
        for name, frames in [("bad", 65), ("healthy", 60)]
    ]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    old_manifest = manifest.read_bytes()
    ground, own = collision_depths(arrays, model)
    np.savez_compressed(audit / "frame_geometry/bad.npz", ground_depth=ground, self_depth=own)
    (audit / "full_geometry.json").write_text(
        json.dumps([{"name": "bad", "bad_before_original_failure": True}])
    )
    (audit / "verified_source_startup_outliers.json").write_text('[{"name":"bad"}]')
    (audit / "verified_boundary_outliers.json").write_text("[]")
    script = Path(__file__).resolve().parents[1] / "scripts/data/repair_motion_starts.py"
    work, output = tmp_path / "work", tmp_path / "variant"
    base = [sys.executable, str(script), "--work-dir", str(work)]
    subprocess.run(
        base
        + [
            "--dataset",
            str(dataset),
            "--manifest",
            str(manifest),
            "--audit-dir",
            str(audit),
            "--output",
            str(output),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not output.exists()
    subprocess.run(base + ["--apply"], check=True, capture_output=True, text=True)
    assert (dataset / "bad.npz").read_bytes() == parent_bytes
    assert (work / "backup/bad.npz").read_bytes() == parent_bytes
    assert manifest.read_bytes() == old_manifest
    assert source.read_text() == "original CSV is preserved\n"
    assert (output / "healthy.npz").stat().st_ino == (dataset / "healthy.npz").stat().st_ino
    with np.load(output / "bad__startfix01.npz") as f:
        assert len(f["joint_pos"]) == 60
        np.testing.assert_array_equal(f["joint_pos"], arrays["joint_pos"][5:])
    plan = json.loads((work / "plan.json").read_text())
    assert plan["applied"] and plan["after_clips"] == 2 and plan["parent_data_unchanged"]
    retry = subprocess.run(base + ["--apply"], check=False, capture_output=True, text=True)
    assert retry.returncode != 0 and "Already applied" in retry.stderr

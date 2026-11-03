"""Temporal cuts must not leak jumps into derivatives or change joint angles."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supermimic.data.motion_convert import ResampledMotion, forward_kinematics
from supermimic.data.temporal_cleaning import clean_spans, jump_edges, rebuild_span
from supermimic.robots import get_robot


@pytest.mark.parametrize(
    "edges,expected,method",
    [
        ([0], [(5, 200)], "trim_0.1s"),
        ([3], [(5, 200)], "trim_0.1s"),
        ([4], [(25, 200)], "trim_0.5s"),
        ([23], [(25, 200)], "trim_0.5s"),
        ([24], [(25, 200)], "split_at_jumps"),
        ([99], [(0, 100), (100, 200)], "split_at_jumps"),
        ([0, 99, 198], [(1, 100), (100, 199)], "split_at_jumps"),
        (list(range(199)), [], "split_at_jumps"),
        ([], [(0, 200)], "unchanged"),
    ],
)
def test_cut_edges_and_discard_only_short_contiguous_spans(edges, expected, method):
    assert clean_spans(200, edges, 50, 50) == (expected, method)


def compact_motion():
    robot = get_robot("casbot02")
    model = robot.model(collision_model="primitive")
    n, fps = 120, 50.0
    joints = np.zeros((n, 27))
    joints[60:, 0] = 1.5
    root = np.tile([0.0, 0.0, 0.9], (n, 1))
    root[:, 0] = np.arange(n) * 0.002
    root[60:, 0] += 3
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    quat[::2] *= -1  # Sign flips must not be mistaken for orientation changes.
    motion = ResampledMotion(
        fps,
        root,
        quat,
        joints,
        np.gradient(root, 1 / fps, axis=0),
        np.zeros((n, 3)),
        np.gradient(joints, 1 / fps, axis=0),
    )
    body = forward_kinematics(model, motion)
    ids = [body.body_names.index(name) for name in robot.tracked_bodies]
    arrays = {
        "fps": np.array([fps], dtype=np.float32),
        "joint_pos": joints.astype(np.float32),
        "joint_vel": motion.joint_vel.astype(np.float32),
        "body_pos_w": body.pos_w[:, ids].astype(np.float32),
        "body_quat_w": body.quat_w[:, ids].astype(np.float32),
        "body_lin_vel_w": body.lin_vel_w[:, ids].astype(np.float32),
        "body_ang_vel_w": body.ang_vel_w[:, ids].astype(np.float32),
        "body_names": np.array(robot.tracked_bodies),
        "joint_names": np.array([model.joint(i).name for i in range(1, model.njnt)]),
        "source": np.array("preserved.csv"),
    }
    return model, arrays


def test_compact_motion_rebuild_preserves_poses_and_local_derivatives():
    model, arrays = compact_motion()
    np.testing.assert_array_equal(jump_edges(arrays), [59])
    spans, _ = clean_spans(120, jump_edges(arrays), 50, 50)
    assert spans == [(0, 60), (60, 120)]
    for start, end in spans:
        rebuilt = rebuild_span(arrays, model, start, end)
        assert not len(jump_edges(rebuilt))
        np.testing.assert_array_equal(rebuilt["joint_pos"], arrays["joint_pos"][start:end])
        np.testing.assert_array_equal(rebuilt["body_names"], arrays["body_names"])
        np.testing.assert_allclose(rebuilt["joint_vel"], 0, atol=1e-7)
        np.testing.assert_allclose(rebuilt["body_lin_vel_w"][:, 0, 0], 0.1, atol=1e-5)
        assert rebuilt["source"].item() == "preserved.csv"
    assert np.max(abs(arrays["joint_vel"])) > 30  # Original arrays weren't changed.


def test_staged_install_preserves_original_bytes_and_updates_manifest(tmp_path):
    from supermimic.data.motion_convert import write_motion_npz

    model, arrays = compact_motion()
    dataset = tmp_path / "data"
    dataset.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("original source is not edited\n")
    arrays["source"] = np.array(str(source))
    write_motion_npz(dataset / "broken.npz", arrays)
    original_bytes = (dataset / "broken.npz").read_bytes()
    healthy = rebuild_span(arrays, model, 0, 60)
    write_motion_npz(dataset / "healthy.npz", healthy)
    unchanged_bytes = (dataset / "healthy.npz").read_bytes()
    manifest = dataset / "manifest_filtered.csv"
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
        for name, frames in [("broken", 120), ("healthy", 60)]
    ]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    old_manifest = manifest.read_bytes()
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "name,frames,jump_count,first_jump_frame,last_jump_frame\nbroken,120,1,59,59\n"
    )
    work = tmp_path / "repair"
    script = Path(__file__).resolve().parents[1] / "scripts/data/clean_temporal_jumps.py"
    base = [sys.executable, str(script), "--work-dir", str(work)]
    subprocess.run(
        base
        + [
            "--dataset",
            str(dataset),
            "--manifest",
            str(manifest),
            "--audit",
            str(audit),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (dataset / "broken.npz").read_bytes() == original_bytes
    assert manifest.read_bytes() == old_manifest
    subprocess.run(base + ["--apply"], check=True, capture_output=True, text=True)
    assert not (dataset / "broken.npz").exists()
    assert (dataset / "healthy.npz").read_bytes() == unchanged_bytes
    assert (work / "backup/broken.npz").read_bytes() == original_bytes
    assert (work / "backup/manifest_filtered.csv").read_bytes() == old_manifest
    assert source.read_text() == "original source is not edited\n"
    plan = json.loads((work / "plan.json").read_text())
    assert plan["applied"] and plan["after_clips"] == 3
    for part in (dataset / "broken__clean_part01.npz", dataset / "broken__clean_part02.npz"):
        with np.load(part) as f:
            assert len(f["joint_pos"]) == 60
            assert not len(jump_edges(f))
            assert (
                f["temporal_original_sha256"].item() == hashlib.sha256(original_bytes).hexdigest()
            )
    # A second apply cannot accidentally cut the already-cleaned files again.
    retry = subprocess.run(base + ["--apply"], check=False, capture_output=True, text=True)
    assert retry.returncode != 0 and "refusing to trim twice" in retry.stderr

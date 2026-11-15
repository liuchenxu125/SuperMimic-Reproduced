"""Human input math, strict pairing and independent optimizer/checkpoint contracts."""

import csv
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from supermimic.human.constants import HUMAN_GOAL_DIM, HUMAN_JOINTS, HUMAN_SOMA_SOURCE
from supermimic.human.data import REST, Pair, load_pair, split_pairs
from supermimic.human.features import (
    fixed_human_masks,
    heading_features,
    human_features,
    mask_human_goal,
)
from supermimic.human.model import RunningNormalizer
from supermimic.human.teacher import digest
from supermimic.human.training import HumanTrainConfig, HumanTrainer
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.teleop.reference import active_sim_bundle, load_contract
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

ROOT = Path(__file__).resolve().parents[1]


def write_fixture(directory, *, tiny=True):
    """Synthetic fixture only, never a replacement for SEED human data."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original_bundle = active_sim_bundle(ROOT)
    meta = load_contract(original_bundle)
    meta["model_file"] = str(original_bundle / meta["model_file"])
    pcfg = PolicyCfg(num_joints=27, goal_dim=690, obs_term_scale=(1.0, 5.0, 2.0, 1.0, 1.0, 1.0))
    if tiny:
        pcfg.encoder_hidden = pcfg.decoder_hidden = (16, 12)
    teacher = Policy(pcfg)
    from dataclasses import asdict

    wc = WorldModelCfg(
        hidden=(16, 16) if tiny else (2048, 2048, 2048),
        term_hidden=(8,),
        state_spec=StateSpecCfg(num_joints=27),
    )
    wm = WorldModel(wc, heads=None, device="cpu")
    ckpt = directory / "teacher.pt"
    torch.save(
        {
            "policy": teacher.state_dict(),
            "wm": wm.state_dict(),
            "config": {
                "robot": "casbot02",
                "policy": asdict(pcfg),
                "world_model": asdict(wc),
                "env": {"anchor_obs": False, "anchor_ori_obs": True},
                "tf32": False,
            },
        },
        ckpt,
    )
    meta["checkpoint_sha256"] = digest(ckpt)
    meta["checkpoint_path"] = str(ckpt)
    bundle = directory / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "manifest.json").write_text(json.dumps(meta))
    model = mujoco.MjModel.from_xml_path(meta["model_file"])
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[[0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0], [m["q0"] for m in meta["motors"]]]
    mujoco.mj_forward(model, data)
    n = 80
    robot_path, human_path = directory / "robot.npz", directory / "human.npz"
    np.savez(
        robot_path,
        fps=50.0,
        joint_names=meta["joint_names"],
        body_names=[model.body(i).name.removeprefix("robot/") for i in range(1, model.nbody)],
        joint_pos=np.tile(data.qpos[7:], (n, 1)),
        joint_vel=np.zeros((n, 27)),
        body_pos_w=np.tile(data.xpos[None, 1:], (n, 1, 1)),
        body_quat_w=np.tile(data.xquat[None, 1:], (n, 1, 1)),
        body_lin_vel_w=np.zeros((n, model.nbody - 1, 3)),
        body_ang_vel_w=np.zeros((n, model.nbody - 1, 3)),
    )
    positions = np.zeros((n, 24, 3))
    positions[..., 2] = 1.0
    positions[:, 7:9, 2] = 0.08
    positions[:, 15, 2] = 1.7
    positions[:, 20] = [0.1, 0.6, 1.0]
    positions[:, 21] = [0.1, -0.6, 1.0]
    np.savez(
        human_path,
        fps=50.0,
        human_joint_names=HUMAN_JOINTS,
        human_pos_w=positions,
        human_quat_w=np.tile(REST.as_quat(scalar_first=True), (n, 24, 1)),
    )
    manifest = directory / "pairs.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "robot_npz", "human_npz"])
        writer.writeheader()
        writer.writerow(
            {"name": "synthetic_stand", "robot_npz": str(robot_path), "human_npz": str(human_path)}
        )
    return ckpt, bundle, manifest, meta


def test_human_heading_masks_and_dimensions():
    points, frames, heading = fixed_human_masks(2, "five", 10, torch.device("cpu"))
    quat = (
        torch.tensor(REST.as_quat(scalar_first=True), dtype=torch.float32)
        .expand(2, 10, 24, 4)
        .clone()
    )
    robot = torch.tensor([1.0, 0, 0, 0]).expand(2, 4)
    rel = heading_features(robot, quat, points)
    np.testing.assert_allclose(rel.numpy(), np.tile([1.0, 0, 0, 1.0, 0, 0], (2, 10)), atol=1e-6)
    pos = torch.zeros(2, 10, 24, 3)
    pos[..., 2] = 1.0
    original = human_features(pos, quat, points)
    pos[..., 0] += 3.0
    pos[..., 1] -= 4.0
    torch.testing.assert_close(human_features(pos, quat, points), original)
    feats = torch.randn(2, 2160)
    packed = mask_human_goal(feats, points, frames, rel, heading)
    assert packed.shape == (2, HUMAN_GOAL_DIM)
    blocks = packed[:, :2160].reshape(2, 10, 24, 9)
    assert not blocks[:, :, ~points[0]].any()


def test_normalizer_matches_population_moments_and_freezes():
    norm = RunningNormalizer(3)
    a, b = torch.randn(5, 3), torch.randn(7, 3)
    norm.update(a)
    norm.update(b)
    all_rows = torch.cat([a, b])
    torch.testing.assert_close(norm.mean, all_rows.mean(0))
    torch.testing.assert_close(norm.var, all_rows.var(0, unbiased=False))
    saved = norm.mean.clone()
    norm.eval()
    norm.update(a + 10)
    torch.testing.assert_close(norm.mean, saved)


def test_pairing_rejects_frame_mismatch_and_bad_joint_order(tmp_path):
    _, _, _, meta = write_fixture(tmp_path)
    pair = Pair("synthetic", tmp_path / "robot.npz", tmp_path / "human.npz")
    r, h = load_pair(pair, meta)
    assert r.shape == (80, 67) and h.shape == (80, 24, 7)
    with pytest.raises(ValueError, match="frame mismatch"):
        load_pair(Pair(pair.name, pair.robot_npz, pair.human_npz, human_start_frame=1), meta)
    r, _ = load_pair(
        Pair(pair.name, pair.robot_npz, pair.human_npz, human_start_frame=1, length=79), meta
    )
    assert len(r) == 79
    meta["joint_names"] = list(reversed(meta["joint_names"]))
    with pytest.raises(ValueError, match="joint order"):
        load_pair(pair, meta)


def test_validation_keeps_mirrored_clips_together():
    pairs = [Pair(n, Path("r"), Path("h")) for n in ("walk", "walk_M", "dance", "dance_M", "stand")]
    train, val = split_pairs(pairs, 0.2)
    assert {p.name.removesuffix("_M") for p in train}.isdisjoint(
        {p.name.removesuffix("_M") for p in val}
    )


def test_independent_copy_step_freezes_original_policy_and_resumes(tmp_path):
    torch.set_num_threads(1)
    ckpt, bundle, pairs, _ = write_fixture(tmp_path / "data")
    cfg = HumanTrainConfig(
        pairs,
        tmp_path / "run",
        teacher_ckpt=ckpt,
        bundle=bundle,
        device="cpu",
        iterations=2,
        batch_size=2,
        horizon=2,
        obs_noise=False,
        validation_fraction=0,
        log_every=1,
    )
    trainer = HumanTrainer(cfg)
    # Exported scenes contain terrain bodies; torso feedback must still match
    # MuJoCo world FK for tilted roots and a turned waist.
    state = torch.zeros(2, 67)
    state[:, 2] = 0.9
    state[:, 3:7] = torch.tensor(
        Rotation.from_euler("xyz", [[0.1, -0.2, 0.3], [-0.1, 0.15, -0.2]]).as_quat(
            scalar_first=True
        ),
        dtype=torch.float32,
    )
    state[:, 7:34] = trainer.teacher.q0
    state[:, 7 + 12] = torch.tensor([0.2, -0.3])
    model = mujoco.MjModel.from_xml_path(trainer.teacher.meta["model_file"])
    data = mujoco.MjData(model)
    actual = trainer.teacher.anchor_quat(state).numpy()
    for i in range(2):
        data.qpos[:] = state[i, :34].numpy()
        mujoco.mj_forward(model, data)
        expected = data.xquat[model.body("robot/waist_yaw_link").id]
        np.testing.assert_allclose(actual[i], expected, atol=1e-6)
    obs, goal = torch.randn(2, 930), torch.randn(2, 690)
    with torch.no_grad():
        old_actions = trainer.teacher.policy(obs, goal)
    old_human = [p.detach().clone() for p in trainer.human.parameters()]
    parent_digest = digest(ckpt)
    metrics = trainer.step()
    trainer.iteration = 1
    assert metrics["obs_dim"] == 930 and metrics["human_goal_dim"] == 2254
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(old_human, trainer.human.parameters()))
    with torch.no_grad():
        torch.testing.assert_close(
            trainer.policy(obs, goal, interface="motion"), old_actions, atol=0, rtol=0
        )
    trainer.assert_frozen()
    assert digest(ckpt) == parent_digest
    trainer.save("human_latest.pt")
    export_spec = importlib.util.spec_from_file_location(
        "human_export", ROOT / "scripts/human/export_human.py"
    )
    export_module = importlib.util.module_from_spec(export_spec)
    export_spec.loader.exec_module(export_module)
    export_meta = export_module.export(tmp_path / "run/human_latest.pt", tmp_path / "export")
    assert export_meta["obs_dim"] == 930 and export_meta["trace_max_abs_error"] < 2e-6
    cfg.resume = tmp_path / "run/human_latest.pt"
    resumed = HumanTrainer(cfg)
    assert resumed.iteration == 1
    assert all(
        torch.equal(a, b) for a, b in zip(trainer.human.parameters(), resumed.human.parameters())
    )
    # Saved sampler/RNG state reproduces the next training update exactly.
    expected = resumed.step()
    resumed_again = HumanTrainer(cfg)
    actual = resumed_again.step()
    assert expected["loss"] == actual["loss"]


def test_soma_bvh_units_rest_frames_and_120_to_50hz(tmp_path):
    from supermimic.human.bvh import write_bvh
    from supermimic.human.soma import SOMA_REST_QUAT, convert_human_clip

    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    # BVH hierarchy is depth first, not SMPL slot order. Use a parent-before-child
    # DFS order and remap every joint, as a real SOMA file does.
    order = []

    def visit(i):
        order.append(i)
        for j, p in enumerate(parents):
            if p == i:
                visit(j)

    visit(0)
    remap = {old: new for new, old in enumerate(order)}
    names = [HUMAN_SOMA_SOURCE[i] for i in order]
    parent_order = [-1 if parents[i] < 0 else remap[parents[i]] for i in order]
    offsets = [np.array([0.0, 100.0, 0.0]) if i == 0 else np.array([0.0, 10.0, 0.0]) for i in order]
    channels = [["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"]]
    channels += [["Zrotation", "Xrotation", "Yrotation"] for _ in names[1:]]
    rest = Rotation.from_quat(SOMA_REST_QUAT, scalar_first=True)
    local_angles = [
        (rest[i] if parents[i] < 0 else rest[parents[i]].inv() * rest[i]).as_euler(
            "ZXY", degrees=True
        )
        for i in order
    ]
    frame = np.r_[[0.0, 100.0, 0.0], np.concatenate(local_angles)]
    frames = np.tile(frame, (13, 1))
    frames[:, 0] = np.arange(13)
    source, target = tmp_path / "fixture.bvh", tmp_path / "human.npz"
    write_bvh(source, names, parent_order, offsets, channels, frames, 1 / 120)
    convert_human_clip(str(source), str(target))
    with np.load(target) as data:
        assert data["human_pos_w"].shape == (6, 24, 3)
        np.testing.assert_allclose(data["human_pos_w"][1, 0], [0.024, 0.0, 1.0], atol=1e-6)
        expected = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        actual = Rotation.from_quat(data["human_quat_w"][0], scalar_first=True).as_matrix()
        np.testing.assert_allclose(actual, np.tile(expected, (24, 1, 1)), atol=2e-6)


def test_archive_subset_ignores_unselected_and_links(tmp_path):
    import io
    import tarfile

    spec = importlib.util.spec_from_file_location(
        "human_extract", ROOT / "scripts/human/extract_seed_subset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "motions.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        link = tarfile.TarInfo("soma_uniform/bvh/day/wanted.bvh")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
        for name in ("soma_uniform/bvh/day/wanted.bvh", "soma_uniform/bvh/day/other.bvh"):
            entry = tarfile.TarInfo(name)
            entry.size = 4
            tar.addfile(entry, io.BytesIO(b"data"))
    report = module.extract(archive, tmp_path / "selected", ["wanted"])
    assert not report["missing"]
    assert (tmp_path / "selected/day/wanted.bvh").read_bytes() == b"data"
    assert not (tmp_path / "selected/day/other.bvh").exists()

"""Casbot02 asset fidelity, dimensions, FK, library parity and short training."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.robots import get_robot
from supermimic.robots.profiles import CASBOT_ASSETS


def test_copied_assets_and_motor_mapping():
    from mjlab.entity import Entity

    profile = get_robot("casbot02")
    for name, digest in json.loads(
        (CASBOT_ASSETS / "copied_files_sha256.json").read_text()
    ).items():
        assert hashlib.sha256((CASBOT_ASSETS / name).read_bytes()).hexdigest() == digest
    motors = profile.motor_parameters()["motors"]
    entity = Entity(profile.entity_cfg())
    model = entity.spec.compile()
    assert (model.nq, model.nv, model.nbody, model.nu) == (34, 33, 31, 27)
    assert abs(model.body_mass.sum() - 50.653931) < 1e-6
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        name = model.joint(joint).name
        p = motors[name]
        assert model.actuator_gainprm[actuator, 0] == p["stiffness"]
        assert model.actuator_biasprm[actuator, 2] == -p["damping"]
        np.testing.assert_allclose(
            model.actuator_forcerange[actuator], [-p["effort_limit"], p["effort_limit"]]
        )
        dof = model.jnt_dofadr[joint]
        assert model.dof_armature[dof] == p["armature"]
        assert model.dof_damping[dof] == model.dof_frictionloss[dof] == 0
        assert profile.action_scale()[name] == 0.25 * p["effort_limit"] / p["stiffness"]
    assert profile.anchor_body == profile.com_body == "base_link"


def test_casbot02_fk_and_gradients():
    from supermimic.kinematics.g1_fk import G1Kinematics

    model = get_robot("casbot02").model()
    rng = np.random.default_rng(42)
    qpos = np.zeros((8, model.nq))
    qpos[:, :3] = rng.normal(size=(8, 3))
    quat = rng.normal(size=(8, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    qpos[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], (8, 27))
    qvel = rng.normal(size=(8, model.nv))
    qp = torch.tensor(qpos, requires_grad=True)
    qv = torch.tensor(qvel, requires_grad=True)
    out = G1Kinematics(model, "cpu", torch.float64)(qp, qv)
    data = mujoco.MjData(model)
    velocity = np.zeros(6)
    for t in range(8):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(out["pos"][t].detach(), data.xpos[1:], atol=1e-10)
        for body in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            np.testing.assert_allclose(
                out["lin_vel"][t, body - 1].detach(), velocity[3:], atol=1e-9
            )
            np.testing.assert_allclose(
                out["ang_vel"][t, body - 1].detach(), velocity[:3], atol=1e-9
            )
    (out["pos"].square().sum() + out["lin_vel"].square().sum()).backward()
    assert torch.isfinite(qp.grad).all() and torch.isfinite(qv.grad).all()


@pytest.fixture
def casbot_clip(tmp_path):
    """A smooth synthetic reference independent of external downloaded datasets."""
    from mjlab.entity import Entity

    from supermimic.data.motion_convert import convert_csv

    profile = get_robot("casbot02")
    model = Entity(profile.entity_cfg()).spec.compile()
    raw = np.tile(model.key_qpos[0], (120, 1))
    raw[:, 3:7] = raw[:, [4, 5, 6, 3]]  # CSV xyzw
    raw[:, 0] = np.arange(120) * 0.001
    raw[:, 7] += 0.005 * np.sin(np.arange(120) * 0.05)
    source, output = tmp_path / "standing.csv", tmp_path / "standing.npz"
    np.savetxt(source, raw, delimiter=",")
    convert_csv(source, output, 30, 50, robot="casbot02")
    return output


def test_conversion_library_and_network_dimensions(casbot_clip):
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    profile = get_robot("casbot02")
    layout = profile.layout
    assert (layout.state_dim, layout.obs_dim, layout.goal_dim, layout.xi_dim) == (67, 87, 64, 31)
    q0 = torch.zeros(27)
    kwargs = {"root_body_name": "base_link", "robot_name": "casbot02", "device": "cpu"}
    full = MotionLibrary([casbot_clip], profile.tracked_bodies, q0, **kwargs)
    compact = MotionLibrary(
        [casbot_clip], profile.tracked_bodies, q0, compact=True, store_dtype=torch.float32, **kwargs
    )
    frames = torch.arange(20)
    np.testing.assert_allclose(full.bodies(frames).pos_w, compact.bodies(frames).pos_w, atol=2e-5)
    np.testing.assert_allclose(
        full.goal_window(frames, 10), compact.goal_window(frames, 10), atol=1e-5
    )
    qp, qv = full.ref_qpos_qvel(frames)
    state = torch.cat([qp, qv], -1)
    spec = StateSpec(StateSpecCfg(num_joints=27, explicit_integration=False), "cpu")
    torch.testing.assert_close(spec.integrate(state, torch.zeros(20, 66)), state)
    assert spec.canonical(state).shape == (20, 67)
    xi = torch.zeros(20, 31)
    assert spec.normalize_xi(xi).shape == xi.shape
    obs = analytic_observation(state, torch.zeros(20, 27), xi, q0)
    hist = flatten_history_term_major(obs[None].expand(10, -1, -1))
    assert hist.shape == (20, 870)
    cfg = PolicyCfg(num_joints=27, goal_dim=640, encoder_hidden=(64,), decoder_hidden=(64,))
    policy = Policy(cfg)
    assert policy(hist, full.goal_window(frames, 10)).shape == (20, 27)
    critic = Critic(CriticCfg(hidden=(64,)), pcfg=cfg)
    assert critic(
        spec.canonical(state), full.goal_window(frames, 10), xi, torch.zeros(20, 4)
    ).shape == (20,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_casbot02_training_and_checkpoint(casbot_clip, tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(robot="casbot02", run_name="casbot02_smoke", log_root=tmp_path)
    cfg.data.npz_dir = casbot_clip.parent
    cfg.env.num_envs = 32
    cfg.env.replay_capacity = 64
    cfg.env.anchor_obs = True
    cfg.world_model.hidden = (128, 128)
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (128, 64)
    cfg.wm_trainer.batch_size = 64
    cfg.wm_trainer.batches_per_update = 1
    cfg.wm_trainer.compile = "default"
    cfg.policy_trainer.batch_size = 32
    cfg.policy_trainer.horizon = 4
    cfg.policy_trainer.compile = "default"
    cfg.schedule.total_iterations = 35
    cfg.schedule.warmup_iterations = 20
    cfg.schedule.log_interval = cfg.schedule.checkpoint_interval = 35
    trainer = Trainer(cfg)
    try:
        trainer.run()
        assert trainer.num_updates > 0
        starts = trainer.buffer.sample_starts(8)
        assert starts.state.shape == (8, 67) and starts.hist_actions.shape == (10, 8, 27)
        assert torch.isfinite(starts.state).all()
        ckpt = tmp_path / "casbot02_smoke/ckpt_latest.pt"
        policy, saved = load_policy_from_checkpoint(ckpt, "cuda:0")
        assert saved["robot"] == "casbot02"
        assert saved["policy"]["goal_dim"] == 649
        assert torch.isfinite(
            policy(torch.zeros(1, 870, device="cuda"), torch.zeros(1, 649, device="cuda"))
        ).all()
    finally:
        trainer.env.close()
        trainer.writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_casbot02_downloaded_lafan1_env():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    root = Path(__file__).resolve().parents[1] / "data/npz/lafan1_casbot02"
    files = [
        root / f"{name}.npz" for name in ("walk1_subject1", "dance1_subject1", "jumps1_subject1")
    ]
    if not all(path.exists() for path in files):
        pytest.skip("Casbot02 LAFAN1 not converted")
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[str(p) for p in files],
            robot="casbot02",
            num_envs=8,
            replay_capacity=128,
            anchor_obs=True,
        )
    )
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (8, 870) and obs["goal"].shape == (8, 649)
        for _ in range(100):
            obs, reward, *_ = env.step(torch.randn(8, 27, device=env.device) * 0.1)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        windows = env.recorder_manager.get_term("replay").buffer.sample_windows(32, 8)
        assert windows.states.shape == (9, 32, 67) and windows.xi.shape == (32, 31)
        assert torch.isfinite(windows.states).all()
    finally:
        env.close()

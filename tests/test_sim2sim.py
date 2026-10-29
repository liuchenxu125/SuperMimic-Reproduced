"""Deployment-boundary checks: layout, frames, controller mapping and export."""

from dataclasses import asdict

import mujoco
import numpy as np
import pytest
import torch

from supermimic.deployment.sim2sim import History, Reference, Sim2Sim, observation, rotation


def test_native_observation_and_history_match_training():
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

    rng = np.random.default_rng(73)
    states = rng.normal(size=(10, 67)) * 0.2
    states[:, 3:7] = rng.normal(size=(10, 4))
    states[:, 3:7] /= np.linalg.norm(states[:, 3:7], axis=-1, keepdims=True)
    actions = rng.normal(size=(10, 27))
    q0 = rng.normal(size=27) * 0.1
    expected = analytic_observation(
        torch.tensor(states), torch.tensor(actions), torch.zeros(10, 31), torch.tensor(q0)
    ).float()
    h = History(27, 10)
    for i in range(10):
        frame = observation(states[i, :34], states[i, 34:], actions[i], q0)
        np.testing.assert_allclose(frame, expected[i], atol=1e-7)
        if i == 0:
            h.reset(frame)
        else:
            h.append(frame)
    packed = flatten_history_term_major(expected[:, None, :])[0].numpy()
    np.testing.assert_allclose(h.packed(), packed, atol=1e-7)
    h.reset(expected[-1].numpy())
    repeated = flatten_history_term_major(expected[-1:].expand(10, -1)[:, None, :])[0].numpy()
    np.testing.assert_allclose(h.packed(), repeated)


@pytest.fixture(scope="module", params=[(True, False, False, "mesh"),
                                        (False, False, False, "mesh"),
                                        (False, True, False, "mesh"),
                                        (False, False, True, "mesh"),
                                        (False, False, "step", "mesh"),
                                        (False, False, "step", "primitive")],
                ids=["anchor", "proprio", "orientation", "delayed-proprio",
                     "step-delayed-proprio", "primitive"])
def bundle(tmp_path_factory, request):
    from mjlab.entity import Entity

    from supermimic.deployment.export import export_bundle
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.robots import get_robot

    root = tmp_path_factory.mktemp("native-mujoco")
    profile = get_robot("casbot02")
    e = Entity(profile.entity_cfg())
    m = e.spec.compile()
    d = mujoco.MjData(m)
    q = np.tile(m.key_qpos[0], (20, 1))
    q[:, 0] = np.arange(20) * 0.002
    q[:, 2] = 0.91
    q[:, 7] += np.sin(np.arange(20) * 0.1) * 0.01
    velocities = np.zeros((20, m.nv))
    velocities[:, 0] = 0.1
    bp, bq = [], []
    for qp, qv in zip(q, velocities):
        d.qpos[:] = qp
        d.qvel[:] = qv
        mujoco.mj_forward(m, d)
        bp.append(d.xpos[1:].copy())
        bq.append(d.xquat[1:].copy())
    npz = root / "small.npz"
    np.savez(
        npz,
        fps=[50.0],
        joint_names=np.array(e.joint_names),
        body_names=np.array([m.body(i).name for i in range(1, m.nbody)]),
        joint_pos=q[:, 7:].astype(np.float32),
        joint_vel=velocities[:, 6:].astype(np.float32),
        body_pos_w=np.array(bp, dtype=np.float32),
        body_quat_w=np.array(bq, dtype=np.float32),
        body_lin_vel_w=np.tile([0.1, 0, 0], (20, m.nbody - 1, 1)).astype(np.float32),
        body_ang_vel_w=np.zeros((20, m.nbody - 1, 3), np.float32),
    )
    anchor, anchor_ori, delay, collision = request.param
    pc = PolicyCfg(
        num_joints=27, goal_dim=649 if anchor else 646 if anchor_ori else 640,
        encoder_hidden=(32,), decoder_hidden=(32,)
    )
    torch.manual_seed(17)
    p = Policy(pc)
    cfg = {
        "robot": "casbot02",
        "policy": asdict(pc),
        "env": {"future_frames": 10, "anchor_obs": anchor,
                **({"actuator_delay": True} if delay else {}),
                **({"actuator_delay_mode": "step"} if delay == "step" else {}),
                **({"collision_model": "primitive"} if collision == "primitive" else {}),
                **({"anchor_ori_obs": True} if anchor_ori else {})},
        "world_model": {"height_threshold": 0.25, "tilt_threshold": 0.8},
    }
    ck = root / "checkpoint.pt"
    torch.save({"policy": p.state_dict(), "config": cfg}, ck)
    meta = export_bundle(ck, [npz], root / "bundle")
    return root / "bundle", p, meta, npz


def test_reference_goal_and_spatial_alignment_match_training(bundle):
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.kinematics.rotations import (
        matrix_from_quat,
        quat_apply_inverse,
        quat_inv,
        quat_mul,
    )
    from supermimic.robots import get_robot

    path, _, meta, source = bundle
    q0 = torch.tensor([x["q0"] for x in meta["motors"]])
    profile = get_robot("casbot02")
    lib = MotionLibrary(
        [source],
        profile.tracked_bodies,
        q0,
        device="cpu",
        root_body_name="base_link",
        robot_name="casbot02",
    )
    ref = Reference(path / meta["motions"]["small"], meta["joint_names"], 10, 0.02)
    quat = np.array([0.9, 0.1, -0.2, 0.3])
    quat /= np.linalg.norm(quat)
    pos = np.array([0.2, -0.1, 0.9])
    for f in [0, 4, 19]:
        g = ref.goal(f, pos, quat, True)
        frame = torch.tensor([f])
        qp = torch.tensor(quat, dtype=torch.float32)[None]
        pp = torch.tensor(pos, dtype=torch.float32)[None]
        r = lib.ref(frame)
        p = quat_apply_inverse(qp, r.root_pos_w - pp)
        o = matrix_from_quat(quat_mul(quat_inv(qp), r.root_quat_w))[..., :2].reshape(1, 6)
        expected = torch.cat((lib.goal_window(frame, 10), p, o), -1)[0].numpy()
        np.testing.assert_allclose(g, expected, atol=3e-7)
    before = ref.goal(4, pos, quat, True)
    # Apply the same arbitrary world XY/yaw transform to robot and reference.
    delta = 0.73
    dq = np.array([np.cos(delta / 2), 0, 0, np.sin(delta / 2)])
    from supermimic.deployment.sim2sim import quat_multiply, yaw

    r = rotation(dq)
    offset = np.array([2, -3, 0])
    new_start = (r @ ref.root_pos[0] + offset)[:2]
    ref.align(0, new_start, float(yaw(ref.root_quat[0])) + delta)
    after = ref.goal(4, r @ pos + offset, quat_multiply(dq, quat), True)
    np.testing.assert_allclose(before, after, atol=3e-7)


def test_export_actor_and_motor_order(bundle):
    path, policy, meta, _ = bundle
    sim = Sim2Sim(path)
    sim.reset("small")
    assert sim.command_delay.max_lag == (2 if meta["actuator_delay"] else 0)
    assert sim.command_delay.mode == meta["actuator_delay_mode"]
    source = torch.load(path.parent / "checkpoint.pt", weights_only=False)["config"]["env"]
    assert sim.command_delay.mode == source.get("actuator_delay_mode", "episode")
    assert meta["collision_model"] == source.get("collision_model", "mesh")
    assert len(meta["foot_geom_names"]) == (14 if meta["collision_model"] == "primitive" else 2)
    assert sim.target_frame == 1
    assert meta["joint_names"][12] == "waist_yaw_joint"
    assert sim.actuator_to_joint[-1] == 12  # XML actuator order is different.
    assert sim.model.opt.timestep == 0.005
    assert sim.decimation == 4
    for f in [0, 1, 4]:
        obs, goal = sim.inputs()
        assert obs.shape == (870,) and goal.shape == (meta["goal_dim"],)
        with torch.inference_mode():
            expected = policy(torch.from_numpy(obs[None]), torch.from_numpy(goal[None]))
            actual = sim.actor(torch.from_numpy(obs[None]), torch.from_numpy(goal[None]))
        torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)
        sim.step()
        assert np.isfinite(sim.data.qpos).all()
    a = np.linspace(-0.5, 0.5, 27)
    target = sim.position_targets(a)
    for actuator, joint in enumerate(sim.actuator_to_joint):
        p = meta["motors"][joint]
        assert target[joint] == pytest.approx(p["q0"] + p["scale"] * a[joint])
        assert sim.model.actuator_gainprm[actuator, 0] == p["kp"]
        np.testing.assert_array_equal(sim.model.actuator_biasprm[actuator, :3], [0, -p["kp"], -p["kd"]])
        np.testing.assert_array_equal(sim.model.actuator_forcerange[actuator], [-p["effort_limit_nm"], p["effort_limit_nm"]])
    sim.completed = True
    sim.reset("small")
    assert not sim.completed
    with pytest.raises(FloatingPointError):
        sim.position_targets(np.full(27, np.nan))


def test_bad_reference_rejected(bundle, tmp_path):
    path, _, meta, _ = bundle
    with np.load(path / meta["motions"]["small"]) as src:
        data = {k: src[k] for k in src.files}
    data["joint_names"] = data["joint_names"][::-1]
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, **data)
    with pytest.raises(ValueError, match="joint order"):
        Reference(wrong, meta["joint_names"], 10, 0.02)
    with pytest.raises(ValueError, match="quaternion"):
        rotation([0, 0, 0, 0])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_divergent_checkpoint_rejected_before_export(bundle, tmp_path, bad_value):
    from supermimic.deployment.export import export_bundle

    path, _, _, source = bundle
    checkpoint = torch.load(path.parent / "checkpoint.pt", weights_only=False)
    weights = next(value for value in checkpoint["policy"].values() if value.ndim == 2)
    weights.view(-1)[0] = bad_value
    broken = tmp_path / "broken.pt"
    torch.save(checkpoint, broken)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Policy contains NaN/Inf"):
        export_bundle(broken, [source], output)
    assert not output.exists()


def test_proprio_goal_ignores_unavailable_root_pose(bundle):
    path, _, meta, _ = bundle
    if meta["anchor_obs"]:
        pytest.skip("Only applies to the proprio interface")
    sim = Sim2Sim(path)
    sim.reset("small")
    goal = sim.reference.goal(1, None, None, False)
    unavailable = sim.reference.goal(1, np.full(3, np.nan), np.full(4, np.nan), False)
    np.testing.assert_array_equal(goal, unavailable)
    assert goal.shape == (640,)


def test_orientation_goal_preserves_reference_and_needs_no_position(bundle):
    path, _, _, _ = bundle
    sim = Sim2Sim(path)
    sim.reset("small")
    ref = sim.reference
    frame = 4
    quat = np.array([np.cos(0.3), 0, 0, np.sin(0.3)])
    expected = (rotation(quat).T @ rotation(ref.root_quat[frame]))[:, :2].reshape(-1)
    goal = ref.goal(frame, None, quat, False, True)
    unavailable = ref.goal(frame, np.full(3, np.nan), quat, False, True)
    assert goal.shape == (646,)
    np.testing.assert_array_equal(goal[:640], ref.goal(frame, None, None, False))
    np.testing.assert_array_equal(goal, unavailable)
    np.testing.assert_allclose(goal[640:], expected, atol=1e-7)
    # Full anchor mode already includes orientation: enabling both must not duplicate it.
    full = ref.goal(frame, np.zeros(3), quat, True, True)
    assert full.shape == (649,)
    np.testing.assert_array_equal(full[-6:], goal[-6:])


def test_reference_overlay_geometry_without_gl_context(bundle):
    from supermimic.deployment.sim2sim import ReferenceGhost

    path, _, _, _ = bundle
    sim = Sim2Sim(path)
    sim.reset("small")
    ghost = ReferenceGhost(sim.model)
    scene = mujoco.MjvScene(sim.model, maxgeom=1000)
    ghost.update(sim.reference, 4, scene)
    assert scene.ngeom > 0
    np.testing.assert_allclose(ghost.data.qpos[:3], sim.reference.root_pos[4])
    assert all(scene.geoms[i].rgba[3] == pytest.approx(0.3) for i in range(scene.ngeom))
    # Changing the overlay must not change the controlled robot's position.
    np.testing.assert_allclose(sim.data.qpos[:3], sim.reference.root_pos[0])

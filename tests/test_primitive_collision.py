"""Primitive collisions preserve morphology while keeping feet and self contacts."""
import mujoco
import numpy as np
import pytest
from mjlab.entity import Entity

from supermimic.robots import get_robot
from supermimic.robots.profiles import CASBOT_ASSETS


def test_primitive_model_preserves_inertia_actuators_and_fk():
    profile = get_robot("casbot02")
    mesh = Entity(profile.entity_cfg(collision_model="mesh")).spec.compile()
    primitive = Entity(profile.entity_cfg(collision_model="primitive")).spec.compile()
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                  "jnt_pos", "jnt_axis", "jnt_range", "dof_armature", "dof_damping",
                  "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(mesh, field), getattr(primitive, field), err_msg=field)
    assert (mesh.nq, mesh.nv, mesh.nu) == (primitive.nq, primitive.nv, primitive.nu) == (34, 33, 27)
    a, b = mujoco.MjData(mesh), mujoco.MjData(primitive)
    rng = np.random.default_rng(131)
    for _ in range(12):
        a.qpos[:] = mesh.key_qpos[0]
        a.qpos[7:] = rng.uniform(mesh.jnt_range[1:, 0], mesh.jnt_range[1:, 1])
        b.qpos[:] = a.qpos
        mujoco.mj_forward(mesh, a)
        mujoco.mj_forward(primitive, b)
        np.testing.assert_array_equal(a.xpos, b.xpos)
        np.testing.assert_array_equal(a.xquat, b.xquat)


def test_primitive_geometry_soles_and_default_clearance():
    m = Entity(get_robot("casbot02").entity_cfg(collision_model="primitive")).spec.compile()
    active = np.where((m.geom_contype != 0) | (m.geom_conaffinity != 0))[0]
    assert len(active) == 19
    assert set(m.geom_type[active]) <= {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                        int(mujoco.mjtGeom.mjGEOM_MESH)}
    source = mujoco.MjModel.from_xml_path(str(CASBOT_ASSETS / "xml/casbot02.xml"))
    for side in ("left", "right"):
        foot, original = m.geom(f"{side}_foot_collision"), source.geom(f"{side}_foot_collision")
        assert m.geom_type[foot.id] == mujoco.mjtGeom.mjGEOM_MESH
        assert m.geom_condim[foot.id] == source.geom_condim[original.id] == 3
        assert (m.geom_contype[foot.id], m.geom_conaffinity[foot.id]) == (
            source.geom_contype[original.id], source.geom_conaffinity[original.id])
        np.testing.assert_allclose(foot.friction[:3], original.friction[:3])
        np.testing.assert_allclose(foot.solref[:2], original.solref[:2])
        hip = m.geom(f"{side}_hip_collision")
        assert m.body(m.geom_bodyid[hip.id]).name == f"{side}_leg_pelvic_pitch_link"
        assert m.geom_type[hip.id] == mujoco.mjtGeom.mjGEOM_CAPSULE
        visual = next(g for g in range(source.ngeom)
                      if source.body(source.geom_bodyid[g]).name == f"{side}_leg_pelvic_pitch_link"
                      and source.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                      and source.geom_contype[g] == 0)
        mesh = int(source.geom_dataid[visual])
        start, count = int(source.mesh_vertadr[mesh]), int(source.mesh_vertnum[mesh])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, source.geom_quat[visual])
        points = source.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + source.geom_pos[visual]
        quat_matrix = np.empty(9)
        mujoco.mju_quat2Mat(quat_matrix, hip.quat)
        direction = quat_matrix.reshape(3, 3)[:, 2]
        start_point = hip.pos - direction * hip.size[1]
        end_point = hip.pos + direction * hip.size[1]
        segment = end_point - start_point
        along = np.clip(((points - start_point) @ segment) / (segment @ segment), 0, 1)
        distance = np.linalg.norm(points - (start_point + along[:, None] * segment), axis=1)
        assert (distance <= hip.size[0] + 1e-8).mean() >= 0.99
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    assert d.ncon == 0  # No false mechanical self-intersections in the preparation pose.
    rng = np.random.default_rng(71)
    found_self_contact = False
    for _ in range(30):
        d.qpos[7:] = rng.uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1])
        mujoco.mj_forward(m, d)
        found_self_contact |= d.ncon > 0
    assert found_self_contact  # Non-adjacent self collisions remain physical.


def test_mesh_remains_legacy_default_for_checkpoint_environment():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg

    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused"], robot="casbot02"))
    m = Entity(cfg.scene.entities["robot"]).spec.compile()
    active = np.where((m.geom_contype != 0) | (m.geom_conaffinity != 0))[0]
    assert set(m.geom_type[active]) == {int(mujoco.mjtGeom.mjGEOM_MESH)}


def test_new_training_defaults_and_small_viewer_capacity():
    from supermimic.config import TrainConfig
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg

    assert TrainConfig(robot="casbot02").env.collision_model == "primitive"
    a = SuperMimicEnvCfgArgs(npz_files=["unused"], robot="casbot02", collision_model="primitive")
    cfg = make_env_cfg(a)
    assert (cfg.sim.nconmax, cfg.sim.njmax) == (48, 256)
    a.num_envs = 1
    cfg = make_env_cfg(a)
    assert (cfg.sim.nconmax, cfg.sim.njmax) == (128, 256)


def test_resume_cannot_silently_change_collision_model(tmp_path):
    from dataclasses import asdict

    import torch

    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(robot="casbot02", run_name="new", log_root=tmp_path)
    cfg.env.collision_model = "primitive"
    saved = asdict(cfg)
    saved["env"].pop("collision_model")
    saved["policy"]["goal_dim"] = 690
    p = tmp_path / "legacy.pt"
    torch.save({"iteration_unit": "update", "iteration": 10, "env_steps": 65,
                "num_updates": 10, "collection_accumulator": 0., "config": saved}, p)
    cfg.resume = p
    with pytest.raises(ValueError, match="Resume collision model differs"):
        Trainer(cfg)

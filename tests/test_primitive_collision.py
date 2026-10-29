"""Primitive collisions preserve morphology while keeping feet and self contacts."""
import json

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
    assert len(active) == 29
    assert set(m.geom_type[active]) <= {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}
    assert (m.geom_contype[active] == 1).all() and (m.geom_conaffinity[active] == 1).all()
    bounds = json.loads((CASBOT_ASSETS / "primitive_collision.json").read_text())["foot_mesh_bounds"]
    for side in ("left", "right"):
        for i in range(1, 8):
            g = m.geom(f"{side}_foot{i}_collision")
            assert m.geom_condim[g.id] == 3 and m.geom_priority[g.id] == 1
            np.testing.assert_allclose(g.pos[2] - g.size[0], bounds[side]["min"][2], atol=1e-9)
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
    saved["policy"]["goal_dim"] = 640
    p = tmp_path / "legacy.pt"
    torch.save({"iteration_unit": "update", "iteration": 10, "env_steps": 65,
                "num_updates": 10, "collection_accumulator": 0., "config": saved}, p)
    cfg.resume = p
    with pytest.raises(ValueError, match="Resume collision model differs"):
        Trainer(cfg)

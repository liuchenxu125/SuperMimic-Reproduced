"""A limit migration must never waive the actuator or observation contract."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "deployment_updater", Path(__file__).resolve().parents[1] / "scripts/update_casbot02_deployment.py"
)
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


@pytest.fixture
def metadata():
    return dict(motors=[dict(kp=100., kd=5.)], joint_names=["elbow"],
                sdk_joint_names=["elbow"], obs_dim=870, goal_dim=640, history=10,
                future_frames=10, control_dt=.02, joint_limits=[[-1.8675, 0.]],
                soft_joint_limits=[[-1.774125, -.093375]])


def test_joint_limit_migration_requires_explicit_acceptance(metadata):
    updated = copy.deepcopy(metadata)
    updated["joint_limits"] = [[-2.26, 0.]]
    updated["soft_joint_limits"] = [[-2.147, -.113]]
    with pytest.raises(ValueError, match="accept-joint-limit-update"):
        updater.verify_contract(updated, metadata)
    changes = updater.verify_contract(updated, metadata, True)
    assert set(changes) == {"joint_limits", "soft_joint_limits"}
    assert changes["joint_limits"]["before"] == metadata["joint_limits"]
    assert changes["joint_limits"]["after"] == updated["joint_limits"]
    # The recorded transition must be serializable in deployment.json.
    json.dumps(changes)


@pytest.mark.parametrize("field,value", [("motors", [dict(kp=50., kd=2.)]),
                                       ("goal_dim", 646), ("joint_names", ["other"])])
def test_limit_acceptance_does_not_accept_other_contract_changes(metadata, field, value):
    updated = copy.deepcopy(metadata)
    updated[field] = value
    with pytest.raises(ValueError, match=field):
        updater.verify_contract(updated, metadata, True)


def test_heading_goal_migration_requires_matching_schema_and_explicit_flag(metadata):
    updated = copy.deepcopy(metadata)
    updated.update(goal_dim=690, goal_encoding='reference_heading_v1')
    with pytest.raises(ValueError, match='accept-goal-update'):
        updater.verify_contract(updated, metadata)
    assert updater.verify_contract(updated, metadata, accept_goal_update=True) == {}
    updated['anchor_ori_obs'] = True
    with pytest.raises(ValueError, match='goal_dim'):
        updater.verify_contract(updated, metadata, accept_goal_update=True)


def test_goal_acceptance_preserves_motor_guard(metadata):
    updated = copy.deepcopy(metadata)
    updated.update(goal_dim=690, goal_encoding='reference_heading_v1', motors=[])
    with pytest.raises(ValueError, match='motors'):
        updater.verify_contract(updated, metadata, accept_goal_update=True)

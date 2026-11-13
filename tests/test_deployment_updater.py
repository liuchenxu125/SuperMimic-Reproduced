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
    with pytest.raises(ValueError, match='orientation'):
        updater.verify_contract(updated, metadata, accept_goal_update=True)


def test_goal_acceptance_preserves_motor_guard(metadata):
    updated = copy.deepcopy(metadata)
    updated.update(goal_dim=690, goal_encoding='reference_heading_v1', motors=[])
    with pytest.raises(ValueError, match='motors'):
        updater.verify_contract(updated, metadata, accept_goal_update=True)


def test_explicit_motor_migration_preserves_other_interface_guards(metadata):
    updated = copy.deepcopy(metadata)
    updated['motors'] = [dict(kp=120., kd=6.)]
    with pytest.raises(ValueError, match='motors'):
        updater.verify_contract(updated, metadata)
    assert updater.verify_contract(updated, metadata, accept_motor_update=True) == {}
    updated['sdk_joint_names'] = ['wrong']
    with pytest.raises(ValueError, match='sdk_joint_names'):
        updater.verify_contract(updated, metadata, accept_motor_update=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_orientation_history_migration_requires_explicit_matching_contract(metadata, reverse):
    before = copy.deepcopy(metadata)
    before.update(goal_dim=690, goal_encoding='reference_heading_v1',
                  anchor_obs=False, anchor_ori_obs=False)
    after = copy.deepcopy(before)
    after.update(obs_dim=930, anchor_ori_obs=True)
    if reverse:
        before, after = after, before
    with pytest.raises(ValueError, match='accept-orientation-update'):
        updater.verify_contract(after, before)
    assert updater.verify_contract(after, before, accept_orientation_update=True) == {}
    for key, value in [('goal_dim', 696), ('motors', []), ('anchor_obs', True), ('obs_dim', 936),
                       ('anchor_ori_obs', not after['anchor_ori_obs'])]:
        wrong = copy.deepcopy(after)
        wrong[key] = value
        with pytest.raises(ValueError):
            updater.verify_contract(wrong, before, accept_orientation_update=True)

"""The active training actuators exactly retain cpu7's motor contract."""
import hashlib
import json

import mujoco
import numpy as np
import pytest

from supermimic.robots import get_robot
from supermimic.robots.profiles import CASBOT_ASSETS


# Independently recorded from the cpu7 training asset snapshot, including its
# historical waist armature/gain combination (do not recompute those gains).
CPU7_PARAMETERS_SHA256 = "624b573ea04761337fb045ccf258e1373ec2275583495229be7a0d7770d9fb45"


def expected_motor(name):
    if name.startswith('leg_'):
        if int(name.split('_')[1][1:]) in (1, 2, 4):
            return .07, 276.3489232147048, 17.5929188596, 120., 14.
        return .029, 114.48741104609199, 7.28849495612, 80., 14.
    if name == 'waist_yaw_joint':
        return .039, 96.72212312514668, 6.15752160086, 60., 14.
    if int(name.split('_')[-2]) in (1, 2, 4):
        return .033, 130.27877808693228, 8.29380460524, 40., 12.2
    return .0245, 96.72212312514668, 6.15752160086, 30., 9.3


@pytest.mark.parametrize('collision_model', ['mesh', 'primitive'])
def test_cpu7_motors_reach_actual_implicit_pd_and_xml(collision_model):
    from mjlab.entity import Entity

    profile = get_robot('casbot02')
    parameters = profile.motor_parameters()['motors']
    model = Entity(profile.entity_cfg(collision_model=collision_model)).spec.compile()
    source_model = profile.model(collision_model=collision_model)
    limits = json.loads((CASBOT_ASSETS / 'source_manifest.json').read_text())['training_motor_update_20261107']['preserved_joint_ranges']
    xml_name = 'xml/casbot02.xml' if collision_model == 'mesh' else 'xml/casbot02_primitives.xml'
    assert model.actuator_forcelimited.all()
    assert model.nu == 27
    for actuator in range(model.nu):
        j = int(model.actuator_trnid[actuator, 0])
        name = model.joint(j).name
        armature, kp, kd, effort, speed = expected_motor(name)
        p = parameters[name]
        dof = int(model.jnt_dofadr[j])
        assert p['armature'] == model.dof_armature[dof] == armature
        assert source_model.dof_armature[source_model.jnt_dofadr[source_model.joint(name).id]] == armature
        assert p['stiffness'] == model.actuator_gainprm[actuator, 0] == kp
        assert p['damping'] == -model.actuator_biasprm[actuator, 2] == kd
        assert p['effort_limit'] == effort and p['velocity_limit'] == speed
        np.testing.assert_array_equal(model.actuator_forcerange[actuator], [-effort, effort])
        np.testing.assert_array_equal(model.jnt_range[j], np.fromstring(limits[xml_name][name], sep=' '))
        assert profile.action_scale()[name] == .25 * effort / kp
    # Verify actual implicit-PD force saturation, not only metadata values.
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    data.ctrl[:] = 1e6
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, model.actuator_forcerange[:, 1])


def test_exact_cpu7_parameter_snapshot_and_default_pose():
    from mjlab.entity import Entity

    path = CASBOT_ASSETS / 'motor_parameters.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CPU7_PARAMETERS_SHA256
    params = json.loads(path.read_text())
    model = Entity(get_robot('casbot02').entity_cfg()).spec.compile()
    for name, expected in params['default_joint_pos'].items():
        j = model.joint(name)
        assert model.key_qpos[0, int(j.qposadr[0])] == expected

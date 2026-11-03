"""The 690-goal actor needs raw references, not independently cached windows."""
import struct

import numpy as np
import pytest

from supermimic.deployment.motion_binary import encode_motion


def reference():
    return dict(root_pos_w=np.array([[1., 2., .8], [1.2, 2.3, .9]]),
                root_quat_w=np.array([[1., 0., 0., 0.]]*2),
                joint_pos=np.zeros((2, 27)), joint_vel=np.ones((2, 27)),
                root_lin_vel_w=np.array([[.1, .2, .3], [.4, .5, .6]]),
                root_ang_vel_w=np.array([[.7, .8, .9], [1., 1.1, 1.2]]))


def test_raw_reference_record_preserves_world_velocities_and_joint_order():
    fields = reference()
    blob, count = encode_motion(fields, 'reference_heading_v1')
    assert blob[:8] == b'SMREF002'
    assert struct.unpack('<II', blob[8:16]) == (count, 67)
    records = np.frombuffer(blob, dtype='<f4', offset=16).reshape(count, 67)
    for key, start, stop in [('root_pos_w', 0, 3), ('root_quat_w', 3, 7),
                             ('joint_pos', 7, 34), ('joint_vel', 34, 61),
                             ('root_lin_vel_w', 61, 64), ('root_ang_vel_w', 64, 67)]:
        np.testing.assert_allclose(records[:, start:stop], fields[key], atol=1e-7)


def test_legacy_format_stays_separate():
    fields = reference()
    fields['goal_features'] = np.ones((2, 64))
    blob, count = encode_motion(fields, 'per_frame_v1')
    assert blob[:8] == b'SMREF001'
    assert struct.unpack('<II', blob[8:16]) == (count, 125)
    with pytest.raises(ValueError, match='Unsupported goal encoding'):
        encode_motion(fields, 'unknown')


def test_bad_reference_is_rejected_before_building_a_bundle():
    fields = reference()
    fields['root_lin_vel_w'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid motion'):
        encode_motion(fields, 'reference_heading_v1')
    fields = reference()
    fields['root_quat_w'][0] = 0
    with pytest.raises(ValueError, match='quaternion'):
        encode_motion(fields, 'reference_heading_v1')

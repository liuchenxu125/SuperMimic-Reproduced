"""SDK source mapping, time base and repaired training targets."""
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.casbot02_data import convert_data, decode_data, load_data
from supermimic.robots import get_robot


def write_data(path, raw, header=False):
    with path.open('w') as stream:
        if header:
            names = ['unused']*61
            names[0], names[60] = 'Base_Latation_X', 'waist_yaw_joint'
            stream.write(','.join(names)+'\n')
        for row in raw:
            stream.write(','.join(map(str, row))+',\n')


def test_sdk_mapping_keeps_first_frame_waist_and_complete_wrists(tmp_path):
    raw = np.zeros((3, 61))
    raw[:, :3] = [1., 2., .9]
    raw[:, 3] = np.pi/2
    raw[:, 5] = np.pi/2
    raw[:, 12:24] = np.arange(12)*.01
    raw[:, 24:38] = np.arange(14)*.02
    raw[:, 60] = .23
    for header in (False, True):
        path = tmp_path/f'sdk_{header}.data'
        write_data(path, raw, header)
        loaded, info = load_data(path)
        assert info['header_detected'] == header and len(loaded) == 3
        decoded, offset = decode_data(loaded)
        assert np.isclose(offset, np.pi/2)
        np.testing.assert_allclose(decoded[:, :3], [[2., 1., .9]]*3)
        rotation = Rotation.from_quat(decoded[0, 3:7]).as_matrix()
        np.testing.assert_allclose(rotation@[1., 0., 0.], [0., -1., 0.], atol=1e-10)
        q = decoded[0, 7:]
        assert len(q) == 27 and q[12] == .23
        np.testing.assert_allclose(q[:12], raw[0, 12:24])
        np.testing.assert_allclose(q[13:20], raw[0, 24:31])
        np.testing.assert_allclose(q[20:27], raw[0, 31:38])
        assert q[18] == .10 and q[25] == .24  # wrist pitch would be dropped by AMP's 22-DOF mapping
    decoded, offset = decode_data(np.zeros((3, 61)))
    assert offset == 0.
    np.testing.assert_allclose(decoded[:, 3:7], [[0., 0., 0., 1.]]*3)


def test_resample_and_clamp_rebuild_velocities_and_body_targets(tmp_path):
    profile = get_robot('casbot02'); model = profile.model()
    raw = np.zeros((61, 61))
    raw[:, 0] = np.arange(61)/1000*.1
    raw[:, 2] = .9
    raw[:, 3] = np.pi/2
    raw[:, 27] = model.jnt_range[model.joint('upper_left_4_joint').id, 0] - .3
    raw[:, 29] = .12  # preserve wrist pitch
    raw[:, 60] = .15
    raw[0, 6:12] = 1e6  # SDK initialization spike must not enter derived targets
    source = tmp_path/'motion.data'; write_data(source, raw)
    arrays, report = convert_data(source, model, 1000)
    assert report['output_frames'] == 4 and report['frames_clamped'] == 4
    assert report['output_limit_statistics']['frames_outside_hard_limits'] == 0
    q = arrays['joint_pos'].astype(np.float64)
    assert (q >= model.jnt_range[1:, 0]).all() and (q <= model.jnt_range[1:, 1]).all()
    assert arrays['joint_names'][18] == 'upper_left_6_joint'
    np.testing.assert_allclose(q[:, 18], .12)
    np.testing.assert_allclose(q[:, 12], .15)
    np.testing.assert_allclose(arrays['joint_vel'], np.gradient(q, .02, axis=0))
    np.testing.assert_allclose(arrays['body_lin_vel_w'][:, 0], [[0., .1, 0.]]*4, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(arrays['body_quat_w'], axis=-1), 1., atol=1e-7)
    original, untouched = convert_data(source, model, 1000, clamp_joints=False)
    assert untouched['frames_clamped'] == 0
    assert untouched['output_limit_statistics']['frames_outside_hard_limits'] == 4
    assert np.max(np.abs(original['body_pos_w']-arrays['body_pos_w'])) > .01


def test_retarget_elbow_pose_is_preserved_in_both_training_models(tmp_path):
    raw = np.zeros((5, 61))
    raw[:, 2] = .9
    raw[:, 27] = raw[:, 34] = -2.059394  # Real X1_Xiao360 peak elbow flexion.
    source = tmp_path/'elbows.data'
    write_data(source, raw)
    results = []
    for collision_model in ('mesh', 'primitive'):
        model = get_robot('casbot02').model(collision_model=collision_model)
        for name in ('upper_left_4_joint', 'upper_right_4_joint'):
            np.testing.assert_array_equal(model.jnt_range[model.joint(name).id], [-2.26, 0.])
        arrays, report = convert_data(source, model, 100)
        assert report['source_limit_statistics']['frames_outside_hard_limits'] == 0
        assert report['frames_clamped'] == 0
        np.testing.assert_allclose(arrays['joint_pos'][:, [16, 23]], -2.059394)
        results.append(arrays)
    for name in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
                 'body_lin_vel_w', 'body_ang_vel_w'):
        np.testing.assert_array_equal(results[0][name], results[1][name])

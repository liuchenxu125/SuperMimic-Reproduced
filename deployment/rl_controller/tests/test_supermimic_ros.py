"""SDK 2.2.37 ROS contract test with synthetic sensors; no hardware endpoints.

Models mode-7-only topics, independent GetRobotMode service, 49-joint/29-gain
feedback, integer SDK PD, and cached commands. Does not model the arm library,
EtherCAT firmware, or physical balance. All motor topics use an isolated prefix.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import struct

import numpy as np
import rclpy
import yaml
from crb_ros_msg.msg import JointStateData, JoystickCmdReport
from crb_ros_msg.srv import GetRobotMode
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger


def test(runner, config, dry, case, block_publish=None, delay_policy=None):
    cfg = yaml.safe_load(config.read_text())
    bundle = (config.parent / cfg['bundle']).resolve()
    cfg['motions'] = {name: str((config.parent / path).resolve())
                      for name, path in cfg.get('motions', {}).items()}
    automatic = case in ('auto', 'auto_skip_prepare')
    cfg.update(bundle=str(bundle), auto_start_on_mode7=automatic,
               prepare_seconds=0. if case in ('auto_skip_prepare', 'end_hold') else 2., entry_blend_seconds=1.2,
               hold_last_frame_on_finish=case == 'end_hold',
               preview_seconds=.3, return_seconds=1., auto_start_timeout_seconds=2.)
    prefix = '/supermimic_offline_test'
    for key in ('joint', 'standard_joint', 'imu', 'pose', 'mode', 'command'):
        cfg[key + '_topic'] = prefix + '/' + key
    cfg['mode_service'] = prefix + '/get_robot_mode'
    cfg['joystick_enabled'] = case in ('joystick', 'end_hold')
    cfg['joystick_topic'] = prefix + '/joystick'
    if case == 'joystick': cfg['preview_seconds'] = 0.
    manifest = json.loads((bundle / 'manifest.json').read_text())
    names = manifest['sdk_joint_names'] + ['head_pitch_joint', 'head_yaw_joint']
    kp = np.r_[np.trunc([m['kp'] for m in manifest['motors']]), [30., 30.]]
    kd = np.r_[np.trunc([m['kd'] for m in manifest['motors']]), [2., 2.]]
    row = np.fromfile(bundle / 'motion.bin', dtype='<f4', offset=16, count=125)
    q = np.r_[np.clip(row[7:34], np.array(manifest['soft_joint_limits'])[:, 0],
                     np.array(manifest['soft_joint_limits'])[:, 1]), [.13, -.21]]
    if cfg.get('prepare_pose') == 'default':
        q[:27] = [cfg['prepare_joint_pos'][n] for n in manifest['joint_names']]
    if case == 'auto_skip_prepare':
        # Exceed the old default-pose start tolerance: a skipped preparation
        # must hold measured joints, not move or wait for the unused default.
        q[3] += .2
    # Actual SDK publishes legs, arms, heads, waist, then fingers (compact PD).
    order = list(range(12)) + list(range(13, 27)) + [27, 28, 12]
    rclpy.init()
    node = rclpy.create_node('sdk237_synthetic_contract')
    pubs = {key: node.create_publisher(typ, cfg[key + '_topic'], qos_profile_sensor_data)
            for key, typ in [('joint', JointStateData), ('standard_joint', JointState),
                             ('imu', Imu), ('pose', PoseStamped)]}
    pubs['mode'] = node.create_publisher(Int32, cfg['mode_topic'], 10)
    pubs['joystick'] = node.create_publisher(JoystickCmdReport, cfg['joystick_topic'], qos_profile_sensor_data)
    state = dict(mode=7, enabled=True, imu=True, debug=True, head=True,
                 feedback=True, bad_pd=False, dq=0., query=True, mode_topic=True,
                 joy=True, joy_pressed=[2] if case == 'joystick' else [], joy_held=[], joy_axis=0)
    received, statuses = [], []
    def mode_reply(request, response):
        # Exact switch from hl_motion 2.2.37 Motion::get_robot_mode:
        # The raw enum assigned initially is OVERWRITTEN by this classification.
        raw = state['mode']
        response.mode = 2 if raw in (0, 1) else 3 if raw == 2 else 4
        response.mode_name = {2: 'ZERO', 3: 'STAND', 4: 'WALK'}[response.mode]
        return response
    mode_server = node.create_service(GetRobotMode, cfg['mode_service'], mode_reply)
    command_topic = '/supermimic_controller/dry_run_command' if dry else cfg['command_topic']
    command_sub = node.create_subscription(JointStateData, command_topic, received.append, 10)
    status_sub = node.create_subscription(String, '/supermimic_controller/status',
                                         lambda msg: statuses.append(msg.data), 10)
    clients = {n: node.create_client(Trigger, '/supermimic_controller/' + n)
               for n in ('prepare', 'start', 'stop', 'finish', 'play')}
    parameters = node.create_client(SetParameters, '/supermimic_controller/set_parameters')
    # Valid SDK hand names are unnecessary in the 29-gain indexed case. The
    # compact case uses the same real hand name repeated only AFTER non-hand
    # entries; hand state is outside this policy and intentionally ignored.
    hand_names = ['left_thumb_metacarpal_joint'] * 20

    def pump(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if state['enabled'] and state['mode'] == 7:
                stamp = node.get_clock().now().to_msg()
                msg = JointStateData(); msg.header.stamp = stamp
                selection = order if state['head'] else [i for i in order if i < 27]
                msg.name = [names[i] for i in selection]
                msg.position = q[selection].tolist(); msg.velocity = [state['dq']] * len(selection)
                if state['feedback'] and received:
                    applied = received[-1]
                    lookup = {name: i for i, name in enumerate(applied.name)}
                    msg.kp = [float(int(applied.kp[lookup[n]])) for n in msg.name]
                    msg.kd = [float(int(applied.kd[lookup[n]])) for n in msg.name]
                    if state['bad_pd']:
                        msg.kp = [400.] * len(msg.kp)  # SDK default gains override
                msg.name += hand_names
                msg.position = list(msg.position) + [0.] * 20; msg.velocity = list(msg.velocity) + [0.] * 20
                msg.effort = [0.] * len(msg.name)
                if state['debug']: pubs['joint'].publish(msg)
                standard = JointState(); standard.header.stamp = stamp
                standard.name = msg.name; standard.position = msg.position; standard.velocity = msg.velocity
                pubs['standard_joint'].publish(standard)
                if state['imu']:
                    imu = Imu(); imu.header.stamp = stamp; imu.header.frame_id = cfg['imu_frame']
                    imu.orientation.w = 1.; pubs['imu'].publish(imu)
                if manifest['anchor_obs']:
                    pose = PoseStamped(); pose.header.stamp = stamp; pose.header.frame_id = cfg['world_frame']
                    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, row[:3])
                    pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = map(float, row[3:7])
                    pubs['pose'].publish(pose)
                if state['mode_topic']: pubs['mode'].publish(Int32(data=7))  # NEVER publish a non-7 topic sample
                if state['joy']:
                    joy = JoystickCmdReport(); joy.pressed_buttons = state['joy_pressed']
                    joy.long_pressed = state['joy_held']; joy.axis_x = state['joy_axis']
                    pubs['joystick'].publish(joy)
            rclpy.spin_once(node, timeout_sec=.002)
            time.sleep(.003)

    def call(name):
        fut = clients[name].call_async(Trigger.Request()); end = time.monotonic() + 4
        while not fut.done() and time.monotonic() < end: pump(.01)
        assert fut.done(), name + ' timeout'
        return fut.result()

    def latest(phase, seconds=1.1):
        pump(seconds)
        deadline = time.monotonic() + 1.2  # status publishes only once per second
        while (not statuses or not statuses[-1].startswith(phase)) and time.monotonic() < deadline:
            pump(.03)
        assert statuses and statuses[-1].startswith(phase), statuses[-3:]

    def damping():
        assert received
        np.testing.assert_allclose(received[-1].kp, 0.)
        np.testing.assert_allclose(received[-1].kd, kd)
        np.testing.assert_allclose(received[-1].effort, 0.)

    with tempfile.TemporaryDirectory(prefix='sm_sdk237_test_') as tmp:
        tmp = Path(tmp)
        sdk_path = tmp / 'motion_params.yaml'
        sdk_path.write_text('debug_mode_params:\n  use_default_kp_kd: false\n')
        cfg['sdk_config_file'] = str(sdk_path)
        if case in ('motion_switch', 'end_hold'):
            # Real references/actor, shortened ONLY inside an isolated fixture.
            # The full-clip path must exceed the old test's .3s preview cap.
            sources = {manifest['clip']: str(bundle), **cfg['motions']}
            fixtures = {}
            for name, source in sources.items():
                source = Path(source)
                meta = json.loads((source / 'manifest.json').read_text())
                count = 81 if case == 'end_hold' else (201 if name == manifest['clip'] else 81)
                blob = (source / meta['motion_file']).read_bytes()
                folder = tmp / name; folder.mkdir()
                meta.update(frames=count, actor_file=str((source / meta['actor_file']).resolve()))
                (folder / 'manifest.json').write_text(json.dumps(meta))
                (folder / 'motion.bin').write_bytes(b'SMREF001' + struct.pack('<II', count, 125) + blob[16:16 + count * 500])
                fixtures[name] = str(folder)
            cfg.update(bundle=fixtures[manifest['clip']], motions=fixtures, preview_seconds=0.)
        path = tmp / 'test.yaml'; path.write_text(yaml.safe_dump(cfg))
        # Reject default SDK gain substitution BEFORE creating real publishers.
        sdk_path.write_text('debug_mode_params:\n  use_default_kp_kd: true\n')
        reject = subprocess.run([str(runner), str(path), '--ros-args', '-p', 'dry_run:=false'],
                                capture_output=True, text=True, timeout=15)
        assert reject.returncode and 'use_default_kp_kd=false' in reject.stderr, reject.stderr
        sdk_path.write_text('debug_mode_params:\n  use_default_kp_kd: false\n')
        log_path = tmp / 'runner.log'; flag = tmp / 'block_command'; policy_delay = tmp / 'delay_policy'
        child_env = os.environ.copy()
        if block_publish: child_env.update(LD_PRELOAD=str(block_publish), SM_TEST_BLOCK_COMMAND=str(flag))
        if delay_policy: child_env.update(LD_PRELOAD=str(delay_policy), SM_TEST_DELAY_POLICY=str(policy_delay))
        with log_path.open('w') as log:
            process = subprocess.Popen([str(runner), str(path), '--ros-args', '-p',
                                        'dry_run:=' + str(dry).lower()], stdout=log,
                                       stderr=subprocess.STDOUT, env=child_env)
            try:
                assert clients['prepare'].wait_for_service(timeout_sec=10), log_path.read_text()
                latest('IDLE', 1.5); damping()
                assert 'command_mode=sdk237_position_pd' in statuses[-1]
                assert node.count_publishers('/motion/debug/joint_cmd') == 0
                assert node.count_publishers('/motion/joint_cmd') == 0
                if dry: assert node.count_publishers(cfg['command_topic']) == 0
                if not automatic: assert not call('start').success
                if case == 'joystick':
                    # Startup-held X must not fire when first released.
                    state['joy_pressed'] = []; pump(.3); latest('IDLE', .8)

                    def pulse(button=None, axis=0):
                        state['joy_pressed'] = [] if button is None else [button]
                        state['joy_axis'] = axis; pump(.15)
                        state['joy_pressed'] = []; state['joy_axis'] = 0; pump(.15)

                    # SDK LB/RB/BACK chords cannot leak into X or B actions.
                    for modifier in [4, 5, 6]:
                        state['joy_pressed'] = [modifier, 2]; state['joy_held'] = [modifier]; pump(.1)
                        state['joy_pressed'] = [2]; state['joy_held'] = []; pump(.1)
                        state['joy_pressed'] = []; pump(.1)
                    latest('IDLE', .5)
                    pulse(axis=1); pulse(axis=-1); latest('IDLE', .8)
                    assert 'joystick_choice=' + manifest['clip'] + ' ' in statuses[-1]
                    configured_order = cfg['joystick_motion_order']
                    start_index = configured_order.index(manifest['clip'])
                    menu_order = configured_order[start_index:] + configured_order[:start_index]
                    state['joy_pressed'] = [2]; state['joy_held'] = [2]
                    latest('IDLE', 1.2)  # holding X never repeats or starts early
                    state['joy_pressed'] = []; state['joy_held'] = []
                    latest('RUN', 4.1)
                    assert ' motion=' + menu_order[1] + ' ' in statuses[-1]
                    # Live X keeps the actor running. Repeated X during the
                    # bridge is ignored; no RETURN/PREPARE/HANDOFF is allowed.
                    status_begin = len(statuses)
                    began = time.monotonic()
                    pulse(button=2); pulse(button=2)
                    deadline = began + 2.8  # includes 1 Hz status sampling
                    while time.monotonic() < deadline:
                        if (statuses[-1].startswith('RUN') and
                            ' motion=' + menu_order[2] + ' ' in statuses[-1] and
                            'frame=0 ' not in statuses[-1]): break
                        pump(.03)
                    assert time.monotonic() - began < 3.
                    assert ' motion=' + menu_order[2] + ' ' in statuses[-1], statuses[-1]
                    assert not any(s.startswith(('RETURN', 'PREPARE', 'READY', 'HANDOFF', 'FAULT'))
                                   for s in statuses[status_begin:]), statuses[status_begin:]
                    assert 'frame=0 ' not in statuses[-1]
                    # B remains return-and-hold, including during a live bridge.
                    pulse(button=2); pulse(button=1); latest('HOLD', 2.)
                    assert ' motion=' + menu_order[3] + ' ' in statuses[-1]
                    assert 'joystick_pending= ' in statuses[-1]
                    pulse(button=2); latest('RUN', 4.)
                    assert ' motion=' + menu_order[4] + ' ' in statuses[-1]
                    pulse(button=1); latest('HOLD', 2.)
                    # Selecting via the terminal remains possible. X then wraps
                    # from the final configured clip back to the first.
                    request = SetParameters.Request()
                    request.parameters = [Parameter('motion', value=menu_order[-1]).to_parameter_msg()]
                    future = parameters.call_async(request)
                    deadline = time.monotonic() + 3
                    while not future.done() and time.monotonic() < deadline: pump(.01)
                    assert future.done() and future.result().results[0].successful
                    pulse(button=2); latest('RUN', 4.)
                    assert ' motion=' + menu_order[0] + ' ' in statuses[-1]
                    pulse(button=1); latest('HOLD', 2.)
                    log_text = log_path.read_text()
                    assert 'Joystick X: ' + menu_order[0] + ' -> ' + menu_order[1] in log_text
                    assert 'Joystick X: ' + menu_order[-1] + ' -> ' + menu_order[0] in log_text
                    assert 'State: RUN | current=' in log_text
                    assert 'RUN dry_run=' not in log_text  # verbose status is topic-only by default
                    state['imu'] = False; latest('FAULT', .4)
                    state['imu'] = True; pulse(button=2); latest('FAULT', 1.2)
                    assert call('stop').success; pump(.1)
                    state['joy'] = False; pump(.7)
                    state['joy_pressed'] = [2]; state['joy'] = True; pump(.2)
                    state['joy_pressed'] = []; latest('IDLE', 1.)
                    damping()  # no replay after a joystick receipt gap
                elif case == 'end_hold':
                    assert call('play').success
                    latest('END_HOLD', 4.)
                    assert 'frame=80 ' in statuses[-1] and 'fault= ' in statuses[-1], statuses[-1]
                    begin = len(received)
                    status_begin = len(statuses)
                    pump(1.3)
                    assert len(received) > begin + 100, 'Final hold stopped publishing commands'
                    assert all(msg.kp[0] > 0 for msg in received[begin:]), 'Final hold entered damping'
                    assert all(s.startswith('END_HOLD') for s in statuses[status_begin:]), statuses[status_begin:]
                    # Continue from the stationary final reference through the existing live bridge.
                    motion_order = cfg['joystick_motion_order']
                    expected = motion_order[(motion_order.index(manifest['clip']) + 1) % len(motion_order)]
                    state['joy_pressed'] = [2]; pump(.15)
                    status_begin = len(statuses)
                    state['joy_pressed'] = []; latest('END_HOLD', 4.)
                    assert ' motion=' + expected + ' ' in statuses[-1], statuses[-1]
                    assert not any(s.startswith(('PREPARE', 'READY', 'HANDOFF', 'RETURN', 'FAULT'))
                                   for s in statuses[status_begin:]), statuses[status_begin:]
                    # B and the explicit finish service remain intentional return-and-HOLD.
                    state['joy_pressed'] = [1]; pump(.15)
                    state['joy_pressed'] = []; latest('HOLD', 1.8)
                    assert call('play').success; latest('END_HOLD', 4.)
                    assert call('finish').success; latest('HOLD', 1.8)
                    assert call('play').success; latest('END_HOLD', 4.)
                    state['imu'] = False; latest('FAULT', .5); damping()
                    state['imu'] = True; latest('FAULT', 1.)
                    assert not call('start').success
                elif case == 'motion_switch':
                    assert parameters.wait_for_service(timeout_sec=5)

                    def select(name):
                        req = SetParameters.Request()
                        req.parameters = [Parameter('motion', value=name).to_parameter_msg()]
                        fut = parameters.call_async(req)
                        end = time.monotonic() + 3
                        while not fut.done() and time.monotonic() < end: pump(.01)
                        assert fut.done()
                        return fut.result().results[0]

                    assert not select('missing_clip').successful
                    assert call('play').success
                    latest('RUN', 4.5)
                    assert not select('dance1_subject3').successful  # no goal discontinuity
                    assert not call('play').success
                    latest('HOLD', 4.)
                    assert 'frame=200 ' in statuses[-1], statuses[-1]  # whole reference
                    assert 'preview_seconds=0.000000' in statuses[-1]
                    assert select('dance1_subject3').successful
                    assert call('play').success  # same process; clear old policy history
                    latest('RUN', 4.1)
                    # The user helper also supports an early switch: RETURN first,
                    # then parameter selection and /play. Keep synthetic SDK live.
                    helper_log = tmp / 'helper.log'
                    with helper_log.open('w') as out:
                        helper = subprocess.Popen(['/usr/bin/python3', str(Path(__file__).resolve().parents[1] /
                                                   'scripts/supermimic_motion.py'), 'walk2_subject4'],
                                                  stdout=out, stderr=subprocess.STDOUT)
                        deadline = time.monotonic() + 20
                        while helper.poll() is None and time.monotonic() < deadline: pump(.02)
                        if helper.poll() is None: helper.kill(); helper.wait()
                        assert helper.returncode == 0, helper_log.read_text()
                    latest('HOLD', 4.)
                    assert 'motion=walk2_subject4 ' in statuses[-1] and 'frame=80 ' in statuses[-1], statuses[-1]
                    # Every preloaded clip can be selected without restarting.
                    for name in cfg['motions']:
                        assert select(name).successful, name
                    assert call('stop').success; pump(.1); damping()
                elif automatic:
                    assert 'auto_start=waiting_mode_edge' in statuses[-1]
                    # Service class 4 also describes navigation. It must not
                    # count as debug mode or start the policy without raw topic 7.
                    state['mode'] = 6; pump(.5)
                    assert statuses[-1].startswith('IDLE')
                    state['mode'] = 2; pump(.4)  # class 3, confirmed non-debug
                    state['mode'] = 6; pump(.4)  # class 4, still no raw 7
                    count = len(received); pump(.25)
                    assert len(received) == count
                    state['mode'] = 7; latest('HOLD', 7.5)
                    assert any(s.startswith('HANDOFF') and ' frame=0 ' in s for s in statuses), statuses
                    assert 'auto_start=consumed' in statuses[-1]
                    if case == 'auto_skip_prepare':
                        assert not any(s.startswith('PREPARE') for s in statuses), statuses
                        active = [msg for msg in received if any(msg.kp)]
                        assert active
                        np.testing.assert_allclose(active[0].position, q, atol=1e-6)
                        np.testing.assert_allclose(received[-1].position, q, atol=1e-6)
                    state['mode'] = 1; pump(.4)
                    state['mode'] = 7; latest('FAULT', 2.)
                    assert not call('start').success
                elif case in ('jitter', 'below_late_threshold', 'slow_policy', 'sustained_late'):
                    assert delay_policy is not None
                    assert call('prepare').success; latest('READY', 2.7)
                    delay, calls = {'jitter': (25, 1), 'slow_policy': (45, 1), 'sustained_late': (25, 40), 'below_late_threshold': (21, 30)}[case]
                    policy_delay.write_text(f'{delay} {calls}')
                    assert call('start').success
                    if case in ('jitter', 'below_late_threshold'):
                        latest('HOLD', 4.)
                        assert not policy_delay.exists(), 'Run interposer did not execute'
                        fields = dict(t.split('=', 1) for t in statuses[-1].split() if '=' in t)
                        assert int(fields['late_updates']) >= 1 and int(fields['consecutive_late']) == 0, statuses[-1]
                        assert float(fields['max_policy_compute_ms']) >= delay, statuses[-1]
                    else:
                        latest('FAULT', 1.5); damping()
                        if case == 'sustained_late':
                            assert 'Policy continuously late: 20 consecutive updates' in statuses[-1], statuses[-1]
                        else:
                            assert ('ms hard limit' in statuses[-1] or 'Policy target timeout' in statuses[-1]
                                    or 'Policy result expired' in statuses[-1]), statuses[-1]
                elif case == 'pd_fault':
                    state['bad_pd'] = True; pump(.2)
                    assert not call('prepare').success
                    state['bad_pd'] = False; pump(.2)
                    assert call('prepare').success; latest('READY', 2.7)
                    state['bad_pd'] = True; latest('FAULT'); damping()
                    state['bad_pd'] = False; latest('FAULT')
                    assert not call('start').success
                elif case == 'target_envelope':
                    assert call('prepare').success; latest('READY', 2.7)
                    q[0] += .7; state['dq'] = .1; pump(.15)
                    msg = received[-1]
                    sdk_tau = msg.kp[0] * (msg.position[0] - q[0]) - msg.kd[0] * state['dq']
                    assert abs(sdk_tau + manifest['motors'][0]['effort_limit_nm']) < 1e-7, sdk_tau
                    assert msg.effort[0] == 0. and msg.kp[0] == kp[0]
                    assert call('stop').success; pump(.15); damping()
                    # SDK computes damping from its own current velocity, even
                    # though the message deliberately carries zero effort.
                    msg = received[-1]
                    assert msg.kp[0] == 0. and -msg.kd[0] * state['dq'] < 0.
                elif case == 'shutdown':
                    assert call('prepare').success; latest('READY', 2.7)
                    process.send_signal(signal.SIGINT)
                    pump(.5)
                    assert process.poll() == 0, 'Graceful shutdown failed'
                    damping()  # Last cached SDK command must not be position tracking.
                elif case == 'topic_loss':
                    assert call('prepare').success; latest('READY', 2.7)
                    state['mode_topic'] = False
                    latest('FAULT'); damping()
                    assert 'raw mode topic unavailable/stale' in statuses[-1]
                    assert 'mode_class=4' in statuses[-1]  # query stays healthy
                elif case == 'query_loss':
                    assert call('prepare').success; latest('READY', 2.7)
                    node.destroy_service(mode_server); mode_server = None
                    latest('FAULT'); damping()
                    assert 'mode query unavailable/stale' in statuses[-1]
                else:
                    state['head'] = False; pump(.2)
                    rejected = call('prepare')
                    assert not rejected.success and 'head_' in rejected.message, rejected
                    state['head'] = True; pump(.2)
                    if not dry:
                        state['feedback'] = False; pump(.2)
                        rejected = call('prepare')
                        assert not rejected.success and 'PD feedback' in rejected.message, rejected
                        state['feedback'] = True; pump(.2)
                    assert call('prepare').success; latest('READY', 2.7)
                    if block_publish:
                        flag.touch(); latest('READY', 1.4)
                        assert not flag.exists()
                        assert float(statuses[-1].split('max_publish_ms=')[1].split()[0]) >= 340
                    q[:27] += .02; q[27:] += .03; pump(.12)
                    msg = received[-1]
                    assert msg.name == names
                    np.testing.assert_allclose(msg.kp, kp); np.testing.assert_allclose(msg.kd, kd)
                    np.testing.assert_allclose(msg.position[:27], q[:27] - .02, atol=1e-7)
                    np.testing.assert_allclose(msg.position[27:], [.13, -.21], atol=1e-7)
                    np.testing.assert_allclose(msg.effort, 0.)
                    # SDK 2.2.37 overwrites effort with this PD; old zero-gain
                    # deployment would produce zero, so this catches that defect.
                    actual_pd = np.array(msg.kp[:12]) * (np.array(msg.position[:12]) - q[:12])
                    np.testing.assert_allclose(actual_pd, -.02 * kp[:12], atol=1e-6)
                    q[:27] -= .02; q[27:] -= .03; pump(.1)
                    assert call('start').success
                    latest('HANDOFF', .8)
                    assert ' frame=0 ' in statuses[-1]
                    latest('HOLD', 3.)
                    assert call('stop').success; pump(.1); damping()
                    assert call('prepare').success; latest('READY', 2.7)
                    state['imu'] = False; latest('FAULT'); damping()
                    state['imu'] = True; latest('FAULT')
                    assert not call('start').success
                    # Mode query observes the exit while ALL SDK topics stop.
                    state['mode'] = 1; pump(.3); count = len(received); pump(.3)
                    assert len(received) == count, 'Commands published outside queried mode 7'
                print(json.dumps(dict(case=case, dry_run=dry, checks='passed', commands=len(received))), flush=True)
            except BaseException:
                print(log_path.read_text(), flush=True)
                raise
            finally:
                process.send_signal(signal.SIGINT); process.wait(timeout=8)
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--block-publish', type=Path)
    p.add_argument('--auto-tests', action='store_true')
    p.add_argument('--delay-policy', type=Path, help='Test-only ORT Run delay interposer')
    p.add_argument('--case', choices=['manual', 'pd_fault', 'query_loss', 'shutdown', 'auto', 'auto_skip_prepare', 'target_envelope', 'topic_loss', 'jitter', 'below_late_threshold', 'slow_policy', 'sustained_late', 'motion_switch', 'joystick', 'end_hold'])
    args = p.parse_args()
    os.environ['ROS_DOMAIN_ID'] = '186'; os.environ['ROS_LOCALHOST_ONLY'] = '1'
    cases = [('auto', False), ('auto', True)] if args.auto_tests else [
        ('manual', True), ('manual', False), ('pd_fault', False), ('query_loss', False), ('shutdown', False), ('topic_loss', False)]
    if args.case: cases = [(args.case, False)]
    for case, dry in cases:
        test(args.runner.resolve(), args.config.resolve(), dry, case, args.block_publish, args.delay_policy)

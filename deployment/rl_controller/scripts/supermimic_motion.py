#!/usr/bin/env python3
"""List or play a preloaded clip without restarting supermimic_runner.

An explicit play request finishes an active clip, waits for RETURN/HOLD, selects
the named reference, then calls /play (prepare + handoff + tracking). No SDK mode
changes, fault recovery, or controller restarts are performed.
"""
import argparse
import time

import rclpy
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion', nargs='?')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--node', default='/supermimic_controller')
    parser.add_argument('--timeout', type=float, default=45.)
    args = parser.parse_args()
    if not args.list and not args.motion:
        parser.error('provide a motion name, or --list')
    if not 1 <= args.timeout <= 300:
        parser.error('--timeout must be in 1..300 seconds')
    rclpy.init()
    node = rclpy.create_node('supermimic_motion_request')
    status = {'text': '', 'received': 0.}

    def received(msg):
        status.update(text=msg.data, received=time.monotonic())

    sub = node.create_subscription(String, args.node + '/status', received, 1)
    get_parameters = node.create_client(GetParameters, args.node + '/get_parameters')
    set_parameters = node.create_client(SetParameters, args.node + '/set_parameters')

    def wait_future(future):
        deadline = time.monotonic() + args.timeout
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
        if not future.done():
            raise RuntimeError('Request timed out; inspect controller status before retrying')
        return future.result()

    def wait_phase(allowed):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if time.monotonic() - status['received'] > 3.:
                continue
            phase = status['text'].split()[0]
            if phase == 'FAULT':
                raise RuntimeError(status['text'])
            if phase in allowed:
                return phase
        raise RuntimeError('Timed out waiting for ' + '/'.join(sorted(allowed)) + ': ' + status['text'])

    def trigger(name, allow_finished=False):
        client = node.create_client(Trigger, args.node + '/' + name)
        try:
            if not client.wait_for_service(timeout_sec=5.):
                raise RuntimeError('Service unavailable: ' + name)
            result = wait_future(client.call_async(Trigger.Request()))
            if not result.success:
                if allow_finished and result.message == 'No active motion to finish':
                    return  # It may have completed since the last 1 Hz status.
                raise RuntimeError(name + ': ' + result.message)
            print(result.message, flush=True)
        finally:
            node.destroy_client(client)

    try:
        if not get_parameters.wait_for_service(timeout_sec=5.) or not set_parameters.wait_for_service(timeout_sec=5.):
            raise RuntimeError('Controller parameter services unavailable')
        request = GetParameters.Request(); request.names = ['available_motions']
        names = list(wait_future(get_parameters.call_async(request)).values[0].string_array_value)
        if args.list:
            print('\n'.join(names))
            return
        if args.motion not in names:
            raise RuntimeError('Unknown motion; available: ' + ', '.join(names))
        phase = wait_phase({'IDLE', 'PREPARE', 'READY', 'HANDOFF', 'RUN', 'SWITCH', 'RETURN', 'HOLD', 'DONE'})
        if phase in {'PREPARE', 'READY', 'HANDOFF', 'RUN', 'SWITCH'}:
            trigger('finish', allow_finished=True)
            wait_phase({'HOLD', 'DONE'})
        elif phase == 'RETURN':
            wait_phase({'HOLD', 'DONE'})
        request = SetParameters.Request()
        request.parameters = [Parameter('motion', value=args.motion).to_parameter_msg()]
        result = wait_future(set_parameters.call_async(request))
        if not result.results[0].successful:
            raise RuntimeError(result.results[0].reason)
        print('Selected: ' + args.motion, flush=True)
        if not args.select_only:
            trigger('play')
            # Ignore the old HOLD sample while waiting for the explicit play request.
            wait_phase({'HANDOFF', 'RUN'})
            print('Tracking: ' + args.motion, flush=True)
    finally:
        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, KeyboardInterrupt) as error:
        raise SystemExit(str(error))

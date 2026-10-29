"""Closed-loop native MuJoCo checks of the C++ live reference bridge (no ROS)."""
import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from supermimic.deployment.sim2sim import Sim2Sim, observation, rotation


def evaluate(args, source, destination, cfg):
    sim = Sim2Sim(args.sim_bundle, threads=1)
    sim.reset(source, args.start_frame)
    before = round(args.before_seconds / sim.dt)
    bridge_steps = int(np.ceil(args.switch_seconds / sim.dt))
    bank = {name: (args.config.parent / path).resolve() for name, path in cfg['motions'].items()}
    for name in (source, destination):
        metadata = json.loads((bank[name] / 'manifest.json').read_text())
        for key in ('checkpoint_sha256', 'motors', 'joint_names', 'goal_dim', 'control_dt'):
            if metadata[key] != sim.meta[key]:
                raise ValueError(f'Sim/hardware contract mismatch: {name}/{key}')
        if metadata['anchor_obs'] or metadata['goal_dim'] != 640:
            raise ValueError('Live transition test requires the proprio-only 640-goal actor')
    # Frame 0 is the bridge endpoint; frames 1..L-1 then play exactly once.
    # A short destination normally finishes before the requested 10 seconds.
    after_steps = min(round(args.after_seconds/sim.dt), metadata['frames']-1)
    count = before + bridge_steps + after_steps
    process = subprocess.Popen([str(args.probe.resolve()), str(bank[source]), str(bank[destination]),
                                str(before), str(args.switch_seconds), str(args.start_frame)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    kp, kd, effort = (np.array([m[key] for m in sim.meta['motors']])
                      for key in ('kp', 'kd', 'effort_limit_nm'))
    if args.explicit_pd:
        # Match SDK 2.2.37 integer gains and native 1 kHz bounded effort PD.
        kp, kd = np.trunc(kp), np.trunc(kd)
        sim.model.opt.timestep = .001
        sim.decimation = 20
        sim.model.actuator_gainprm[:] = 0
        sim.model.actuator_gainprm[:, 0] = 1
        sim.model.actuator_biasprm[:] = 0
    maxima = dict(observation_error=0., actor_error=0., height_error_m=0., tilt_error=0.,
                  joint_rmse_rad=0., target_step_rad=0.)
    rows, previous_target, failed = [], None, None
    try:
        for tick in range(count):
            record = np.r_[sim.data.qpos[7:], sim.data.qvel[6:], [np.nan]*3,
                           sim.data.qpos[3:7], sim.data.qvel[3:6]].astype('<f8')
            process.stdin.write(record.tobytes()); process.stdin.flush()
            raw = process.stdout.read((870 + 640 + 54 + 36)*4)
            if len(raw) != (870 + 640 + 54 + 36)*4:
                raise RuntimeError(f'C++ probe stopped: {process.poll()}')
            out = np.frombuffer(raw, dtype='<f4')
            obs, goal, action, target, ref = out[:870], out[870:1510], out[1510:1537], out[1537:1564], out[1564:]
            error = float(np.max(np.abs(sim.history.packed() - obs)))
            maxima['observation_error'] = max(maxima['observation_error'], error)
            np.testing.assert_allclose(obs, sim.history.packed(), atol=2e-5, rtol=3e-6)
            if tick % 50 == 0 or abs(tick-before) < 3 or abs(tick-before-bridge_steps) < 3:
                with torch.inference_mode():
                    expected = sim.actor(torch.from_numpy(obs.copy()[None]), torch.from_numpy(goal.copy()[None]))[0].numpy()
                error = float(np.max(np.abs(action-expected)))
                maxima['actor_error'] = max(maxima['actor_error'], error)
                np.testing.assert_allclose(action, expected, atol=2e-4, rtol=2e-5)
            if previous_target is not None:
                maxima['target_step_rad'] = max(maxima['target_step_rad'], float(np.max(np.abs(target-previous_target))))
            previous_target = target.copy()
            for substep in range(sim.decimation):
                # Keep the exported delay's 5 ms sampling period even when PD
                # is evaluated at 1 kHz (0/5/10 ms, rather than 0/1/2 ms).
                if not args.explicit_pd or substep % 5 == 0:
                    delayed = sim.command_delay.apply(target)
                if args.explicit_pd:
                    tau = np.clip(kp*(delayed-sim.data.qpos[7:])-kd*sim.data.qvel[6:], -effort, effort)
                    sim.data.ctrl[:] = tau[sim.actuator_to_joint]
                else:
                    sim.data.ctrl[:] = delayed[sim.actuator_to_joint]
                mujoco.mj_step(sim.model, sim.data)
            mujoco.mj_forward(sim.model, sim.data)
            height = float(abs(sim.data.qpos[2]-ref[2]))
            tilt = float(abs(rotation(sim.data.qpos[3:7])[2, 2]-rotation(ref[3:7])[2, 2]))
            rmse = float(np.sqrt(np.mean((sim.data.qpos[7:]-ref[7:34])**2)))
            for key, value in [('height_error_m', height), ('tilt_error', tilt), ('joint_rmse_rad', rmse)]:
                maxima[key] = max(maxima[key], value)
            rows.append(dict(tick=tick, frame=int(ref[34]), bridge=bool(ref[35]), height=height, tilt=tilt))
            sim.history.append(observation(sim.data.qpos, sim.data.qvel, action, sim.q0))
            if height > sim.meta['height_threshold'] or tilt > sim.meta['tilt_threshold']:
                failed = dict(tick=tick, seconds=(tick+1)*sim.dt,
                              stage='source' if tick < before else 'bridge' if ref[35] else 'destination')
                break
    finally:
        process.stdin.close(); process.stdout.close()
        if process.wait(timeout=5) != 0: raise RuntimeError('C++ probe failed')
    return dict(source=source, destination=destination, explicit_pd=args.explicit_pd,
                checkpoint_sha256=sim.meta['checkpoint_sha256'],
                start_frame=args.start_frame, before_seconds=args.before_seconds,
                switch_seconds=args.switch_seconds, after_seconds=after_steps*sim.dt,
                requested_after_seconds=args.after_seconds,
                completed=failed is None, failed=failed, simulated_seconds=len(rows)*sim.dt,
                bridge_updates=sum(r['bridge'] for r in rows), maxima=maxima,
                root_position_supplied=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-bundle', type=Path, default=Path('outputs/casbot02_primitive_020000/sim2sim'))
    p.add_argument('--config', type=Path, default=Path('rl_controller/config/supermimic_walk1_subject2_proprio.yaml'))
    p.add_argument('--probe', type=Path, default=Path('rl_controller/build/supermimic_ros/supermimic_live_switch_probe'))
    p.add_argument('--pair', nargs=2, action='append', help='Repeat for selected source/destination pairs; default all X edges')
    p.add_argument('--before-seconds', type=float, default=5.)
    p.add_argument('--after-seconds', type=float, default=10.)
    p.add_argument('--switch-seconds', type=float, default=.8)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--explicit-pd', action='store_true')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.before_seconds < .02 or args.after_seconds < 0 or not .2 <= args.switch_seconds <= 5.:
        p.error('Require before>=.02s, after>=0 and switch within .2..5s')
    cfg = yaml.safe_load(args.config.read_text())
    order = cfg['joystick_motion_order']
    pairs = args.pair or list(zip(order, order[1:]+order[:1]))
    reports = []
    for source, destination in pairs:
        report = evaluate(args, source, destination, cfg)
        reports.append(report); print(json.dumps(report), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(results=reports, all_completed=all(r['completed'] for r in reports)), indent=2)+'\n')


if __name__ == '__main__':
    main()

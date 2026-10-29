"""Evaluate every exported motion in native CPU MuJoCo without using training GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

_sim = None


def initialize(bundle):
    global _sim
    from supermimic.deployment.sim2sim import Sim2Sim

    os.nice(10)
    _sim = Sim2Sim(Path(bundle), threads=1)


def evaluate(name):
    import numpy as np

    began = time.perf_counter()
    # Reset the delay RNG independently for every clip. Worker scheduling must
    # not change a clip's latency sequence or outcome.
    _sim.command_delay.rng = np.random.default_rng(0)
    _sim.reset(name)
    while not _sim.finished:
        _sim.step()
    report = _sim.report()
    height, tilt = _sim.metrics[-1][:2]
    report.update(reference_seconds=_sim.reference.length*_sim.dt,
                  reference_frames=_sim.reference.length,
                  duration_fraction=_sim.steps/_sim.reference.length,
                  terminal_height_error_m=float(height),terminal_tilt_error=float(tilt),
                  termination=('height' if height>_sim.meta['height_threshold'] else
                               'tilt' if tilt>_sim.meta['tilt_threshold'] else 'clip_end'),
                  wall_seconds=time.perf_counter()-began)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    bundle = args.bundle.resolve()
    metadata = json.loads((bundle/'manifest.json').read_text())
    names = list(metadata['motions'])
    destination = args.report or bundle/'evaluation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, errors = {}, {}
    began = time.perf_counter()

    def save():
        ordered = [rows[name] for name in names if name in rows]
        completed = [row['clip'] for row in ordered if row['completed_clip'] and not row['failed']]
        failed = [row['clip'] for row in ordered if row['failed']]
        result = dict(checkpoint_iteration=metadata['checkpoint_iteration'],
                      checkpoint_sha256=metadata['checkpoint_sha256'], bundle=str(bundle),
                      sampled_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      backend='native MuJoCo CPU / TorchScript; not GPU Warp benchmark',
                      protocol=dict(start_frame=0,reset_to_reference_pose_and_velocity=True,
                                    delay_seed_per_clip=0,actuator_delay=metadata.get('actuator_delay',False),
                                    actuator_delay_mode=metadata.get('actuator_delay_mode','episode'),
                                    observation_noise=False,action_noise=False,domain_randomization=False,
                                    foot_friction=1.,height_threshold=metadata['height_threshold'],
                                    tilt_threshold=metadata['tilt_threshold'],
                                    definition='Complete full clip without reference-relative height/tilt termination; no global path-accuracy threshold'),
                      total=len(names),evaluated=len(rows),completed=len(completed),failed=len(failed),
                      completion_rate=len(completed)/len(names),complete_clips=completed,failed_clips=failed,
                      errors=errors,results=ordered,wall_seconds=time.perf_counter()-began)
        temporary = destination.with_suffix(destination.suffix+'.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(destination)

    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(str(bundle),)) as executor:
        pending = {executor.submit(evaluate,name):name for name in names}
        for future in as_completed(pending):
            name = pending[future]
            try:
                row = future.result();rows[name]=row
                print(f'{len(rows)+len(errors)}/{len(names)} {name}: '
                      f'{"COMPLETE" if row["completed_clip"] and not row["failed"] else "FAIL"} '
                      f'{row["steps"]}/{row["reference_frames"]} ({row["simulated_seconds"]:.2f}s)',flush=True)
            except Exception as error:
                errors[name] = f'{type(error).__name__}: {error}'
                print(f'{name}: ERROR {errors[name]}',flush=True)
            save()
    print(f'Results: {destination}',flush=True)
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()

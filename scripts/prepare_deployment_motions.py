"""Prepare offline-selectable reference clips sharing a fixed hardware actor.

Keeps the verified model/motor/observation contract from an existing bundle.
Does not retrain, re-export ONNX, start ROS, or operate a robot.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import torch

from supermimic.envs.mdp.motion_library import RefState, goal_features


DEFAULT_CLIPS = ['walk1_subject2', 'walk1_subject1', 'walk2_subject4',
                 'walk3_subject1', 'walk4_subject1', 'dance1_subject3', 'dance1_subject2']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-bundle', type=Path, default=Path(
        'rl_controller/config/model/supermimic_walk1_subject2_proprio'))
    parser.add_argument('--npz-dir', type=Path, default=Path('data/npz/lafan1_casbot02_codex'))
    parser.add_argument('--output', type=Path, default=Path('rl_controller/config/model/supermimic_motion_bank'))
    parser.add_argument('--clips', nargs='+', default=DEFAULT_CLIPS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    meta = json.loads((args.base_bundle / 'manifest.json').read_text())
    if meta['format'] != 'supermimic-hardware-v1' or meta['robot'] != 'casbot02':
        raise ValueError('Expected a Casbot02 hardware bundle')
    actor = args.base_bundle / meta['actor_file']
    if digest(actor) != meta['actor_sha256']:
        raise ValueError('Base actor hash mismatch')
    if len(set(args.clips)) != len(args.clips) or any(Path(c).name != c or c in ('.', '..') for c in args.clips):
        raise ValueError('Clip names must be unique plain filenames without extensions')
    q0 = torch.tensor([m['q0'] for m in meta['motors']], dtype=torch.float32)
    pending = []
    for clip in args.clips:
        path = args.npz_dir / f'{clip}.npz'
        with np.load(path, allow_pickle=False) as data:
            if data['joint_names'].tolist() != meta['joint_names']:
                raise ValueError(f'{clip}: joint order mismatch')
            if not np.isclose(float(data['fps'].item()), 1. / meta['control_dt']):
                raise ValueError(f'{clip}: expected 50 Hz reference')
            root = data['body_names'].tolist().index('base_link')
            arrays = [data['body_pos_w'][:, root], data['body_quat_w'][:, root],
                      data['joint_pos'], data['body_lin_vel_w'][:, root],
                      data['body_ang_vel_w'][:, root], data['joint_vel']]
            ref = RefState(*[torch.from_numpy(a.copy()).float() for a in arrays])
            with torch.inference_mode():
                features = goal_features(ref, q0).numpy()
            records = np.concatenate([arrays[0], arrays[1], arrays[2], arrays[5], features], axis=1).astype('<f4')
        if records.shape[1] != 125 or len(records) < 2 or not np.isfinite(records).all():
            raise ValueError(f'{clip}: invalid motion shape/data')
        if not np.allclose(np.linalg.norm(records[:, 3:7], axis=1), 1., atol=.01):
            raise ValueError(f'{clip}: invalid root quaternion')
        blob = b'SMREF001' + struct.pack('<II', len(records), 125) + records.tobytes()
        # The original successful clip must remain byte-identical to its deployment.
        if clip == meta['clip'] and blob != (args.base_bundle / meta['motion_file']).read_bytes():
            raise ValueError(f'{clip}: reference differs from existing deployment; check source dataset')
        motion_meta = copy.deepcopy(meta)
        motion_meta.update(clip=clip, frames=len(records), actor_file='../policy/actor.onnx',
                           motion_file='motion.bin', motion_sha256=hashlib.sha256(blob).hexdigest(),
                           source_motion=str(path.resolve()), source_motion_sha256=digest(path))
        pending.append((clip, blob, motion_meta))
    shared = args.output / 'policy' / 'actor.onnx'
    if shared.exists() and digest(shared) != meta['actor_sha256']:
        raise ValueError('Motion bank contains another policy; choose a new --output directory')
    shared.parent.mkdir(parents=True, exist_ok=True)
    if not shared.exists(): shutil.copyfile(actor, shared)
    catalog = dict(checkpoint_iteration=meta['checkpoint_iteration'], checkpoint_sha256=meta['checkpoint_sha256'],
                   actor_sha256=meta['actor_sha256'], actor_file='policy/actor.onnx', clips=[])
    for clip, blob, motion_meta in pending:
        folder = args.output / clip
        folder.mkdir(exist_ok=True)
        (folder / 'motion.bin').write_bytes(blob)
        (folder / 'manifest.json').write_text(json.dumps(motion_meta, indent=2) + '\n')
        catalog['clips'].append(dict(name=clip, frames=motion_meta['frames'],
                                    seconds=round(motion_meta['frames'] * meta['control_dt'], 2),
                                    bundle=str(folder), motion_sha256=motion_meta['motion_sha256']))
    (args.output / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps(catalog, indent=2))


if __name__ == '__main__':
    main()

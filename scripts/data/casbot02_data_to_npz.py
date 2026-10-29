"""Convert SDK .data clips to 50 Hz, 27-DOF SuperMimic NPZ files on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from supermimic.data.casbot02_data import convert_data
from supermimic.robots import get_robot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/npz/lafan1_casbot02_codex'))
    parser.add_argument('--input-fps', type=float, default=1000., help='Fallback for names without _NNNHZ; explicit filename rates take priority')
    parser.add_argument('--output-fps', type=float, default=50.)
    parser.add_argument('--pattern', default='*.data')
    parser.add_argument('--no-clamp-joints', action='store_true', help='Preserve source joint angles even outside hard limits')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--report', type=Path, default=Path('outputs/casbot02_data_conversion/report.json'))
    args = parser.parse_args()
    files = sorted(args.input_dir.expanduser().resolve().glob(args.pattern))
    if not files: parser.error('No matching .data files')
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    paths = [output/(source.stem+'.npz') for source in files]
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.overwrite: parser.error(f'Outputs already exist (use --overwrite): {existing}')
    model = get_robot('casbot02').model()
    report = dict(input_dir=str(args.input_dir.resolve()), output_dir=str(output), robot='casbot02',
                  joint_count=27, body_count=model.nbody-1, output_fps=args.output_fps,
                  root_convention='AMP converter: x=raw[1], y=raw[0], z=raw[2]; pitch=raw[3]-auto_offset, roll=raw[4], yaw=-raw[5]',
                  joint_columns_zero_based=[*range(12, 24), 60, *range(24, 31), *range(31, 38)], clips=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(files, paths):
        explicit = re.search(r'_(\d+(?:\.\d+)?)HZ(?:$|_)', source.stem, flags=re.I)
        fps = float(explicit[1]) if explicit else args.input_fps
        arrays, details = convert_data(source, model, fps, args.output_fps, clamp_joints=not args.no_clamp_joints)
        for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'):
            if not np.isfinite(arrays[key]).all(): raise ValueError(f'{source}: nonfinite {key}')
        if not np.allclose(np.linalg.norm(arrays['body_quat_w'], axis=-1), 1., atol=2e-6):
            raise ValueError(f'{source}: invalid body quaternion norms')
        fd, tmp = tempfile.mkstemp(dir=output, suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as stream: np.savez_compressed(stream, **arrays)
            os.replace(tmp, destination)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        details.update(source=str(source), output=str(destination), source_sha256=digest(source),
                       output_sha256=digest(destination), input_fps_source='filename' if explicit else 'fallback')
        report['clips'].append(details)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        print(f'{source.name}: {details["source_frames"]} @ {fps:g} Hz -> {details["output_frames"]} @ {args.output_fps:g} Hz '
              f'({details["seconds"]:.2f}s), clamped {details["frames_clamped"]} frames', flush=True)
    print(f'Wrote {len(files)} clips to {output}\nReport: {args.report}')


if __name__ == '__main__': main()

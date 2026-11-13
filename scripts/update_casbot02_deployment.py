"""Export, validate and optionally install a Casbot02 proprio checkpoint locally.

Run from the project with its uv environment. No ROS/hardware commands or remote
uploads are sent. --apply replaces the local model bundles after validation;
--build-arm64 additionally rebuilds the local package used by install_remote.sh.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def run(command, log, cwd=ROOT):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, command)), cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-8000:], file=sys.stderr)
        raise RuntimeError(f'Command failed ({result.returncode}); full log: {log}')


def menu_text(text, motions, order):
    """Replace only the two YAML entries, keeping other settings/comments."""
    root = yaml.compose(text)
    replacements = []
    for key, value in root.value:
        if key.value == 'motions':
            fragment = yaml.safe_dump({'motions': motions}, allow_unicode=True, sort_keys=False).rstrip()+'\n\n'
        elif key.value == 'joystick_motion_order':
            fragment = 'joystick_motion_order: '+yaml.safe_dump(order,allow_unicode=True,default_flow_style=True,width=10000).strip()
        else:
            continue
        replacements.append((key.start_mark.index,value.end_mark.index,fragment))
    if len(replacements) != 2: raise ValueError('Config must contain motions and joystick_motion_order')
    for start,end,fragment in sorted(replacements,reverse=True): text=text[:start]+fragment+text[end:]
    return text


def verify_contract(new_meta, old_meta, accept_joint_limit_update=False, accept_goal_update=False,
                    accept_orientation_update=False, accept_motor_update=False):
    """Keep actuator/interface guards, with an explicit opt-in for new limits."""
    if new_meta['motors'] != old_meta['motors'] and not accept_motor_update:
        raise ValueError('Export changed motors; verified training assets and --accept-motor-update required')
    for key in ('joint_names', 'sdk_joint_names',
                'history', 'future_frames', 'control_dt'):
        if new_meta[key] != old_meta[key]:
            raise ValueError(f'Export changed {key}; verify training-time robot constants before updating')
    before_obs = (old_meta['obs_dim'], old_meta.get('anchor_obs', False), old_meta.get('anchor_ori_obs', False))
    after_obs = (new_meta['obs_dim'], new_meta.get('anchor_obs', False), new_meta.get('anchor_ori_obs', False))
    if before_obs != after_obs:
        supported = ({before_obs, after_obs} == {(870, False, False), (930, False, True)}
                     and all(meta['goal_dim'] == 690 and meta.get('goal_encoding') == 'reference_heading_v1'
                             for meta in (old_meta, new_meta)))
        if not (accept_orientation_update and supported):
            raise ValueError('Export changed obs_dim/orientation feedback; supported 870<->930 migration '
                             'requires --accept-orientation-update')
    before = (old_meta['goal_dim'], old_meta.get('goal_encoding', 'per_frame_v1'))
    after = (new_meta['goal_dim'], new_meta.get('goal_encoding', 'per_frame_v1'))
    if before != after:
        supported = (before == (640, 'per_frame_v1') and after == (690, 'reference_heading_v1')
                     and not any(meta.get(flag, False) for meta in (new_meta, old_meta)
                                 for flag in ('anchor_obs', 'anchor_ori_obs')))
        if not (accept_goal_update and supported):
            raise ValueError('Export changed goal_dim/encoding; verified 640->690 migration requires --accept-goal-update')
    changes = {key: {'before': old_meta[key], 'after': new_meta[key]}
               for key in ('joint_limits', 'soft_joint_limits')
               if new_meta[key] != old_meta[key]}
    if changes and not accept_joint_limit_update:
        raise ValueError('Export changed joint limits; verify training-time XML, then use '
                         '--accept-joint-limit-update to record and apply these limits')
    return changes


def install_bundles(pairs, config, text, backup):
    """Prepare copies first, then rename each directory; roll back on errors."""
    backup.mkdir()
    shutil.copy2(config, backup/config.name)
    token = uuid.uuid4().hex
    config_tmp = config.with_name('.'+config.name+'.'+token)
    pending, installed = [], []
    try:
        for source, destination in pairs:
            shutil.copytree(destination, backup/destination.name)
            temp = destination.with_name('.'+destination.name+'.new-'+token)
            pending.append(temp)
            shutil.copytree(source, temp)
        for (_, destination), temp in zip(pairs, pending):
            old = destination.with_name('.'+destination.name+'.old-'+token)
            destination.rename(old)
            installed.append((destination, old))
            temp.rename(destination)
        config_tmp.write_text(text)
        os.replace(config_tmp, config)
    except BaseException:
        for destination, old in reversed(installed):
            if destination.exists(): shutil.rmtree(destination)
            old.rename(destination)
        shutil.copy2(backup/config.name, config)
        raise
    finally:
        for temp in pending:
            if temp.exists(): shutil.rmtree(temp)
        if config_tmp.exists(): config_tmp.unlink()
    for _, old in installed: shutil.rmtree(old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'rl_controller/config/supermimic_walk1_subject2_proprio.yaml')
    parser.add_argument('--npz-dir', type=Path, help='Default: data.npz_dir saved in checkpoint')
    parser.add_argument('--robot-assets-dir', type=Path,
                        help='Archived Casbot02 assets matching training (XML, meshes, motor_parameters.json)')
    parser.add_argument('--accept-motor-update', action='store_true',
                        help='Accept explicitly verified training motor changes; requires --robot-assets-dir')
    parser.add_argument('--output', type=Path, help='New, empty export directory; default timestamped under outputs/')
    parser.add_argument('--prepend-motions', nargs='+', help='Add/prepend NPZ clip names to the existing joystick cycle')
    parser.add_argument('--initial-motion', help='Default clip on automatic startup; must be in the resulting motion menu')
    parser.add_argument('--motion-order', nargs='+', help='Replace the joystick cycle exactly; other configured motions remain available')
    parser.add_argument('--remove-motions', nargs='+',
                        help='Remove configured clips from the deployment bank and joystick cycle; training data is untouched')
    parser.add_argument('--accept-goal-update', action='store_true', help='Migrate the legacy 640-goal proprio bundle to verified 690-goal heading-relative encoding')
    parser.add_argument('--accept-orientation-update', action='store_true',
                        help='Switch between 870+690 and 930+690 (pelvis rot6d in every history frame)')
    parser.add_argument('--accept-joint-limit-update', action='store_true',
                        help='Accept verified training-time joint limit changes; motor/interface changes still fail')
    parser.add_argument('--apply', action='store_true', help='Back up and update LOCAL deployment bundles after checks')
    parser.add_argument('--build-arm64', action='store_true', help='Also build local ARM64 package; requires --apply and Docker')
    args = parser.parse_args()
    if args.build_arm64 and not args.apply: parser.error('--build-arm64 requires --apply')
    if args.accept_motor_update and args.robot_assets_dir is None:
        parser.error('--accept-motor-update requires an explicit --robot-assets-dir snapshot')
    os.chdir(ROOT)
    os.environ.pop('PYTHONPATH', None)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    torch.set_num_threads(2)
    ckpt, config = args.ckpt.resolve(strict=True), args.config.resolve(strict=True)
    cfg_text = config.read_text(); cfg = yaml.safe_load(cfg_text)
    main_bundle = (config.parent/cfg['bundle']).resolve(strict=True)
    old_meta = json.loads((main_bundle/'manifest.json').read_text())
    removed = set(args.remove_motions or [])
    if len(removed) != len(args.remove_motions or []):
        raise ValueError('Duplicate --remove-motions entries')
    if any(Path(name).name != name or name in ('.', '..') for name in removed):
        raise ValueError('Removed motions must be plain clip names')
    if removed - set(cfg['motions']):
        raise ValueError(f'Cannot remove unknown motions: {sorted(removed - set(cfg["motions"]))}')
    cfg['motions'] = {name: path for name, path in cfg['motions'].items() if name not in removed}
    cfg['joystick_motion_order'] = [name for name in cfg['joystick_motion_order'] if name not in removed]
    if args.prepend_motions:
        new = args.prepend_motions
        if len(set(new)) != len(new): raise ValueError('Duplicate --prepend-motions entries')
        if any(Path(name).name != name or name in ('.','..') for name in new):
            raise ValueError('Motion names must be plain clip names')
        bank_relative = Path(cfg['motions'][old_meta['clip']]).parent
        cfg['motions'] = {**{name:cfg['motions'].get(name,str(bank_relative/name)) for name in new},**cfg['motions']}
        cfg['joystick_motion_order'] = [*new,*[name for name in cfg['joystick_motion_order'] if name not in new]]
    if args.motion_order:
        if len(set(args.motion_order)) != len(args.motion_order) or any(name not in cfg['motions'] for name in args.motion_order):
            raise ValueError('--motion-order requires unique configured motions')
        cfg['joystick_motion_order'] = args.motion_order
    initial = args.initial_motion or old_meta['clip']
    if initial not in cfg['joystick_motion_order'] or initial not in cfg['motions']:
        raise ValueError('Initial motion must be included in motions and joystick_motion_order')
    clips = list(dict.fromkeys([initial, *cfg['joystick_motion_order'], *cfg['motions']]))
    if any(Path(name).name != name or name in ('.', '..') for name in clips):
        raise ValueError('Motion names must be plain clip names')
    destinations = [(config.parent/cfg['motions'][clip]).resolve() for clip in clips]
    bank = destinations[0].parent
    if any(p.parent != bank or p.name != name for name, p in zip(clips, destinations)):
        raise ValueError('Expected one shared motion bank directory')
    model_root = (ROOT/'rl_controller/config/model').resolve()
    if main_bundle.parent != model_root or bank.parent != model_root or main_bundle == bank:
        raise ValueError('Local installation must use distinct bundles under rl_controller/config/model')
    if set(p.name for p in bank.iterdir() if (p/'manifest.json').is_file()) - set(clips) - removed:
        raise ValueError('Motion bank has unconfigured clips; include them in motions before updating')
    source_hash = digest(ckpt)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    saved = payload['config']; policy_cfg = saved['policy']; env_cfg = saved['env']
    if (saved.get('robot'), policy_cfg['num_joints'], policy_cfg['history'],
        env_cfg.get('anchor_obs', False), env_cfg['future_frames']) != (
            'casbot02', 27, 10, False, 10) or policy_cfg['goal_dim'] != 690:
        raise ValueError('This updater requires Casbot02 870/930 history +690 goal ->27 without root position')
    orientation = env_cfg.get('anchor_ori_obs', False)
    scales = policy_cfg.get('obs_term_scale', [])
    if len(scales) != (6 if orientation else 5):
        raise ValueError('Saved observation scales disagree with the orientation history flag')
    obs_dim = (93 if orientation else 87) * policy_cfg['history']
    if not all(torch.isfinite(tensor).all() for tensor in payload['policy'].values()):
        raise ValueError('Actor contains NaN/Inf; choose another checkpoint')
    iteration = payload['iteration']; unit = payload.get('iteration_unit', 'unknown')
    npz_dir = (args.npz_dir or Path(saved['data']['npz_dir'])).resolve(strict=True)
    for clip in clips:
        if not (npz_dir/f'{clip}.npz').is_file(): raise FileNotFoundError(npz_dir/f'{clip}.npz')
    output = (args.output or ROOT/'outputs/casbot02_deployment'/f'{ckpt.parent.name}_{ckpt.stem}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    if output.exists(): raise FileExistsError(f'Choose a new --output directory: {output}')
    output.mkdir(parents=True)
    snapshot = output/ckpt.name
    shutil.copy2(ckpt, snapshot)
    if digest(snapshot) != source_hash or digest(ckpt) != source_hash:
        raise RuntimeError('Checkpoint changed while copying; choose a fixed numbered checkpoint')
    del payload
    logs = output/'logs'; logs.mkdir()
    sim, hardware, motion_bank = output/'sim2sim', output/'hardware', output/'motion_bank'
    python = Path(sys.executable)
    export_command = [python, ROOT/'scripts/export_sim2sim.py', '--ckpt', snapshot, '--npz-dir', npz_dir,
                      '--clips', *clips, '--output', sim]
    if args.robot_assets_dir is not None:
        export_command += ['--robot-assets-dir', args.robot_assets_dir.resolve(strict=True)]
    run(export_command, logs/'01_export_sim.log')
    run([python, ROOT/'scripts/export_hardware.py', '--bundle', sim, '--clip', initial,
         '--output', hardware], logs/'02_export_onnx.log')
    new_meta = json.loads((hardware/'manifest.json').read_text())
    # Older checkpoints do not embed motor constants: preserve the known local
    # contract, instead of silently exporting with changed q0/gains/scales.
    limit_changes = verify_contract(new_meta, old_meta, args.accept_joint_limit_update, args.accept_goal_update,
                                    args.accept_orientation_update, args.accept_motor_update)
    motor_changes = {name: {'before': before, 'after': after}
                     for name, before, after in zip(new_meta['joint_names'], old_meta['motors'], new_meta['motors'])
                     if before != after}
    if motor_changes:
        print(f'Applying verified training motor update for {len(motor_changes)} joints', flush=True)
    if limit_changes:
        print('Applying verified new joint limits; actuator/interface contract unchanged', flush=True)
    run([python, ROOT/'scripts/prepare_deployment_motions.py', '--base-bundle', hardware,
         '--npz-dir', npz_dir, '--output', motion_bank, '--clips', *clips], logs/'03_motion_bank.log')
    staged_cfg = copy.deepcopy(cfg)
    staged_cfg['bundle'] = str(hardware)
    staged_cfg['motions'] = {name: str(motion_bank/name) for name in clips}
    staged_config = output/'deployment.yaml'; staged_config.write_text(yaml.safe_dump(staged_cfg, sort_keys=False))
    run(['bash', ROOT/'rl_controller/scripts/build_supermimic.sh'], logs/'04_build_native.log')
    native = ROOT/'rl_controller/build/supermimic_ros'
    run([native/'supermimic_motion_bank_check', hardware, motion_bank], logs/'05_bank_check.log')
    run([native/'supermimic_live_switch_check', hardware, motion_bank/clips[1]], logs/'06_history_check.log')
    run([python, ROOT/'scripts/validate_hardware.py', '--sim-bundle', sim, '--hardware-bundle', hardware,
         '--probe', native/'supermimic_probe', '--seconds', '10', '--from-rest',
         '--report', output/'cpp_default_10s.json'], logs/'07_cpp_parity.log')
    run([python, ROOT/'scripts/validate_motion_transition.py', '--sim-bundle', sim,
         '--config', staged_config, '--switch-seconds', cfg.get('switch_blend_seconds', .8),
         '--explicit-pd', '--report', output/'x_transitions.json'], logs/'08_transitions.log')
    transitions = json.loads((output/'x_transitions.json').read_text())
    if not transitions['all_completed']:
        raise RuntimeError(f'Native transition check failed; deployment not replaced. See {output}/x_transitions.json')
    record = dict(checkpoint=str(ckpt), snapshot=str(snapshot), checkpoint_sha256=source_hash,
                  checkpoint_iteration=iteration, iteration_unit=unit, actor_sha256=new_meta['actor_sha256'],
                  input_dims=[obs_dim, policy_cfg['goal_dim']], actions=27, clips=clips, npz_dir=str(npz_dir),
                  orientation_history=orientation, previous_obs_dim=old_meta['obs_dim'],
                  goal_encoding=new_meta['goal_encoding'], previous_goal_dim=old_meta['goal_dim'],
                  initial_motion=initial,joystick_motion_order=cfg['joystick_motion_order'],
                  removed_motions=sorted(removed),
                  sim_bundle=str(sim), hardware_bundle=str(hardware), motion_bank=str(motion_bank),
                  validation=f'10s C++ parity from rest + {len(transitions["results"])}-edge short transitions; not full-clip success evaluation',
                  motor_contract_unchanged=not bool(motor_changes), motor_changes=motor_changes,
                  joint_limit_changes=limit_changes,
                  robot_assets_dir=str(args.robot_assets_dir.resolve()) if args.robot_assets_dir else None,
                  applied=False, arm64_built=False, uploaded_to_orin=False,
                  hardware_tested=False, limitations='Fixed switch points and nominal native CPU physics; complete SDK/firmware not modeled')
    record_path = output/'deployment.json'
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    if args.apply:
        # Abort if another local edit happened during validation.
        if config.read_text() != cfg_text: raise RuntimeError('Deployment config changed during export; retry with a new output')
        text = menu_text(cfg_text,cfg['motions'],cfg['joystick_motion_order']) if args.prepend_motions or args.motion_order or removed else cfg_text
        text = re.sub(r'^# Casbot02 proprio policy: .+$', f'# Casbot02 proprio policy: {obs_dim} history + {policy_cfg["goal_dim"]} reference -> 27 actions, 50 Hz.', text, flags=re.M)
        text = re.sub(r'^# (?:HANDOFF gains/scales and the new default pose match this training run\.|Gains, effort limits, scales and q0 match the verified(?: cpu\d+)? training asset snapshot\.)$',
                      '# Gains, effort limits, scales and q0 match the verified training asset snapshot.',
                      text, flags=re.M)
        text = re.sub(r'^# .+ / ckpt_\S+\.pt, .+\.$', f'# {ckpt.parent.name} / {ckpt.name}, {initial}.', text, flags=re.M)
        text = re.sub(r'^# Fixed snapshot exported at update \d+;.*$', f'# Fixed snapshot exported at update {iteration}; collision model: {env_cfg.get("collision_model", "mesh")}.', text, flags=re.M)
        text = re.sub(r'^# Initial automatic motion: .+$',
                      f'# Initial automatic motion: {initial} (from this bundle manifest).', text, flags=re.M)
        backup = output/'previous_deployment'
        catalog_path = motion_bank/'catalog.json'
        catalog = json.loads(catalog_path.read_text())
        for item in catalog['clips']: item['bundle'] = str((bank/item['name']).relative_to(ROOT))
        catalog_path.write_text(json.dumps(catalog, indent=2)+'\n')
        install_bundles([(hardware, main_bundle), (motion_bank, bank)], config, text, backup)
        record.update(applied=True, backup=str(backup))
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        print(f'Updated local deployment; backup: {backup}', flush=True)
        from deploy_casbot02_motion import sync_sim2sim_launcher
        sync_sim2sim_launcher(output, config)
    if args.build_arm64:
        run(['bash', './build_in_docker.sh', 'arm64'], logs/'09_build_arm64.log', ROOT/'rl_controller')
        install = ROOT/'rl_controller/build/install_arm64'
        if removed:
            # CMake directory installs copy new files without pruning old clips.
            targets = [str(Path('/rl/build/install_arm64') / bank.relative_to(ROOT/'rl_controller') / name)
                       for name in sorted(removed)]
            run(['docker', 'run', '--rm', '-v', f'{ROOT/"rl_controller"}:/rl',
                 'hub.casbot.dev/environment/casbot-amd64-builder:latest',
                 'rm', '-rf', '--', *targets], logs/'10_prune_removed_motions.log')
            if any((install/bank.relative_to(ROOT/'rl_controller')/name).exists() for name in removed):
                raise RuntimeError('Removed motion still exists in ARM64 install')
        for source in [config, *main_bundle.rglob('*'), *bank.rglob('*')]:
            if source.is_file():
                target = install/source.relative_to(ROOT/'rl_controller')
                if not target.is_file() or digest(source) != digest(target):
                    raise RuntimeError(f'Installed ARM64 bundle differs: {target}')
        record.update(arm64_built=True, arm64_runner_sha256=digest(install/'bin/supermimic_runner'))
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(f'Done: {record_path}\nCheckpoint: {iteration} {unit}, {obs_dim} + {policy_cfg["goal_dim"]} -> 27\nNo robot upload or motor commands were sent.', flush=True)


if __name__ == '__main__':
    main()

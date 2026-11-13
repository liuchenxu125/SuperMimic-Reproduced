#!/usr/bin/env python3
"""One-shot Casbot02 motion deployment: CSV -> NPZ -> local + ARM64 install.

Pipeline:
  1. Convert a LAFAN1 CSV, or a 61-column SDK motion (Base_Latation_X header), to 50 Hz NPZ.
  2. Symlink the NPZ into the shared staging directory used by the exporter.
  3. Insert the clip relative to an existing menu entry (or append when no anchor is given)
     and run scripts/update_casbot02_deployment.py --apply.
  4. Build and verify the complete ARM64 installation through the updater.
  Also point the local sim2sim launcher at the newly exported bundle.

The default checkpoint and training assets are pinned to cpu7/96000.
Source references default to the active export.
No ROS, no robot upload. Run from the repo root with the uv environment.

Example:
  env -u PYTHONPATH uv run python scripts/deploy_casbot02_motion.py \
    --csv data/data/dance1_subject2.csv --after 踏步
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "logs/casbot02_bones_seed_proprio_cpu7/ckpt_0096000.pt"
DEFAULT_ROBOT_ASSETS = ROOT / "outputs/cpu7_092000_deployment_20261109_preparation/training_assets"
DEFAULT_CONFIG = ROOT / "rl_controller/config/supermimic_walk1_subject2_proprio.yaml"
SIM2SIM_LAUNCHER = ROOT / "scripts/sim2sim_casbot02_bones_096000.sh"
ARM64_IMAGE = "hub.casbot.dev/environment/casbot-amd64-builder:latest"


def run(cmd, **kw):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True, **kw)


def active_deployment(config: Path) -> dict:
    """Follow the installed actor, never silently revert to a hard-coded checkpoint."""
    cfg = yaml.safe_load(config.read_text())
    meta = json.loads((config.parent / cfg['bundle'] / 'manifest.json').read_text())
    records = sorted((ROOT/'outputs/casbot02_deployment').glob('*/deployment.json'),
                     key=lambda path: path.stat().st_mtime, reverse=True)
    for path in records:
        record = json.loads(path.read_text())
        if (record.get('applied') and record.get('checkpoint_sha256') == meta['checkpoint_sha256']
                and record.get('actor_sha256') == meta['actor_sha256']):
            return record
    raise ValueError('No matching applied export record; pass --ckpt, --staging and '
                     '--robot-assets-dir explicitly')


def is_sdk_motion(path: Path) -> bool:
    """61-column CASBOT SDK clip: header starts with Base_Latation_X."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        first = next(csv.reader(stream), [])
    return bool(first) and first[0].strip() == "Base_Latation_X"


def place_npz(produced: Path, npz_out: Path) -> Path:
    if not produced.exists():
        raise FileNotFoundError(f"Converted NPZ not found: {produced}")
    if produced.resolve() != npz_out.resolve():
        produced.replace(npz_out)
    return npz_out


def convert_csv(source: Path, npz_out: Path, robot: str, input_fps: float, output_fps: float,
                robot_assets: Path | None = None) -> Path:
    """Convert one clip to NPZ. SDK 61-column files use the Casbot02 data converter."""
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    if is_sdk_motion(source):
        if robot != "casbot02":
            raise ValueError("SDK 61-column motions are Casbot02-only")
        # Filename _NNNhz wins inside the converter; --input-fps is only the fallback.
        # Keep the report off the historical conversion log.
        command = [
            sys.executable, ROOT / "scripts/data/casbot02_data_to_npz.py",
            "--input-dir", source.parent,
            "--pattern", source.name,
            "--input-fps", str(input_fps),
            "--output-fps", str(output_fps),
            "--output-dir", npz_out.parent,
            "--overwrite",
            "--report", npz_out.with_suffix(".conversion.json"),
        ]
        if robot_assets is not None:
            command += ['--robot-assets-dir', robot_assets]
        run(command)
        return place_npz(npz_out.parent / f"{source.stem}.npz", npz_out)
    run([
        sys.executable, ROOT / "scripts/data/csv_to_npz_batch.py",
        "--robot", robot,
        "--input-dir", source.parent,
        "--pattern", source.name,
        "--input-fps", str(input_fps),
        "--output-fps", str(output_fps),
        "--output-dir", npz_out.parent,
        "--workers", "1",
        "--overwrite",
    ])
    candidate = npz_out.parent / f"{source.stem}.npz"
    nested = npz_out.parent / source.stem / f"{source.stem}.npz"
    if candidate.exists():
        return place_npz(candidate, npz_out)
    if nested.exists():
        return place_npz(nested, npz_out)
    raise FileNotFoundError(f"Converted NPZ not found for {source}")


def link_into_staging(npz: Path, staging: Path, name: str) -> None:
    """Symlink staging/<name>.npz -> the converted NPZ (replace existing)."""
    staging.mkdir(parents=True, exist_ok=True)
    link = staging / f"{name}.npz"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(npz.resolve())
    print(f"[staging] {link} -> {npz}", flush=True)


def read_order(config: Path) -> list[str]:
    cfg = yaml.safe_load(config.read_text())
    order = cfg.get("joystick_motion_order") or []
    if not isinstance(order, list):
        raise ValueError("joystick_motion_order must be a list")
    return list(order)


def build_order(name: str, current: list[str], after: str | None, before: str | None, replace: bool) -> list[str]:
    if replace and name in current:
        current = [n for n in current if n != name]
    if name in current:
        # Already present; keep position unless an anchor is given.
        if after or before:
            current = [n for n in current if n != name]
        else:
            return current
    if after is not None:
        if after not in current:
            raise ValueError(f"--after '{after}' not in joystick_motion_order")
        idx = current.index(after) + 1
        return current[:idx] + [name] + current[idx:]
    if before is not None:
        if before not in current:
            raise ValueError(f"--before '{before}' not in joystick_motion_order")
        idx = current.index(before)
        return current[:idx] + [name] + current[idx:]
    return current + [name]


def ensure_motions_entry(config: Path, name: str) -> None:
    """Add motions: <name>: model/supermimic_motion_bank/<name> if missing."""
    text = config.read_text()
    cfg = yaml.safe_load(text)
    motions = cfg.setdefault("motions", {})
    if name in motions:
        return
    # Insert after the last motion line, matching the existing indentation.
    lines = text.splitlines(keepends=True)
    last_motion_idx = -1
    for i, line in enumerate(lines):
        # Heuristic: a motion entry is "  name: model/supermimic_motion_bank/name".
        # Match indented (2-space) lines that are not comments and reference the bank.
        if line.startswith("  ") and not line.startswith("    ") and ":" in line \
                and not line.lstrip().startswith("#") \
                and "model/supermimic_motion_bank/" in line:
            last_motion_idx = i
    if last_motion_idx < 0:
        raise RuntimeError("Could not find any motion entry in YAML to anchor the new line")
    new_line = f"  {name}: model/supermimic_motion_bank/{name}\n"
    lines.insert(last_motion_idx + 1, new_line)
    config.write_text("".join(lines))
    print(f"[yaml] added motions entry for {name}", flush=True)


def run_updater(ckpt: Path, npz_dir: Path, config: Path, order: list[str], output: Path | None,
                *, name: str | None = None, robot_assets: Path | None = None,
                build_arm64: bool = False, accept_motor_update: bool = False) -> Path:
    cmd = [
        sys.executable, ROOT / "scripts/update_casbot02_deployment.py",
        "--ckpt", ckpt,
        "--npz-dir", npz_dir,
        "--config", config,
        "--motion-order", *order,
        "--apply",
    ]
    if name is not None:
        # Let the updater add the entry transactionally after validation.
        cmd += ['--prepend-motions', name]
    if robot_assets is not None:
        cmd += ['--robot-assets-dir', robot_assets]
    if accept_motor_update:
        cmd += ['--accept-motor-update']
    if build_arm64:
        cmd += ['--build-arm64']
    if output:
        cmd += ["--output", output]
    run(cmd)
    if output is None:
        # The updater prints the record path; find the newest deployment.json.
        base = ROOT / "outputs/casbot02_deployment"
        records = sorted(base.glob(f"{ckpt.parent.name}_{ckpt.stem}_*/deployment.json"), key=lambda p: p.stat().st_mtime)
        if not records:
            raise RuntimeError("Could not locate the updater output directory")
        return records[-1].parent
    return output


def sync_sim2sim_launcher(output: Path, config: Path, launcher: Path = SIM2SIM_LAUNCHER) -> None:
    """Keep the local launcher on the same references/menu as this export."""
    bundle = (output / "sim2sim").resolve()
    meta = json.loads((bundle / "manifest.json").read_text())
    order = read_order(config)
    missing = [name for name in order if name not in meta["motions"]]
    if not order or len(set(order)) != len(order) or missing:
        raise ValueError(f"Exported sim2sim bundle does not match joystick menu: missing={missing}")

    def shell_path(path: Path) -> str:
        path = path.resolve()
        return shlex.quote(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))

    text = launcher.read_text()
    for option, path in (("--bundle", bundle), ("--motion-order-config", config)):
        # Replace only the dedicated option line; preserve other launcher arguments.
        pattern = rf"(?m)^(\s*{re.escape(option)}\s+).*?(\s+\\)$"
        text, count = re.subn(pattern, lambda m: m[1] + shell_path(path) + m[2], text)
        if count != 1:
            raise ValueError(f"Expected one {option} line in {launcher}")
    with tempfile.NamedTemporaryFile(mode="w", dir=launcher.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            temporary.chmod(launcher.stat().st_mode & 0o777)
            os.replace(temporary, launcher)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"[sim2sim] {launcher} -> {bundle}", flush=True)


def sync_arm64(config: Path, name: str) -> None:
    """Copy the new motion bank folder + YAML into build/install_arm64 via Docker."""
    install = ROOT / "rl_controller/build/install_arm64/config"
    if not install.exists():
        print("[arm64] install_arm64 not found; skipping sync", flush=True)
        return
    run([
        "docker", "run", "--rm",
        "-v", f"{ROOT/'rl_controller'}:/rl",
        ARM64_IMAGE,
        "bash", "-c",
        f"rm -rf /rl/build/install_arm64/config/model/supermimic_motion_bank/{name} && "
        f"cp -a /rl/config/model/supermimic_motion_bank/{name} /rl/build/install_arm64/config/model/supermimic_motion_bank/ && "
        "cp -a /rl/config/supermimic_walk1_subject2_proprio.yaml /rl/build/install_arm64/config/",
    ])
    print(f"[arm64] synced {name} + YAML into build/install_arm64", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, help="CSV or SDK .data to convert. Mutually exclusive with --npz.")
    p.add_argument("--npz", type=Path, help="Already-converted NPZ to deploy. Mutually exclusive with --csv.")
    p.add_argument("--name", help="Clip name (default: CSV/NPZ stem)")
    p.add_argument("--robot", choices=("g1", "casbot02"), default="casbot02")
    p.add_argument("--input-fps", type=float, default=30.0)
    p.add_argument("--output-fps", type=float, default=50.0)
    p.add_argument("--npz-store", type=Path, default=ROOT/"outputs/casbot02_deployed_npz", help="Where to keep the converted NPZ")
    p.add_argument("--staging", type=Path, help="Default: NPZ directory of the active export")
    p.add_argument("--ckpt", type=Path, help="Default: cpu7 / ckpt_0096000.pt")
    p.add_argument("--robot-assets-dir", type=Path, help="Verified training assets; default uses the cpu7 snapshot for cpu7/96000")
    p.add_argument("--accept-motor-update", action="store_true", help="Explicitly accept a motor change with --robot-assets-dir")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--after", help="Insert the clip right after this motion in joystick_motion_order")
    p.add_argument("--before", help="Insert the clip right before this motion")
    p.add_argument("--replace", action="store_true", help="If the clip is already in the order, move it to the new position")
    p.add_argument("--output", type=Path, help="Deployment output directory (default: timestamped under outputs/)")
    p.add_argument("--no-arm64", action="store_true", help="Skip the ARM64 package build")
    args = p.parse_args()

    if bool(args.csv) == bool(args.npz):
        p.error("Provide exactly one of --csv or --npz")
    if args.csv and not args.csv.exists():
        p.error(f"CSV not found: {args.csv}")
    if args.npz and not args.npz.exists():
        p.error(f"NPZ not found: {args.npz}")

    if args.ckpt is None:
        args.ckpt = DEFAULT_CKPT
    if args.robot_assets_dir is None and args.ckpt.resolve() == DEFAULT_CKPT.resolve():
        args.robot_assets_dir = DEFAULT_ROBOT_ASSETS

    record = None
    if args.staging is None or args.robot_assets_dir is None:
        try:
            record = active_deployment(args.config)
        except ValueError:
            if args.staging is None:
                raise
    if args.staging is None:
        args.staging = Path(record['npz_dir'])
    if args.robot_assets_dir is None and record is not None:
        if args.ckpt.resolve() != Path(record['checkpoint']).resolve():
            p.error('Changing checkpoint requires an explicit --robot-assets-dir snapshot')
        if record.get('robot_assets_dir'):
            args.robot_assets_dir = Path(record['robot_assets_dir'])
    if args.accept_motor_update and args.robot_assets_dir is None:
        p.error('--accept-motor-update requires --robot-assets-dir')

    source = args.csv or args.npz
    name = args.name or source.stem
    if Path(name).name != name or name in (".", ".."):
        p.error("Clip name must be a plain filename without slashes")

    # Check menu anchors before converting or touching the active YAML.
    current = read_order(args.config)
    order = build_order(name, current, args.after, args.before, args.replace)

    # 1. Obtain the NPZ.
    npz_path = args.npz_store / f"{name}.npz"
    if args.csv:
        convert_csv(args.csv.resolve(), npz_path, args.robot, args.input_fps, args.output_fps,
                    robot_assets=args.robot_assets_dir)
    else:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        if args.npz.resolve() != npz_path.resolve():
            shutil.copy2(args.npz, npz_path)
    print(f"[npz] {npz_path} ({npz_path.stat().st_size} bytes)", flush=True)

    # 2. Link into staging.
    link_into_staging(npz_path, args.staging, name)

    # 3. The updater will add the motion entry only after all checks pass.
    if name not in order:
        order = order + [name]
    print(f"[order] {' '.join(order)}", flush=True)

    # 4. Run the full deployment updater.
    out_dir = run_updater(args.ckpt.resolve(), args.staging.resolve(), args.config, order, args.output,
                          name=name, robot_assets=args.robot_assets_dir,
                          build_arm64=not args.no_arm64, accept_motor_update=args.accept_motor_update)

    # The exporter writes a new timestamped bundle; update the launcher as well.
    sync_sim2sim_launcher(out_dir, args.config)

    print(f"\nDone. Deployment output: {out_dir}")
    print(f"Clip '{name}' is in joystick_motion_order at position {order.index(name)+1}/{len(order)}.")
    print("Robot not uploaded. To upload: cd rl_controller && ./scripts/install_remote.sh")


if __name__ == "__main__":
    main()

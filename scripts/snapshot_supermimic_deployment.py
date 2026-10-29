"""Copy the SuperMimic controller sources from the local SDK checkout for GitHub.

The private rl_controller Git repository stays independent. This explicit file
allowlist excludes its history, credentials, SDK/AMP sources, models and builds.
"""
from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rl_controller"
TARGET = ROOT / "deployment/rl_controller"


def source_paths():
    paths = [p for p in (SOURCE / "supermimic").iterdir()
             if p.name == "CMakeLists.txt" or p.suffix in (".cpp", ".hpp")]
    paths += [SOURCE / "scripts" / name for name in (
        "build_supermimic.sh", "cmake_cache.sh", "run_supermimic.sh",
        "supermimic_motion.py", "diagnose_supermimic_sensors.sh")]
    paths += [SOURCE / "config" / name for name in (
        "supermimic_walk1_subject2_proprio.yaml", "supermimic_walk4.yaml",
        "supermimic_walk4_proprio.yaml", "supermimic_udp_only.xml")]
    paths += [SOURCE / "tests" / name for name in (
        "supermimic_live_switch_check.cpp", "supermimic_live_switch_probe.cpp",
        "supermimic_motion_bank_check.cpp", "supermimic_reference_hold_check.cpp",
        "supermimic_transition_check.cpp", "supermimic_joystick_check.cpp",
        "test_supermimic_ros.py", "test_sensor_probe.py", "block_publish.c",
        "delay_policy.cpp")]
    paths += [p for p in (SOURCE / "cmake/crb_ros_msg_compat").rglob("*")
              if p.is_file() and p.suffix in (".txt", ".xml", ".msg", ".srv", ".md")]
    paths += [SOURCE / "SUPERMIMIC_DEPLOYMENT.md"]
    return sorted(set(paths))


def main():
    paths = source_paths()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked source: {path}")
    hashes = {}
    for path in paths:
        relative = path.relative_to(SOURCE)
        output = TARGET / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (TARGET / "source_snapshot.json").write_text(
        json.dumps({"source_directory": "rl_controller", "sha256": hashes},
                   ensure_ascii=False, indent=2) + "\n")
    print(f"Copied {len(paths)} controller source/config/test files to {TARGET}")


if __name__ == "__main__":
    main()

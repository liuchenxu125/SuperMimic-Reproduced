"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import (
    load_policy_from_checkpoint,
    run_benchmark,
    save_results,
    summarize,
)


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    clip_list: Path | None = None,
    manifest: Path | None = None,
    num_envs: int = 1024,
    device: str = "cuda:0",
    out: Path | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> None:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards>=1 and 0<=shard_index<num_shards")
    if clip_list is not None:
        clips = [line.strip() for line in clip_list.read_text().splitlines() if line.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
    if not files:
        raise ValueError("No matching NPZ clips")
    dataset_sha256 = hashlib.sha256(json.dumps(
        [Path(path).stem for path in files], ensure_ascii=False
    ).encode()).hexdigest()
    total_clips = len(files)
    files = files[shard_index::num_shards]
    if not files:
        raise ValueError("Empty shard; use no more shards than clips")
    # Bind reports to exact checkpoint bytes, not just an iteration number.
    with ckpt.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        # Hash/load the same open file, even if training atomically replaces latest.
        policy, cfg = load_policy_from_checkpoint(stream, device)
    results = run_benchmark(
        policy,
        files,
        device=device,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        anchor_obs=cfg["env"].get("anchor_obs", False),
        anchor_ori_obs=cfg["env"].get("anchor_ori_obs", False),
        actuator_delay=cfg["env"].get("actuator_delay", False),
        actuator_delay_mode=cfg["env"].get("actuator_delay_mode", "episode"),
        collision_model=cfg["env"].get("collision_model", "mesh"),
        robot=cfg.get("robot", "g1"),
    )
    s = summarize(results)
    print(
        f"\n{s['clips']} clips | failed {s['failed']} | full-clip mean {s['mean_norm_return']:.4f} | frame-weighted {s['frame_weighted_norm_return']:.4f} | survived-prefix mean {s['mean_norm_return_survived']:.4f}"
    )
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(
            f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}"
        )
    suffix = f"_shard_{shard_index:03d}_of_{num_shards:03d}" if num_shards > 1 else ""
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    evaluation = {
        "robot": cfg.get("robot", "g1"), "future_frames": cfg["env"]["future_frames"],
        **{key: cfg["env"].get(key, default) for key, default in (
            ("anchor_obs", False), ("anchor_ori_obs", False), ("actuator_delay", False),
            ("actuator_delay_mode", "episode"), ("collision_model", "mesh"))},
    }
    save_results(results, out,
                 meta={**cfg.get("_checkpoint", {}), "sha256": checkpoint_sha256},
                 evaluation=evaluation,
                 shard={"index": shard_index, "count": num_shards, "total_clips": total_clips,
                        "dataset_sha256": dataset_sha256})
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)

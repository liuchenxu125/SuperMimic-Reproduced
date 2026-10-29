"""Merge eval_benchmark shards, preserving full-clip scores and checking duplicates.

Upstream legacy reports need explicit --legacy-survived to convert their
survived-prefix scores. Mixing checkpoint/protocol metadata is rejected.
"""

from pathlib import Path

import tyro

from supermimic.eval.shards import merge_shards, write_merged


def main(shards: list[Path], out: Path, legacy_survived: bool = False) -> None:
    if out.resolve() in {path.resolve() for path in shards}:
        raise ValueError("Output must not overwrite an input shard")
    report = merge_shards(shards, legacy_survived=legacy_survived)
    write_merged(report, out)
    summary = report["summary"]
    print(
        f"{summary['clips']} clips | failed {summary['failed']} | "
        f"full-clip mean {summary['mean_norm_return']:.4f} | "
        f"frame-weighted {summary['frame_weighted_norm_return']:.4f}"
    )
    if not report["merge"]["indexed_completeness_checked"]:
        print("No shard index metadata: duplicate checks passed; dataset completeness unverified")
    if not report["merge"]["checkpoint_bytes_verified"]:
        print("No checkpoint SHA256: matching metadata alone cannot prove identical weights")
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)

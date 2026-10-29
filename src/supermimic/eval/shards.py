"""Merge benchmark shards without mixing checkpoints or score conventions."""

from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from supermimic.eval.benchmark import ClipResult, summarize


def merge_shards(paths: list[Path], *, legacy_survived: bool = False) -> dict:
    if not paths:
        raise ValueError("Supply at least one shard")
    results, names, shard_info, sources = [], set(), [], []
    checkpoint = evaluation = None
    for path in paths:
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            report = json.load(stream)
        current = report.get("normalization") == "full_clip" and report.get("schema_version") == 2
        legacy = report.get("normalization") is None and report.get("schema_version") is None
        if not current and not (legacy_survived and legacy):
            raise ValueError(
                f"{path}: require schema_version=2, normalization=full_clip; "
                "use --legacy-survived only for known upstream survived-step reports"
            )
        meta, protocol = report.get("checkpoint", {}), report.get("evaluation", {})
        if not isinstance(meta, dict) or not isinstance(protocol, dict):
            raise TypeError(f"{path}: checkpoint/evaluation metadata must be objects")
        if checkpoint is None:
            checkpoint, evaluation = meta, protocol
        elif checkpoint != meta or evaluation != protocol:
            raise ValueError(f"{path}: checkpoint/evaluation metadata differs between shards")
        if not report.get("clips"):
            raise ValueError(f"{path}: empty shard")
        for item in report["clips"]:
            row = dict(item)
            name = row["name"]
            if name in names:
                raise ValueError(f"Duplicate clip across shards: {name}")
            frames, steps, score = row["frames"], row["steps"], row["norm_return"]
            if (
                not isinstance(frames, int)
                or not isinstance(steps, int)
                or not isinstance(row["failed"], bool)
                or not 1 <= steps <= frames
                or (not row["failed"] and steps != frames)
                or not math.isfinite(score)
            ):
                raise ValueError(f"{path}: invalid clip result: {name}")
            if row["fail_step"] != (steps if row["failed"] else -1):
                raise ValueError(f"{path}: inconsistent failure step: {name}")
            if legacy:
                row["norm_return_survived"] = score
                row["norm_return"] = score * steps / frames
            elif row.get("norm_return_survived") is not None:
                prefix = row["norm_return_survived"]
                if not math.isfinite(prefix) or not math.isclose(
                    prefix * steps, score * frames, rel_tol=1e-6, abs_tol=1e-8
                ):
                    raise ValueError(f"{path}: inconsistent score normalization: {name}")
            if row.get("terms") and not all(math.isfinite(v) for v in row["terms"].values()):
                raise ValueError(f"{path}: nonfinite reward terms: {name}")
            results.append(ClipResult(**row))
            names.add(name)
        shard_info.append(report.get("shard"))
        sources.append(
            {"path": str(path), "clips": len(report["clips"]), "converted_survived_prefix": legacy}
        )
    if any(info is not None for info in shard_info):
        if any(info is None for info in shard_info):
            raise ValueError("Cannot mix indexed shards and reports without shard metadata")
        first = shard_info[0]
        if any(
            {k: v for k, v in info.items() if k != "index"}
            != {k: v for k, v in first.items() if k != "index"}
            for info in shard_info
        ):
            raise ValueError("Shard dataset/count metadata differs")
        indices = [info["index"] for info in shard_info]
        if (
            len(indices) != first["count"]
            or set(indices) != set(range(first["count"]))
            or len(results) != first["total_clips"]
        ):
            raise ValueError("Missing/duplicate shards or incorrect total clip count")
        digest = hashlib_names(sorted(names))
        if digest != first["dataset_sha256"]:
            raise ValueError("Merged clip names differ from the declared dataset")
    term_keys = [set(r.terms) if r.terms is not None else None for r in results]
    if any(keys != term_keys[0] for keys in term_keys):
        raise ValueError("Reward term sets differ between clips")
    results.sort(key=lambda r: r.name)
    return {
        "schema_version": 2,
        "normalization": "full_clip",
        "checkpoint": checkpoint,
        "evaluation": evaluation,
        "summary": summarize(results),
        "clips": [asdict(r) for r in results],
        "merge": {
            "sources": sources,
            "indexed_completeness_checked": bool(shard_info[0]),
            "checkpoint_bytes_verified": bool(checkpoint.get("sha256")),
        },
    }


def hashlib_names(names: list[str]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(names, ensure_ascii=False).encode()).hexdigest()


def write_merged(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=1, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

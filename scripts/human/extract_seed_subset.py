#!/usr/bin/env python3
"""Extract only BVHs corresponding to existing robot NPZs from the official SOMA tar."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def extract(archive, output_dir, names):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining, found = set(names), []
    # Streaming tar: never materialise the complete 45GB archive or extract paths
    # supplied by it blindly. Only ordinary BVH files can be copied.
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.suffix.lower() != ".bvh" or path.stem not in remaining or not member.isfile():
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            parts = path.parts[path.parts.index("bvh") + 1 :] if "bvh" in path.parts else path.parts
            target = output_dir.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with source.extractfile(member) as inp, temporary.open("wb") as out:
                shutil.copyfileobj(inp, out)
            temporary.replace(target)
            remaining.remove(path.stem)
            found.append(str(target))
            print(f"{len(found)}/{len(names)} {path.stem}", flush=True)
            if not remaining:
                break
    result = {
        "archive": str(Path(archive).resolve()),
        "extracted": found,
        "missing": sorted(remaining),
    }
    (output_dir / "extraction_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=20, help="0 = all selected existing robot clips"
    )
    parser.add_argument("--names", nargs="+", help="Explicit SEED motion names, without extension")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(args.robot_dir, names=args.names, manifest=args.manifest)
    names = sorted({p.stem for p in map(Path, files) if re.search(r"__A\d+", p.stem)})
    if args.limit:
        names = names[: args.limit]
    if not names:
        parser.error("No matching existing SEED robot clips")
    result = extract(args.archive, args.output_dir, names)
    if result["missing"]:
        raise SystemExit("Missing archive motions; see extraction_report.json")


if __name__ == "__main__":
    main()

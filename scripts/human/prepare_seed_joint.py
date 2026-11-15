#!/usr/bin/env python3
"""Stream official SOMA tar -> strictly paired sidecars, without unpacking all BVH.

Safe to resume into the same output when source identity is unchanged. Each BVH
is temporarily extracted, converted, checked, then removed. Original robot NPZs
and manifests are read-only. --limit 0 selects every retained SEED motion.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def convert_pair(job):
    """Spawn-safe CPU worker: only one temporary BVH per queued conversion."""
    from supermimic.human.data import Pair, load_pair
    from supermimic.human.soma import convert_human_clip
    name, robot_path, target, temporary, source_name, meta, min_frames = job
    try:
        if not target.exists():
            convert_human_clip(str(temporary),str(target))
        pair=Pair(name,robot_path,target)
        robot,_=load_pair(pair,meta)
        if len(robot)<min_frames:
            raise ValueError(f"too_short: {len(robot)}")
        row=dict(name=name,robot_npz=str(robot_path),human_npz=str(target),
                 robot_start_frame=0,human_start_frame=0,length=len(robot))
        return row,dict(name=name,status='ready',frames=len(robot),human_source=source_name)
    except (ValueError,KeyError) as exc:
        return None,dict(name=name,status='rejected',reason=str(exc))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--robot-dir",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=1000)
    parser.add_argument("--min-frames",type=int,default=52)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    if args.limit<0 or args.min_frames<52 or args.workers<1:
        parser.error("limit >= 0, min-frames >= 52 and workers >= 1 required")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[key]="1"
    os.environ.setdefault("MPLCONFIGDIR","/tmp/supermimic-human-mpl")
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.human.teacher import digest
    from supermimic.human.joint_training import robot_contract
    meta,_=robot_contract()
    output=args.output_dir.resolve()
    if output==args.robot_dir.resolve() or output in args.robot_dir.resolve().parents:
        parser.error("Use an output directory separate from robot source data")
    output.mkdir(parents=True,exist_ok=True)
    stat=args.archive.stat()
    identity={"archive":str(args.archive.resolve()),"archive_size":stat.st_size,
        "archive_mtime_ns":stat.st_mtime_ns,"robot_dir":str(args.robot_dir.resolve()),
        "manifest_sha256":digest(args.manifest),"min_frames":args.min_frames,
        "joint_names":meta["joint_names"],"conversion":"SOMA-uniform-120to50-v10"}
    identfile=output/"source_identity.json"
    if identfile.exists():
        if json.loads(identfile.read_text())!=identity:
            raise ValueError("Preparation source identity changed; use a new output directory")
    elif any(output.iterdir()):
        raise FileExistsError("Unknown existing output; use a new directory")
    else:
        identfile.write_text(json.dumps(identity,indent=2))
    files=discover_npz(args.robot_dir,manifest=args.manifest)
    robot_files={p.stem:p.resolve() for p in files if re.search(r"__A\d+",p.stem)}
    pairsfile=output/"pairs.csv"
    rows=[]
    if pairsfile.exists():
        with pairsfile.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
    found={r["name"] for r in rows}
    if len(found)!=len(rows) or not found.issubset(robot_files):
        raise ValueError("Prepared pairs contain duplicates/unknown source names")
    reportfile=output/"preparation_report.json"
    audit=json.loads(reportfile.read_text()).get("audit",[]) if reportfile.exists() else []
    rejected={r["name"] for r in audit if r["status"]=="rejected"}
    fields=["name","robot_npz","human_npz","robot_start_frame","human_start_frame","length"]
    def save():
        temp=pairsfile.with_suffix('.csv.tmp')
        with temp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temp.replace(pairsfile)
        temp=reportfile.with_suffix('.json.tmp')
        temp.write_text(json.dumps({"ready":len(rows),"rejected":len(rejected),
            "selected_robot_clips":len(robot_files),"requested_limit":args.limit,
            "pending":len(set(robot_files)-found-rejected),"audit":audit,
            "robot_data_modified":False,"source_bvh_retained":False},ensure_ascii=False,indent=2))
        temp.replace(reportfile)
    if args.limit and len(rows)>=args.limit:
        save();print(json.dumps({"ready":len(rows),"manifest":str(pairsfile),"already_prepared":True}));return
    queue_dir=output/".bvh_queue"
    queue_dir.mkdir(exist_ok=True)
    visited=0
    archive_exhausted=False
    queued=set()
    pending=set()
    executor=None
    if args.workers>1:
        from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
        from multiprocessing import get_context
        executor=ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'))
    def record(result):
        nonlocal visited
        row,entry=result
        if row is not None:
            rows.append(row);found.add(entry['name'])
        else:
            rejected.add(entry['name'])
        audit.append(entry);visited+=1
        if visited%100==0:
            save();print(json.dumps({"ready":len(rows),"rejected":len(rejected)},ensure_ascii=False),flush=True)
    def drain_one():
        done,_=wait(pending,return_when=FIRST_COMPLETED)
        for future in done:
            pending.remove(future);record(future.result())
    try:
        with tarfile.open(args.archive,'r|gz') as archive:
            for member in archive:
                path=PurePosixPath(member.name)
                if path.suffix.lower()!='.bvh' or not member.isfile() or path.stem not in robot_files:
                    continue
                name=path.stem
                if name in found or name in rejected or name in queued:
                    continue
                while pending and (len(pending)>=args.workers*2 or
                                   (args.limit and len(rows)+len(pending)>=args.limit)):
                    drain_one()
                if args.limit and len(rows)>=args.limit:
                    break
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError(f"Unsafe archive member: {path}")
                target=output/'human50'/f'{name}.npz'
                temp_bvh=None
                if not target.exists():
                    fd,tempname=tempfile.mkstemp(suffix='.bvh',dir=queue_dir)
                    temp_bvh=Path(tempname)
                    with archive.extractfile(member) as inp,os.fdopen(fd,'wb') as dest:
                        shutil.copyfileobj(inp,dest)
                job=(name,robot_files[name],target,temp_bvh,str(path),meta,args.min_frames)
                queued.add(name)
                if executor is None:
                    record(convert_pair(job))
                else:
                    pending.add(executor.submit(convert_pair,job))
                if args.limit and len(rows)>=args.limit:
                    break
            else:
                archive_exhausted=True
        while pending:
            drain_one()
        if archive_exhausted:
            for name in sorted(set(robot_files)-found-rejected):
                rejected.add(name)
                audit.append(dict(name=name,status='rejected',reason='missing_human_in_archive'))
    except KeyboardInterrupt:
        print("Interrupted; preserving verified pairs for continuation",flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True,cancel_futures=True)
        for p in queue_dir.glob('*.bvh'):
            p.unlink(missing_ok=True)
        queue_dir.rmdir()
        save()
    print(json.dumps({"ready":len(rows),"rejected":len(rejected),"manifest":str(pairsfile)},ensure_ascii=False),flush=True)
    if not rows:
        raise SystemExit("No strict human/robot pairs; see preparation_report.json")


if __name__=='__main__':
    main()

"""Repair resampling collisions into a NEW NPZ; never replace a live bundle.

Uses local trajectory optimization with joint bounds, displacement/smoothness
costs, preserved coincident source samples, and sampled exterior-mesh clearance.
The collision XML must have the same kinematics/limits as the conversion robot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from supermimic.data.collision_repair import ExteriorMeshChecker, repair_joints
from supermimic.data.motion_convert import (
    ResampledMotion, _so3_derivative, forward_kinematics, write_motion_npz,
)
from supermimic.robots import get_robot


def motion_metrics(q, fps):
    velocity = np.gradient(q.astype(float), 1 / fps, axis=0)
    acceleration = np.gradient(velocity, 1 / fps, axis=0)
    return dict(max_abs_velocity_rad_s=float(np.abs(velocity).max()),
                p99_abs_velocity_rad_s=float(np.percentile(np.abs(velocity), 99)),
                max_abs_acceleration_rad_s2=float(np.abs(acceleration).max()),
                p99_abs_acceleration_rad_s2=float(np.percentile(np.abs(acceleration), 99)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New file, never the input path")
    p.add_argument("--collision-xml", type=Path, required=True, help="XML with exterior mesh geoms in group 1")
    p.add_argument("--margin-mm", type=float, default=2.)
    p.add_argument("--max-change-rad", type=float, default=.15)
    p.add_argument("--substeps", type=int, default=4, help="4 checks 50 Hz input at 200 Hz")
    p.add_argument("--audit-substeps", type=int, default=None, help="Optional finer audit factor; default is --substeps (200 Hz for 50 Hz input)")
    p.add_argument("--pad-frames", type=int, default=3)
    p.add_argument("--clamp-joints", action="store_true", help="Explicitly accept repairing pre-existing small limit violations")
    a = p.parse_args()
    if a.audit_substeps is None:
        a.audit_substeps = a.substeps
    if a.audit_substeps < a.substeps:
        p.error("--audit-substeps must be >= --substeps")
    source = a.input.expanduser().resolve(strict=True)
    target = a.output.expanduser().resolve()
    if target == source or target.exists():
        p.error("--output must be a new file; input and live deployment data are never overwritten")
    report_path = target.with_suffix(".report.json")
    if report_path.exists():
        p.error("Output report already exists; choose a new --output")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    fps = float(arrays["fps"].item())
    input_fps = float(arrays["input_fps"].item())
    profile = get_robot(str(arrays["robot"].item()))
    model = profile.model(collision_model="primitive" if profile.name == "casbot02" else "mesh")
    collision_model = mujoco.MjModel.from_xml_path(str(a.collision_xml.expanduser().resolve(strict=True)))
    names = arrays["joint_names"].tolist()
    if names != [model.joint(i).name for i in range(1, model.njnt)]:
        raise ValueError("NPZ/model joint order mismatch")
    if names != [collision_model.joint(i).name.removeprefix("robot/") for i in range(1, collision_model.njnt)]:
        raise ValueError("NPZ/collision XML joint order mismatch")
    if arrays["body_names"].tolist() != [model.body(i).name for i in range(1, model.nbody)]:
        raise ValueError("NPZ/model body order mismatch")
    np.testing.assert_allclose(model.jnt_range[1:], collision_model.jnt_range[1:], atol=1e-8, rtol=0)
    # Ensure these two models really have matching link frames before optimizing.
    cd, md = mujoco.MjData(collision_model), mujoco.MjData(model)
    body_ids = [collision_model.body("robot/" + n).id if "robot/" + n in
                [collision_model.body(i).name for i in range(collision_model.nbody)]
                else collision_model.body(n).id for n in arrays["body_names"].tolist()]
    root = arrays["body_names"].tolist().index(profile.root_body)
    for f in np.unique(np.linspace(0, len(arrays["joint_pos"]) - 1, 9).astype(int)):
        qp = np.r_[arrays["body_pos_w"][f, root], arrays["body_quat_w"][f, root], arrays["joint_pos"][f]]
        cd.qpos[:] = qp
        md.qpos[:] = qp
        mujoco.mj_kinematics(collision_model, cd)
        mujoco.mj_kinematics(model, md)
        np.testing.assert_allclose(cd.xpos[body_ids], md.xpos[1:], atol=2e-6, rtol=0)
        np.testing.assert_allclose(md.xpos[1:], arrays["body_pos_w"][f], atol=2e-5, rtol=0)
    checker = ExteriorMeshChecker(collision_model)
    before = arrays["joint_pos"].copy()
    began = time.perf_counter()
    q, result = repair_joints(checker, before, fps, input_fps, margin=a.margin_mm / 1000,
                              max_change=a.max_change_rad, substeps=a.substeps,
                              pad_frames=a.pad_frames, clamp_limits=a.clamp_joints)
    refinements = []
    for audit_round in range(3):
        hits = [] if audit_round == 0 and a.audit_substeps == a.substeps else checker.scan(q.astype(float), a.audit_substeps)
        print(f"audit {audit_round}: {len(set(f for f, _, _ in hits))} collision poses at {fps * a.audit_substeps:g} Hz", flush=True)
        if not hits:
            break
        if audit_round == 2:
            raise RuntimeError("Dense audit still detects collisions; output not written")
        windows = []
        for f in sorted({int(f // a.audit_substeps) for f, _, _ in hits}):
            start, end = max(0, f - a.pad_frames), min(len(q) - 1, f + a.pad_frames + 1)
            if windows and start <= windows[-1][1] + 1:
                windows[-1][1] = max(end, windows[-1][1])
            else:
                windows.append([start, end])
        for start, end in windows:
            local, detail = repair_joints(checker, q[start:end + 1], fps, input_fps,
                                          margin=a.margin_mm / 1000, max_change=a.max_change_rad,
                                          substeps=a.audit_substeps, pad_frames=a.pad_frames,
                                          frame_offset=start)
            q[start:end + 1] = local
            refinements.append(dict(window=[start, end], detail=detail))
        if np.abs(q.astype(float) - before.astype(float)).max() > a.max_change_rad + 1e-6:
            raise RuntimeError("Combined refinements exceed the correction budget; output not written")
    result.update(audit_substeps=a.audit_substeps, audit_hz=fps * a.audit_substeps,
                  final_audit_collision_frames=0, audit_refinements=refinements,
                  max_correction_rad=float(np.abs(q.astype(float) - before.astype(float)).max()),
                  modified_frames=int(np.any(np.abs(q.astype(float) - before.astype(float)) > 1e-7, axis=1).sum()))
    positions = arrays["body_pos_w"][:, root].astype(float)
    quaternions = arrays["body_quat_w"][:, root].astype(float)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    motion = ResampledMotion(fps=fps, root_pos=positions, root_quat=quaternions, joint_pos=q.astype(float),
                             root_lin_vel_w=np.gradient(positions, 1 / fps, axis=0),
                             root_ang_vel_w=_so3_derivative(quaternions, 1 / fps),
                             joint_vel=np.gradient(q.astype(float), 1 / fps, axis=0))
    fk = forward_kinematics(model, motion)
    arrays.update(joint_pos=q, joint_vel=motion.joint_vel.astype(np.float32),
                  body_pos_w=fk.pos_w.astype(np.float32), body_quat_w=fk.quat_w.astype(np.float32),
                  body_lin_vel_w=fk.lin_vel_w.astype(np.float32), body_ang_vel_w=fk.ang_vel_w.astype(np.float32),
                  collision_repair_source=np.array(str(source)), collision_repair_method=np.array("local_slsqp_dense_hull_clearance_v1"))
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if len(arrays[key]) != len(q) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid repaired field: {key}")
    if ((q < collision_model.jnt_range[1:, 0]) | (q > collision_model.jnt_range[1:, 1])).any():
        raise ValueError("Saved float32 joints exceed limits")
    np.testing.assert_allclose(arrays["joint_vel"], np.gradient(q.astype(float), 1 / fps, axis=0), atol=2e-5, rtol=2e-6)
    result.update(input=str(source), input_sha256=source_hash, output=str(target),
                  collision_xml=str(a.collision_xml.resolve()), frames=len(q), fps=fps,
                  dense_validation_hz=fps * a.substeps, checked_mesh_pairs=len(checker.pairs),
                  skip_graph_distance=checker.skip_graph_distance, target_margin_mm=a.margin_mm,
                  before=motion_metrics(before, fps), after=motion_metrics(q, fps),
                  correction=motion_metrics(q.astype(float) - before.astype(float), fps),
                  elapsed_seconds=time.perf_counter() - began,
                  limitations="Discrete exterior convex-hull checks; joint interfaces/XML excludes omitted. No proof between validation samples, dynamic feasibility, ground-contact preservation, or policy tracking. Original root trajectory and clip duration preserved.")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash or target.exists() or report_path.exists():
        raise RuntimeError("Input/output changed during repair; nothing written")
    write_motion_npz(target, arrays)
    result["output_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

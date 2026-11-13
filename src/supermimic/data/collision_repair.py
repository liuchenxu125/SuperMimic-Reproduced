"""Conservative exterior-mesh checks and local, collision-constrained resampling.

Meshes are checked as convex hulls. Joint interfaces within two kinematic edges
are excluded explicitly; this is a discrete pose check, not a continuous-motion
certificate. No simulation state is integrated and no policy is evaluated.
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np
from scipy.optimize import minimize


def dense_poses(q, substeps):
    if substeps < 1:
        raise ValueError("substeps must be positive")
    t = np.arange((len(q) - 1) * substeps + 1) / substeps
    lo = np.minimum(t.astype(int), len(q) - 2)
    blend = (t - lo)[:, None]
    return (1 - blend) * q[lo] + blend * q[lo + 1]


class ExteriorMeshChecker:
    def __init__(self, model, *, skip_graph_distance=2):
        self.model, self.data = model, mujoco.MjData(model)
        self.nj = model.nq - 7
        if model.nv != self.nj + 6 or model.njnt != self.nj + 1 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected one free root followed by scalar joints")
        if skip_graph_distance < 0:
            raise ValueError("skip_graph_distance must be nonnegative")
        self.geoms = np.array([g for g in range(model.ngeom)
                               if model.geom_group[g] == 1 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
        if len(self.geoms) < 2:
            raise ValueError("Need at least two exterior meshes in geom group 1")
        self.skip_graph_distance = skip_graph_distance
        excluded = set(map(int, model.exclude_signature))
        pairs, local = [], []
        for i, j in itertools.combinations(range(len(self.geoms)), 2):
            a, b = map(int, model.geom_bodyid[self.geoms[[i, j]]])
            wa, wb = map(int, model.body_weldid[[a, b]])
            ap, bp = self.ancestors(wa), self.ancestors(wb)
            distance = min(ap.index(v) + bp.index(v) for v in set(ap) & set(bp))
            if wa == wb or distance <= skip_graph_distance or (min(a, b) << 16) + max(a, b) in excluded:
                continue
            pairs.append((int(self.geoms[i]), int(self.geoms[j])))
            local.append((i, j))
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.local_pairs = np.asarray(local, dtype=int).reshape(-1, 2)
        corners = []
        for g in self.geoms:
            k = int(model.geom_dataid[g])
            v = model.mesh_vert[model.mesh_vertadr[k]:model.mesh_vertadr[k] + model.mesh_vertnum[k]]
            corners.append(list(itertools.product(*zip(v.min(0), v.max(0)))))
        self.corners = np.asarray(corners)

    def ancestors(self, body):
        path = [body]
        while body:
            body = int(self.model.body_parentid[body])
            path.append(body)
        return path

    def pose(self, q, *, jacobian=False):
        self.data.qpos[:3] = 0
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:] = q
        mujoco.mj_kinematics(self.model, self.data)
        if jacobian:
            mujoco.mj_comPos(self.model, self.data)

    def near(self, q, threshold=0.0):
        self.pose(q)
        if not len(self.pairs):
            return []
        d, m = self.data, self.model
        points = np.einsum("gvc,gdc->gvd", self.corners, d.geom_xmat[self.geoms].reshape(-1, 3, 3))
        points += d.geom_xpos[self.geoms, None, :]
        lo, hi = points.min(1), points.max(1)
        a, b = self.local_pairs.T
        mask = np.all(hi[a] + threshold >= lo[b], axis=1) & np.all(hi[b] + threshold >= lo[a], axis=1)
        found = []
        for k in np.flatnonzero(mask):
            distance = mujoco.mj_geomDistance(m, d, *self.pairs[k], max(threshold, .001), None)
            if distance < threshold - 1e-6:
                found.append((int(k), float(distance)))
        return found

    def scan(self, q, substeps=1):
        return [(f, k, distance) for f, row in enumerate(dense_poses(q, substeps))
                for k, distance in self.near(row)]

    def distances(self, q, selected, margin):
        self.pose(q, jacobian=True)
        m, d = self.model, self.data
        values = np.empty(len(selected))
        jac = np.zeros((len(selected), self.nj))
        for j, k in enumerate(selected):
            g1, g2 = self.pairs[k]
            points = np.empty(6)
            distance = mujoco.mj_geomDistance(m, d, g1, g2, .05, points)
            values[j] = distance - margin
            if distance >= .049999:
                continue  # Distance was truncated at dismax: this constraint is inactive.
            if abs(distance) < 1e-9:
                # The signed segment has no stable normal at exact contact.
                for axis in range(self.nj):
                    perturbed = q.copy()
                    perturbed[axis] += 1e-5
                    self.pose(perturbed)
                    jac[j, axis] = (mujoco.mj_geomDistance(m, d, g1, g2, .05, None) - distance) / 1e-5
                self.pose(q, jacobian=True)
                continue
            ja, jb = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, ja, None, points[:3], int(m.geom_bodyid[g1]))
            mujoco.mj_jac(m, d, jb, None, points[3:], int(m.geom_bodyid[g2]))
            jac[j] = (((points[3:] - points[:3]) / distance) @ (jb - ja))[6:]
        return values, jac


def repair_joints(checker, original, fps, input_fps, *, margin=.002, max_change=.15,
                  substeps=4, pad_frames=3, clamp_limits=False, max_rounds=4,
                  frame_offset=0, progress=print):
    original = np.asarray(original, dtype=float)
    if original.ndim != 2 or original.shape[1] != checker.nj or len(original) < 3 or not np.isfinite(original).all():
        raise ValueError("Invalid joint trajectory")
    if not (np.isfinite([fps, input_fps, margin, max_change]).all() and fps > 0 and input_fps > 0
            and 0 < margin < .02 and max_change > 0 and substeps >= 1 and pad_frames >= 2
            and max_rounds >= 1 and frame_offset >= 0):
        raise ValueError("Invalid FPS/clearance/correction/sampling settings")
    limits = checker.model.jnt_range[1:]
    if not checker.model.jnt_limited[1:].all():
        raise ValueError("All scalar joints must have explicit limits")
    # Stay inside exact double limits after saving the float32 NPZ.
    low, high = limits[:, 0].astype(np.float32), limits[:, 1].astype(np.float32)
    low = np.where(low < limits[:, 0], np.nextafter(low, np.float32(np.inf)), low).astype(float)
    high = np.where(high > limits[:, 1], np.nextafter(high, np.float32(-np.inf)), high).astype(float)
    clipped = np.clip(original, low, high)
    excess = np.abs(clipped - original)
    if excess.max() > 1e-6 and not clamp_limits:
        raise ValueError("Input exceeds joint limits; inspect it before using --clamp-joints")
    base = clipped if clamp_limits else original.copy()
    q = base.copy()
    lower, upper = np.maximum(low, original - max_change), np.minimum(high, original + max_change)
    if np.any(lower > upper):
        raise ValueError("Joint-limit correction exceeds max_change")
    initial = checker.scan(original, substeps)
    logs = []
    weights = np.ones(checker.nj)
    for j in range(checker.nj):
        name = checker.model.joint(j + 1).name
        if any(part in name for part in ("leg_", "hip_", "knee_", "ankle_")):
            weights[j] = 4  # Prefer arm adjustments over perturbing leg references.
    for round_no in range(max_rounds + 1):
        bad = checker.scan(q, substeps)
        progress(f"round {round_no}: {len(set(f for f, _, _ in bad))} dense collision poses", flush=True)
        if not bad:
            saved = q.astype(np.float32)
            if checker.scan(saved.astype(float), substeps):
                raise RuntimeError("Float32 output reintroduced a collision")
            return saved, dict(initial_dense_collision_frames=len(set(f for f, _, _ in initial)),
                               final_dense_collision_frames=0, substeps=substeps,
                               clamped_joint_values=int((excess > 1e-6).sum()) if clamp_limits else 0,
                               max_limit_clamp_rad=float(excess.max()) if clamp_limits else 0.,
                               max_correction_rad=float(np.abs(saved.astype(float) - original).max()),
                               modified_frames=int(np.any(np.abs(saved.astype(float) - original) > 1e-7, axis=1).sum()),
                               windows=logs)
        if round_no == max_rounds:
            break
        windows = []
        for f in sorted({int(f // substeps) for f, _, _ in bad}):
            a, b = max(0, f - pad_frames), min(len(q) - 1, f + pad_frames + 1)
            if windows and a <= windows[-1][1] + 1:
                windows[-1][1] = max(windows[-1][1], b)
            else:
                windows.append([a, b])
        for a, b in windows:
            frames = np.array([f for f in range(a + 1, b)
                               if abs((f + frame_offset) / fps * input_fps
                                      - round((f + frame_offset) / fps * input_fps)) > 1e-7], dtype=int)
            if not len(frames):
                raise RuntimeError("Collision occurs at a fixed source sample; automatic interpolation repair is insufficient")
            indices = frames - a
            segment = base[a:b + 1]
            dense = dense_poses(segment, substeps)
            t = np.arange(len(dense)) / substeps
            lo = np.minimum(t.astype(int), len(segment) - 2)
            blend = t - lo
            W = np.zeros((len(dense), len(frames)))
            for j, index in enumerate(indices):
                W[:, j] = (lo == index) * (1 - blend) + (lo + 1 == index) * blend
            active = np.any(W != 0, axis=1)
            W, dense = W[active], dense[active]
            current_dense = dense_poses(q[a:b + 1], substeps)[active]
            per_pose = [sorted({k for row in (before_row, current_row)
                                for k, _ in checker.near(row, .025)})
                        for before_row, current_row in zip(dense, current_dense)]
            selected = sorted({k for items in per_pose for k in items})
            if not selected:
                raise RuntimeError("No adjustable collision candidates")
            # Common ancestors move both surfaces rigidly and cannot change
            # their separation. Optimize only joints that can affect these pairs.
            relevant_bodies = set()
            for k in selected:
                ba, bb = map(int, checker.model.geom_bodyid[checker.pairs[k]])
                relevant_bodies.update(set(checker.ancestors(ba)) ^ set(checker.ancestors(bb)))
            joints = np.array([j for j in range(checker.nj)
                               if int(checker.model.jnt_bodyid[j + 1]) in relevant_bodies], dtype=int)
            if not len(joints):
                raise RuntimeError("No scalar joints can separate the colliding meshes")

            def delta_values(x):
                delta = np.zeros((len(frames), checker.nj))
                delta[:, joints] = x.reshape(-1, len(joints))
                return delta

            def objective(x):
                delta = np.zeros_like(segment)
                delta[indices] = delta_values(x)
                vel, acc = np.diff(delta, axis=0), np.diff(delta, n=2, axis=0)
                loss = np.sum(delta ** 2 * weights) + .5 * np.sum(vel ** 2) + 4 * np.sum(acc ** 2)
                grad = 2 * delta * weights
                grad[:-1] -= vel
                grad[1:] += vel
                grad[:-2] += 8 * acc
                grad[1:-1] -= 16 * acc
                grad[2:] += 8 * acc
                return loss, grad[indices][:, joints].reshape(-1)

            cache = {}

            def evaluate(x, jacobian=False):
                if "x" not in cache or not np.array_equal(cache["x"], x):
                    poses = dense + W @ delta_values(x)
                    values, gradients = [], []
                    for pose_index, items in enumerate(per_pose):
                        if not items:
                            continue
                        value, gradient = checker.distances(poses[pose_index], items, margin)
                        values.append(value)
                        gradients.append(np.einsum("f,pj->pfj", W[pose_index], gradient[:, joints]).reshape(-1, len(x)))
                    cache.update(x=x.copy(), values=np.concatenate(values) * 100,
                                 jac=np.concatenate(gradients) * 100)
                return cache["jac" if jacobian else "values"]

            bounds = list(zip((lower[frames] - base[frames])[:, joints].reshape(-1),
                              (upper[frames] - base[frames])[:, joints].reshape(-1)))
            result = minimize(objective, (q[frames] - base[frames])[:, joints].reshape(-1), jac=True,
                              method="SLSQP", bounds=bounds,
                              constraints={"type": "ineq", "fun": evaluate, "jac": lambda x: evaluate(x, True)},
                              options={"maxiter": 250, "ftol": 1e-9})
            residual = float(evaluate(result.x).min() / 100)
            log = dict(round=round_no, window=[a, b], iterations=int(result.nit),
                       optimizer_success=bool(result.success), message=str(result.message),
                       min_constraint_residual_m=residual, free_frames=frames.tolist())
            logs.append(log)
            if residual < -1e-4 or not np.isfinite(result.x).all():
                raise RuntimeError(f"Window repair failed: {log}")
            q[frames] = base[frames] + delta_values(result.x)
            progress(f"window {a}:{b}, residual={residual:.6g} m, iterations={result.nit}", flush=True)
    raise RuntimeError("Unresolved collisions after maximum repair rounds; output not written")

"""Optional, full-body PICO adapter for the actual upstream GMR solver.

Load only GMR's IK modules: its package __init__ eagerly imports Torch/viewers,
which are unnecessary in the isolated PICO sender environment. No upstream
source or global Mink API is modified.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .pico import JOINTS, zup_body

DEFAULT_CONFIG = Path(__file__).parent / "ik_configs/xrobot_to_casbot_casbot02.json"
GMR_REF = "703841aa354920715f4e6f84d7080016699af1ac"


def default_gmr_root():
    desktop = Path.home() / "Desktop/GMR"
    bundled = Path(__file__).resolve().parents[3] / "outputs/teleop_dependencies" / ("GMR-" + GMR_REF)
    return desktop if (desktop / "general_motion_retargeting/motion_retarget.py").is_file() else bundled


def resolve_config(config, model):
    """Resolve canonical names against the exported actor's robot/ namespace."""
    result = copy.deepcopy(config)
    if result["human_root_name"] != "Pelvis":
        raise ValueError("Full-body PICO GMR requires human_root_name=Pelvis")

    def body(name):
        for candidate in (name, "robot/" + name):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate) >= 0:
                return candidate
        raise ValueError(f"GMR body {name!r} is missing from this policy model")

    result["robot_root_name"] = body(result["robot_root_name"])
    scale_names = set(result["human_scale_table"])
    offsets = []
    for key in ("ik_match_table1", "ik_match_table2"):
        table = result[key]
        if not result["use_" + key]:
            raise ValueError("Casbot02 GMR requires both IK stages")
        seen = {}
        for name, entry in table.items():
            if len(entry) != 5 or entry[0] not in JOINTS:
                raise ValueError(f"Invalid GMR mapping {name}")
            human, wp, wr, p, q = entry
            if human in seen or not np.isfinite([wp, wr, *p, *q]).all():
                raise ValueError(f"Duplicate/nonfinite GMR mapping {human}")
            if wp < 0 or wr < 0 or wp + wr == 0 or len(p) != 3 or len(q) != 4:
                raise ValueError(f"Invalid GMR weights/offsets for {human}")
            if not np.isclose(np.linalg.norm(q), 1, atol=1e-6):
                raise ValueError(f"GMR quaternion must be normalized wxyz: {human}")
            seen[human] = (p, q)
        if set(seen) != scale_names:
            raise ValueError("Each GMR scaled human body needs exactly one active task in each stage")
        offsets.append(seen)
        result[key] = {body(name): entry for name, entry in table.items()}
    if offsets[0] != offsets[1]:
        raise ValueError("Upstream GMR applies stage1 offsets to both stages; offsets must match")
    scales = list(result["human_scale_table"].values())
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("GMR scales must be finite and positive")
    return result


def make_solver(gmr_root, model_file, config, human_height):
    """Instantiate upstream GMR under a private package, with Mink API adaptation."""
    try:
        import mink
    except ImportError as exc:
        raise RuntimeError("Install isolated GMR dependencies: bash scripts/teleop/setup_gmr.sh") from exc
    package_path = Path(gmr_root) / "general_motion_retargeting"
    if not (package_path / "motion_retarget.py").is_file():
        raise ValueError(f"GMR source missing at {gmr_root}; run scripts/teleop/setup_gmr.sh")
    alias = "_supermimic_gmr_" + uuid.uuid4().hex
    package = types.ModuleType(alias)
    package.__path__ = [str(package_path)]
    sys.modules[alias] = package

    def load(name):
        spec = importlib.util.spec_from_file_location(alias + "." + name, package_path / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    try:
        params = load("params")
        module = load("motion_retarget")
        # GMR passes limits as its sixth positional argument. Mink 1.3 puts
        # safety_break there. Bind explicitly, without monkeypatching Mink.
        def solve_ik(configuration, tasks, dt, solver, damping, limits):
            return mink.solve_ik(configuration, tasks, dt, solver,
                                 damping=damping, limits=limits, safety_break=True)

        api = dict(vars(mink))
        api["solve_ik"] = solve_ik
        module.mink = types.SimpleNamespace(**api)
        with tempfile.TemporaryDirectory(prefix="casbot02-gmr-") as temp:
            path = Path(temp) / "resolved.json"
            path.write_text(json.dumps(config))
            params.ROBOT_XML_DICT["supermimic_casbot02"] = Path(model_file)
            params.IK_CONFIG_DICT.setdefault("xrobot", {})["supermimic_casbot02"] = path
            solver = module.GeneralMotionRetargeting(
                "xrobot", "supermimic_casbot02", actual_human_height=human_height,
                solver="daqp", damping=0.5, verbose=False,
            )
        return solver
    finally:
        for name in (alias + ".motion_retarget", alias + ".params", alias):
            sys.modules.pop(name, None)


class CasbotGMRRetargeter:
    def __init__(self, bundle, meta, *, gmr_root=None, config_path=None,
                 human_height=None, max_joint_speed=6.0):
        self.gmr_root = Path(gmr_root or default_gmr_root()).resolve()
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        config = json.loads(self.config_path.read_text())
        model_file = Path(bundle) / meta["model_file"]
        self.model = mujoco.MjModel.from_xml_path(str(model_file))
        names = [self.model.joint(i).name.removeprefix("robot/") for i in range(1, self.model.njnt)]
        if names != meta["joint_names"] or self.model.nq != 34 or self.model.nv != 33:
            raise ValueError("GMR model and policy must have matching 27-joint order")
        if human_height is not None and not 1.0 <= human_height <= 2.3:
            raise ValueError("--human-height must be in metres (1.0..2.3)")
        if not np.isfinite(max_joint_speed) or not 0 < max_joint_speed <= 20:
            raise ValueError("Invalid GMR joint reference speed")
        self.config = resolve_config(config, self.model)
        self.gmr = make_solver(self.gmr_root, model_file, self.config, human_height)
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.source_sha256 = hashlib.sha256(
            (self.gmr_root / "general_motion_retargeting/motion_retarget.py").read_bytes()).hexdigest()
        self.limits = self.model.jnt_range[1:].copy()
        self.q0 = np.array([m["q0"] for m in meta["motors"]])
        self.max_step = max_joint_speed / 50.0
        self.data = mujoco.MjData(self.model)
        self.foot_ids = [self.model.geom(name).id for name in meta["foot_geom_names"]]
        self.neutral = np.r_[np.zeros(3), [1.0, 0, 0, 0], self.q0]
        self.neutral[2] = 0.002 - self._sole_height(self.neutral)
        self.previous = self.neutral.copy()
        self.ground_shift = 0.0
        self.calibrated = False
        self.metrics = {}
        self.scale = self.gmr.human_scale_table["Pelvis"]

    def _sole_height(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        lowest = []
        for geom in self.foot_ids:
            mesh = int(self.model.geom_dataid[geom])
            if mesh < 0:
                raise ValueError("GMR ground calibration expects exported foot meshes")
            start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
            vertices = self.model.mesh_vert[start:start + count]
            vertices = vertices @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]
            lowest.append(float(vertices[:, 2].min()))
        return min(lowest)

    def human_data(self, raw):
        positions, rotations = zup_body(raw)
        positions = self.alignment.apply(positions - self.origin)
        quaternions = (self.alignment * rotations).as_quat(scalar_first=True)
        return {name: [p, q] for name, p, q in zip(JOINTS, positions, quaternions)}

    def calibrate(self, raw):
        positions, rotations = zup_body(raw)
        ankle_z = positions[[JOINTS.index("Left_Ankle"), JOINTS.index("Right_Ankle")], 2].mean()
        if not 0.35 < positions[0, 2] - ankle_z < 1.5:
            raise ValueError("GMR requires a valid standing 24-joint PICO skeleton and ankle Trackers")
        root_offset = self.gmr.rot_offsets1["Pelvis"]
        forward = (rotations[0] * root_offset).apply([1.0, 0, 0])
        if np.linalg.norm(forward[:2]) < 0.5:
            raise ValueError("Calibrate GMR standing upright")
        self.alignment = Rotation.from_euler("z", -np.arctan2(forward[1], forward[0]))
        self.origin = positions[0].copy()
        self.origin[2] = 0  # preserve height and vertical motion, recenter horizontal origin only
        human = self.human_data(raw)
        q = self.neutral.copy()
        q[:3] = human["Pelvis"][0] * self.scale
        q[3:7] = (self.alignment * rotations[0] * root_offset).as_quat(scalar_first=True)
        self.gmr.configuration.update(q)
        for _ in range(30):
            q = self._solve(human)
        self.ground_shift = self._sole_height(q) - 0.002
        q[2] -= self.ground_shift
        if not 0.4 < q[2] < 1.3:
            raise ValueError("Implausible calibrated GMR pelvis height; check axes/scale/standing pose")
        self.previous = q.copy()
        self.calibrated = True
        self.metrics = {"backend": "gmr", "ground_shift_m": self.ground_shift}
        return q.copy()

    def _solve(self, human):
        try:
            q = self.gmr.retarget(human)
        except Exception as exc:
            # Input loss or infeasible QP pauses the sender through its existing HOLD path.
            raise ValueError(f"GMR solve failed: {type(exc).__name__}: {exc}") from exc
        if q.shape != (34,) or not np.isfinite(q).all():
            raise ValueError("GMR returned invalid qpos")
        q[7:] = np.clip(q[7:], self.limits[:, 0] + 1e-6, self.limits[:, 1] - 1e-6)
        self.gmr.configuration.update(q)
        return q

    def retarget(self, raw):
        if not self.calibrated:
            raise RuntimeError("Calibrate GMR first")
        start = time.perf_counter()
        human = self.human_data(raw)
        # Always warm-start at the last emitted/HOLD reference, including slew caps.
        warm = self.previous.copy()
        warm[2] += self.ground_shift
        self.gmr.configuration.update(warm)
        q = self._solve(human)
        q[2] -= self.ground_shift
        if np.linalg.norm(q[:3] - self.previous[:3]) > 0.15:
            raise ValueError("GMR root jumped >15cm in one sample; check tracking/calibration")
        old = Rotation.from_quat(self.previous[3:7], scalar_first=True)
        if (Rotation.from_quat(q[3:7], scalar_first=True) * old.inv()).magnitude() > 0.35:
            raise ValueError("GMR root orientation jumped; check tracking/calibration")
        q[7:] = np.clip(q[7:], self.previous[7:] - self.max_step, self.previous[7:] + self.max_step)
        actual = q.copy()
        actual[2] += self.ground_shift
        self.data.qpos[:] = actual
        mujoco.mj_forward(self.model, self.data)
        errors, angles = [], []
        for body, (human_name, wp, wr, *_) in self.config["ik_match_table2"].items():
            i = self.model.body(body).id
            p, rot = self.gmr.scaled_human_data[human_name]
            if wp:
                errors.append(np.linalg.norm(p - self.data.xpos[i]))
            if wr:
                angles.append((Rotation.from_quat(rot, scalar_first=True)
                               * Rotation.from_matrix(self.data.xmat[i].reshape(3, 3)).inv()).magnitude())
        self.metrics = {
            "backend": "gmr", "max_position_error_m": float(max(errors)),
            "max_orientation_error_rad": float(max(angles)),
            "max_joint_step_rad": float(np.abs(q[7:] - self.previous[7:]).max()),
            "compute_ms": (time.perf_counter() - start) * 1000,
        }
        self.previous = q.copy()
        return q.copy()

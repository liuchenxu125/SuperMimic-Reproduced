"""Neutral-calibrated differential IK for PICO -> the selected Casbot02 bundle.

This independent front end uses MuJoCo body Jacobians and damped least squares.
It does not edit the robot registry, training assets, rewards, or motor constants.
It is intentionally a first real-time retargeter, not a reproduction of GMR's
offline fit or a guarantee of collision-free/balanced reference motion.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .pico import JOINTS, zup_body

# (PICO joint, robot body, position weight, orientation weight)
TARGETS = [
    ("Spine3", "waist_yaw_link", 2.0, 0.35),
    ("Left_Knee", "left_leg_knee_pitch_link", 2.0, 0.05),
    ("Right_Knee", "right_leg_knee_pitch_link", 2.0, 0.05),
    ("Left_Ankle", "left_leg_ankle_roll_link", 15.0, 0.4),
    ("Right_Ankle", "right_leg_ankle_roll_link", 15.0, 0.4),
    ("Left_Elbow", "left_elbow_pitch_link", 3.0, 0.1),
    ("Right_Elbow", "right_elbow_pitch_link", 3.0, 0.1),
    ("Left_Wrist", "left_wrist_roll_link", 5.0, 0.2),
    ("Right_Wrist", "right_wrist_roll_link", 5.0, 0.2),
]


class CasbotRetargeter:
    def __init__(self, bundle: Path, meta: dict, *, scale=None, max_joint_speed=6.0, iterations=5):
        self.model = mujoco.MjModel.from_xml_path(str(bundle / meta["model_file"]))
        self.data = mujoco.MjData(self.model)
        names = [self.model.joint(i).name.removeprefix("robot/") for i in range(1, self.model.njnt)]
        if names != meta["joint_names"] or self.model.nq != 34:
            raise ValueError("IK model and policy joint order differ")
        self.limits = self.model.jnt_range[1:].copy()
        self.q0 = np.array([m["q0"] for m in meta["motors"]])
        self.max_step = max_joint_speed / 50
        self.iterations, self.user_scale = iterations, scale
        self.ids = []
        for human, robot, wp, wr in TARGETS:
            self.ids.append((JOINTS.index(human), self.model.body("robot/" + robot).id, wp, wr))
        self.base = self.model.body(meta["root_body_name"]).id
        self.neutral = np.r_[np.zeros(3), [1.0, 0, 0, 0], self.q0]
        self.data.qpos[:] = self.neutral
        mujoco.mj_kinematics(self.model, self.data)
        # Put the two actual foot mesh soles on the virtual ground at calibration.
        lowest = []
        for name in meta["foot_geom_names"]:
            geom = self.model.geom(name).id
            mesh = int(self.model.geom_dataid[geom])
            if mesh < 0:
                raise ValueError("Expected foot meshes")
            start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
            v = self.model.mesh_vert[start : start + count]
            world = v @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]
            lowest.append(world[:, 2].min())
        self.neutral[2] = -min(lowest) + 0.002
        self.data.qpos[:] = self.neutral
        mujoco.mj_kinematics(self.model, self.data)
        self.robot_positions = self.data.xpos.copy()
        self.robot_rotations = self.data.xmat.reshape(-1, 3, 3).copy()
        self.calibrated = False
        self.previous = self.neutral.copy()
        self.metrics = {}

    def calibrate(self, body):
        pos, rot = zup_body(body)
        matrices = rot.as_matrix()
        # Calibration requires a plausible full-body standing skeleton, not zero poses.
        ankle_height = np.mean(pos[[JOINTS.index("Left_Ankle"), JOINTS.index("Right_Ankle")], 2])
        pelvis_height = pos[0, 2] - ankle_height
        if not 0.35 < pelvis_height < 1.5:
            raise ValueError(
                "Calibrate standing with valid full-body tracking (pelvis-to-ankle height invalid)"
            )
        self.scale = (
            self.user_scale if self.user_scale is not None else self.neutral[2] / pelvis_height
        )
        if not 0.5 <= self.scale <= 1.8:
            raise ValueError(
                "Unreasonable human/robot scale; check metres, axes and calibration pose"
            )
        self.human_position = pos.copy()
        self.human_rotation = matrices.copy()
        # VR world yaw and the SDK's anatomical pelvis axes need not match the
        # robot axes. Establish forward from the standing ankle line; map all
        # subsequent world/local displacements through this same calibration.
        left = pos[JOINTS.index("Left_Ankle")] - pos[JOINTS.index("Right_Ankle")]
        left[2] = 0.0
        if np.linalg.norm(left) < 0.08:
            raise ValueError("Calibrate with feet apart and valid left/right ankle tracking")
        forward = np.cross(left, [0.0, 0.0, 1.0])
        self.world_alignment = Rotation.from_euler(
            "z", -np.arctan2(forward[1], forward[0])
        ).as_matrix()
        self.local_alignment = self.world_alignment @ matrices[0]
        self.previous = self.neutral.copy()
        self.calibrated = True
        return self.neutral.copy()

    def targets(self, body):
        if not self.calibrated:
            raise RuntimeError("Calibrate first")
        pos, rotations = zup_body(body)
        rot = rotations.as_matrix()
        root_rotation = (
            self.world_alignment @ rot[0] @ self.human_rotation[0].T @ self.world_alignment.T
        )
        root_position = self.neutral[:3] + self.scale * self.world_alignment @ (
            pos[0] - self.human_position[0]
        )
        targets = []
        for human, robot, wp, wr in self.ids:
            local = rot[0].T @ (pos[human] - pos[0])
            local0 = self.human_rotation[0].T @ (
                self.human_position[human] - self.human_position[0]
            )
            p = root_position + root_rotation @ (
                self.robot_positions[robot]
                - self.neutral[:3]
                + self.scale * self.local_alignment @ (local - local0)
            )
            human_delta = (rot[0].T @ rot[human]) @ (
                self.human_rotation[0].T @ self.human_rotation[human]
            ).T
            r = (
                root_rotation
                @ self.local_alignment
                @ human_delta
                @ self.local_alignment.T
                @ self.robot_rotations[robot]
            )
            targets.append((robot, p, r, wp, wr))
        return root_position, root_rotation, targets

    def retarget(self, body):
        root, rotation, targets = self.targets(body)
        if np.linalg.norm(root - self.previous[:3]) > 0.15:
            raise ValueError("PICO root jumped >15cm in one sample; hold and recalibrate")
        old_rotation = Rotation.from_quat(self.previous[[4, 5, 6, 3]])
        if (Rotation.from_matrix(rotation) * old_rotation.inv()).magnitude() > 0.35:
            raise ValueError("PICO root orientation jumped; hold and recalibrate")
        q = self.previous.copy()
        q[:3] = root
        quat = Rotation.from_matrix(rotation).as_quat()
        q[3:7] = quat[[3, 0, 1, 2]]
        lo = np.maximum(self.limits[:, 0] + 1e-6, self.previous[7:] - self.max_step)
        hi = np.minimum(self.limits[:, 1] - 1e-6, self.previous[7:] + self.max_step)
        self.data.qpos[:] = q
        for _ in range(self.iterations):
            mujoco.mj_forward(self.model, self.data)
            jacobians, errors = [], []
            for body_id, p, r, wp, wr in targets:
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jp, jr, body_id)
                actual = self.data.xmat[body_id].reshape(3, 3)
                error_rotation = Rotation.from_matrix(r @ actual.T).as_rotvec()
                jacobians.extend([wp * jp[:, 6:], wr * jr[:, 6:]])
                errors.extend([wp * (p - self.data.xpos[body_id]), wr * error_rotation])
            a, b = np.concatenate(jacobians), np.concatenate(errors)
            # Weak posture prior resolves underconstrained wrist/yaw joints.
            delta = np.linalg.solve(
                a.T @ a + 0.02 * np.eye(27), a.T @ b + 0.002 * (self.q0 - self.data.qpos[7:])
            )
            self.data.qpos[7:] = np.clip(self.data.qpos[7:] + np.clip(delta, -0.08, 0.08), lo, hi)
        mujoco.mj_forward(self.model, self.data)
        self.metrics = {
            "max_position_error_m": max(
                np.linalg.norm(p - self.data.xpos[b]) for b, p, *_ in targets
            ),
            "max_joint_step_rad": float(np.abs(self.data.qpos[7:] - self.previous[7:]).max()),
        }
        self.previous = self.data.qpos.copy()
        return self.previous.copy()

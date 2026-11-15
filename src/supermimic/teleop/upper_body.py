"""Three-point PICO reference front end: two arms over an explicit standing reference.

The head supplies the common translation anchor. Its rotation does not directly
turn the robot, so looking around does not rotate the entire reference. No pelvis,
knees or elbows are inferred from missing full-body measurements.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .pico import zup_poses
from .retarget import CasbotRetargeter


class CasbotUpperBodyRetargeter(CasbotRetargeter):
    def __init__(self, bundle, meta, *, max_hand_displacement=0.35, **kwargs):
        super().__init__(bundle, meta, **kwargs)
        if not 0.05 <= max_hand_displacement <= 0.8:
            raise ValueError("Require hand workspace radius in 0.05..0.8 metres")
        self.workspace = max_hand_displacement
        self.active = np.array(
            [
                i
                for i, name in enumerate(meta["joint_names"])
                if name.startswith(("upper_left_", "upper_right_"))
            ]
        )
        if len(self.active) != 14:
            raise ValueError("Expected the fourteen Casbot02 arm joints")
        self.wrists = [
            self.model.body("robot/" + side + "_wrist_roll_link").id for side in ["left", "right"]
        ]

    def calibrate(self, poses):
        positions, rotations = zup_poses(poses, 3)
        relative = positions[1:] - positions[0]
        distances = np.linalg.norm(relative, axis=1)
        left = positions[1] - positions[2]
        left[2] = 0.0
        span = np.linalg.norm(left)
        if not 0.12 < span < 1.5 or np.any(distances < 0.15) or np.any(distances > 1.6):
            raise ValueError(
                "Stand upright, controllers separated at your sides; enable Head and Controller"
            )
        forward = np.cross(left, [0.0, 0.0, 1.0])
        self.world_alignment = Rotation.from_euler(
            "z", -np.arctan2(forward[1], forward[0])
        ).as_matrix()
        self.scale = 1.0 if self.user_scale is None else self.user_scale
        if not 0.5 <= self.scale <= 1.8:
            raise ValueError("Require upper-body scale in 0.5..1.8")
        self.relative0 = relative.copy()
        self.rotation0 = rotations.as_matrix()[1:].copy()
        self.previous_relative = relative.copy()
        self.previous_rotation = self.rotation0.copy()
        self.previous = self.neutral.copy()
        self.calibrated = True
        self.metrics = {}
        return self.neutral.copy()

    def retarget(self, poses):
        if not self.calibrated:
            raise RuntimeError("Calibrate first")
        positions, rotations = zup_poses(poses, 3)
        relative, rotations = positions[1:] - positions[0], rotations.as_matrix()[1:]
        if np.max(np.linalg.norm(relative - self.previous_relative, axis=1)) > 0.2:
            raise ValueError(
                "Controller jumped >20cm/sample; restore the paused pose or restart calibration"
            )
        angular_step = Rotation.from_matrix(
            rotations @ self.previous_rotation.transpose(0, 2, 1)
        ).magnitude()
        if np.max(angular_step) > 0.75:
            raise ValueError(
                "Controller orientation jumped; restore the paused pose or restart calibration"
            )
        displacements = self.scale * (relative - self.relative0) @ self.world_alignment.T
        lengths = np.linalg.norm(displacements, axis=1)
        displacements *= np.minimum(1.0, self.workspace / np.maximum(lengths, 1e-9))[:, None]
        targets = []
        for i, body in enumerate(self.wrists):
            delta = (
                self.world_alignment @ rotations[i] @ self.rotation0[i].T @ self.world_alignment.T
            )
            targets.append(
                (
                    body,
                    self.robot_positions[body] + displacements[i],
                    delta @ self.robot_rotations[body],
                )
            )
        # Only arm columns enter IK. Root, legs and waist keep the explicit q0
        # standing reference; the downstream actor still controls all 27 joints.
        self.data.qpos[:] = self.neutral
        self.data.qpos[7 + self.active] = self.previous[7 + self.active]
        lo = np.maximum(
            self.limits[self.active, 0] + 1e-6, self.previous[7 + self.active] - self.max_step
        )
        hi = np.minimum(
            self.limits[self.active, 1] - 1e-6, self.previous[7 + self.active] + self.max_step
        )
        for _ in range(self.iterations):
            mujoco.mj_forward(self.model, self.data)
            jacobians, errors = [], []
            for body, p, r in targets:
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jp, jr, body)
                actual = self.data.xmat[body].reshape(3, 3)
                jacobians.extend([5.0 * jp[:, 6 + self.active], 0.15 * jr[:, 6 + self.active]])
                errors.extend(
                    [
                        5.0 * (p - self.data.xpos[body]),
                        0.15 * Rotation.from_matrix(r @ actual.T).as_rotvec(),
                    ]
                )
            a, b = np.concatenate(jacobians), np.concatenate(errors)
            delta = np.linalg.solve(
                a.T @ a + 0.02 * np.eye(14),
                a.T @ b + 0.002 * (self.q0[self.active] - self.data.qpos[7 + self.active]),
            )
            self.data.qpos[7 + self.active] = np.clip(
                self.data.qpos[7 + self.active] + np.clip(delta, -0.08, 0.08), lo, hi
            )
        mujoco.mj_forward(self.model, self.data)
        self.metrics = {
            "max_position_error_m": max(
                float(np.linalg.norm(p - self.data.xpos[body])) for body, p, _ in targets
            ),
            "max_joint_step_rad": float(np.max(np.abs(self.data.qpos[7:] - self.previous[7:]))),
            "workspace_limited": bool(np.any(lengths > self.workspace)),
            "mode": "upper_body_standing_reference",
        }
        self.previous = self.data.qpos.copy()
        self.previous_relative, self.previous_rotation = relative.copy(), rotations.copy()
        return self.previous.copy()

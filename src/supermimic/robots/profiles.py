"""G1 and Casbot02 model profiles; Casbot02 assets/controls reuse local ProtoMotions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .layout import RobotLayout

CASBOT_ASSETS = Path(__file__).parent / "assets/casbot02"
G1_TRACKED = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
CASBOT_TRACKED = (
    "base_link",
    "left_leg_pelvic_roll_link",
    "left_leg_knee_pitch_link",
    "left_leg_ankle_roll_link",
    "right_leg_pelvic_roll_link",
    "right_leg_knee_pitch_link",
    "right_leg_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_wrist_roll_link",
)


@dataclass(frozen=True)
class RobotProfile:
    name: str
    num_joints: int
    root_body: str
    anchor_body: str
    com_body: str
    tracked_bodies: tuple[str, ...]
    foot_geom_regex: str

    @property
    def layout(self):
        return RobotLayout(self.num_joints)

    def spec(self):
        import mujoco

        if self.name == "g1":
            from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

            return get_spec()
        spec = mujoco.MjSpec.from_file(str(CASBOT_ASSETS / "xml/casbot02.xml"))
        # The attached scene owns simulator options (200 Hz physics in MujocoCfg).
        # Reset only source options that differ from MuJoCo defaults, avoiding
        # ambiguous attach-time overrides while leaving the copied XML untouched.
        default_spec = mujoco.MjSpec()
        defaults = default_spec.option
        for name in ("timestep", "iterations", "tolerance", "jacobian"):
            setattr(spec.option, name, getattr(defaults, name))
        # mjlab adds the implicit position actuators from the motor settings below.
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        # Named collision geoms allow foot friction randomization and ghost rendering.
        for i, geom in enumerate(spec.geoms):
            if not geom.name:
                geom.name = f"casbot_{i}_{'collision' if geom.contype else 'visual'}"
        return spec

    def model(self):
        model = self.spec().compile()
        if (model.nq, model.nv) != (self.layout.nq, self.layout.nv):
            raise ValueError(f"{self.name} model does not match its configured joint count")
        return model

    def motor_parameters(self):
        if self.name != "casbot02":
            raise ValueError("Explicit motor table is only provided for Casbot02")
        return json.loads((CASBOT_ASSETS / "motor_parameters.json").read_text())

    def action_scale(self):
        if self.name == "g1":
            from mjlab.asset_zoo.robots import G1_ACTION_SCALE

            return G1_ACTION_SCALE
        return {
            name: 0.25 * p["effort_limit"] / p["stiffness"]
            for name, p in self.motor_parameters()["motors"].items()
        }

    def entity_cfg(self):
        if self.name == "g1":
            from mjlab.asset_zoo.robots import get_g1_robot_cfg

            return get_g1_robot_cfg()
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

        params = self.motor_parameters()
        actuators = tuple(
            BuiltinPositionActuatorCfg(
                target_names_expr=(name,),
                stiffness=p["stiffness"],
                damping=p["damping"],
                effort_limit=p["effort_limit"],
                armature=p["armature"],
                frictionloss=p["friction"],
                viscous_damping=0.0,
            )
            for name, p in params["motors"].items()
        )
        return EntityCfg(
            spec_fn=self.spec,
            init_state=EntityCfg.InitialStateCfg(
                pos=(0, 0, 0.9), joint_pos=params["default_joint_pos"], joint_vel={".*": 0.0}
            ),
            articulation=EntityArticulationInfoCfg(
                actuators=actuators, soft_joint_pos_limit_factor=0.9
            ),
        )


def get_robot(name="g1") -> RobotProfile:
    if name == "g1":
        return RobotProfile(
            name,
            29,
            "pelvis",
            "torso_link",
            "torso_link",
            G1_TRACKED,
            r"^(left|right)_foot[1-7]_collision$",
        )
    if name == "casbot02":
        return RobotProfile(
            name,
            27,
            "base_link",
            "base_link",
            "base_link",
            CASBOT_TRACKED,
            r"^(left|right)_foot_collision$",
        )
    raise ValueError(f"Unknown robot {name!r}; expected 'g1' or 'casbot02'")

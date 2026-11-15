"""Read-only native checkpoint and matching export contract for human post-training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import torch

from supermimic.envs.mdp.motion_library import RefState, goal_features
from supermimic.kinematics.rotations import quat_apply_inverse, quat_from_angle_axis, quat_mul
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.teleop.reference import load_contract
from supermimic.world_model.heads import analytic_observation
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def policy_config(values):
    return PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def robot_state(ref):
    """Wire-style reference 67 -> native state 67 (angular velocity body-local)."""
    return torch.cat(
        [
            ref[..., :34],
            ref[..., 61:64],
            quat_apply_inverse(ref[..., 3:7], ref[..., 64:67]),
            ref[..., 34:61],
        ],
        -1,
    )


class FrozenTeacher:
    def __init__(self, checkpoint, bundle, device="cpu"):
        self.checkpoint, self.bundle = Path(checkpoint).resolve(), Path(bundle).resolve()
        self.meta = load_contract(self.bundle)
        self.sha256 = digest(self.checkpoint)
        if self.sha256 != self.meta["checkpoint_sha256"]:
            raise ValueError(
                "Teacher checkpoint and --bundle differ; use its matching verified export"
            )
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if digest(self.checkpoint) != self.sha256:
            raise ValueError(
                "Teacher checkpoint changed while loading; use a fixed numbered checkpoint"
            )
        if ck.get("config", {}).get("robot") != "casbot02":
            raise ValueError("This independent human task is for Casbot02")
        self.config = ck["config"]
        self.pcfg = policy_config(self.config["policy"])
        if (
            self.pcfg.num_joints,
            self.pcfg.history,
            self.pcfg.goal_dim,
            len(self.pcfg.obs_term_scale),
        ) != (27, 10, 690, 6):
            raise ValueError(
                "Human task must preserve the teacher's 930-history + 690-motion-goal contract"
            )
        if self.config["env"].get("anchor_obs") or not self.config["env"].get("anchor_ori_obs"):
            raise ValueError(
                "Expected native per-history 6D pelvis feedback without anchor position"
            )
        self.device = torch.device(device)
        self.policy = Policy(self.pcfg).to(device)
        self.policy.load_state_dict(ck["policy"])
        self.policy.requires_grad_(False).eval()
        wm_values = dict(self.config["world_model"])
        wm_values["state_spec"] = StateSpecCfg(**wm_values["state_spec"])
        self.wm = WorldModel(WorldModelCfg(**wm_values), heads=None, device=device)
        self.wm.load_state_dict(ck["wm"])
        self.wm.requires_grad_(False).eval()
        self.q0 = torch.tensor([m["q0"] for m in self.meta["motors"]], device=device)
        self.action_scale = torch.tensor([m["scale"] for m in self.meta["motors"]], device=device)
        model = mujoco.MjModel.from_xml_path(str(self.bundle / self.meta["model_file"]))
        names = [model.joint(i).name.removeprefix("robot/") for i in range(1, model.njnt)]
        if names != self.meta["joint_names"] or model.nq != 34:
            raise ValueError("Teacher model joint order differs from the actor")
        self.contract_sha256 = digest(self.bundle / "manifest.json")
        self.model_sha256 = digest(self.bundle / self.meta["model_file"])
        # Exported scenes also have static terrain bodies before the robot. The
        # original single-tree training FK expects the free root at body 1.
        # Resolve just the torso ancestor chain in this independent task instead.
        chain = []
        body = model.body("robot/waist_yaw_link").id
        while body:
            count = int(model.body_jntnum[body])
            joint = int(model.body_jntadr[body]) if count else None
            if count > 1:
                raise ValueError("Unsupported multi-joint torso body")
            if joint is not None and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                if model.jnt_qposadr[joint] != 0:
                    raise ValueError("Teacher free-root address must be zero")
                break
            fixed = torch.tensor(model.body_quat[body].copy(), dtype=torch.float32, device=device)
            axis = (
                None
                if joint is None
                else torch.tensor(model.jnt_axis[joint].copy(), dtype=torch.float32, device=device)
            )
            address = None if joint is None else int(model.jnt_qposadr[joint])
            rest = 0.0 if joint is None else float(model.qpos0[address])
            chain.append((fixed, axis, address, rest))
            body = int(model.body_parentid[body])
        if not body:
            raise ValueError("Torso does not descend from the teacher free root")
        self.anchor_chain = list(reversed(chain))
        self.noise_half = torch.tensor(
            [0.01] * 27 + [0.5] * 27 + [0.2] * 3 + [0.05] * 3 + [0.0] * 27 + [0.05] * 6,
            device=device,
        )

    def motion_goal(self, window):
        def ref(x):
            return RefState(
                x[..., :3], x[..., 3:7], x[..., 7:34], x[..., 61:64], x[..., 64:67], x[..., 34:61]
            )

        return goal_features(ref(window[:, 0]), ref(window), self.q0).flatten(1)

    def observation(self, state, previous_action, xi, ref_quat, noise):
        obs = analytic_observation(state, previous_action, xi, self.q0, ref_quat)
        if noise:
            obs = obs + (torch.rand_like(obs) * 2 - 1) * self.noise_half
        return obs

    def anchor_quat(self, state):
        # The same robot torso frame MorphMimic uses for the human v10 heading block.
        q = state[..., 3:7]
        for fixed, axis, address, rest in self.anchor_chain:
            q = quat_mul(q, fixed.expand_as(q))
            if axis is not None:
                rotation = quat_from_angle_axis(
                    state[..., address] - rest, axis.expand(*state.shape[:-1], 3)
                )
                q = quat_mul(q, rotation)
        return q

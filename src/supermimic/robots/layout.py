"""Floating-base layouts: one free joint followed by scalar hinge joints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotLayout:
    num_joints: int = 29

    @property
    def nq(self):
        return 7 + self.num_joints

    @property
    def nv(self):
        return 6 + self.num_joints

    @property
    def state_dim(self):
        return self.nq + self.nv

    @property
    def delta_dim(self):
        return self.state_dim - 1

    @property
    def obs_dim(self):
        return 3 * self.num_joints + 6

    @property
    def goal_dim(self):
        return 2 * self.num_joints + 10

    @property
    def xi_dim(self):
        return self.num_joints + 4

    @classmethod
    def from_state(cls, state):
        dim = state.shape[-1]
        if dim < 15 or (dim - 13) % 2:
            raise ValueError(f"Invalid floating-base state dimension: {dim}")
        return cls((dim - 13) // 2)


def split_state(state):
    n = RobotLayout.from_state(state).num_joints
    return state.split((3, 4, n, 3, 3, n), dim=-1)


def observation_terms(num_joints=29):
    return (
        ("joint_pos", num_joints),
        ("joint_vel", num_joints),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("actions", num_joints),
    )

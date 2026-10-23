"""Robot dimensions and model profiles shared by simulation and training."""

from .layout import RobotLayout, observation_terms, split_state
from .profiles import RobotProfile, get_robot

__all__ = ["RobotLayout", "RobotProfile", "get_robot", "observation_terms", "split_state"]

"""Extra anchor features appended once after the future-reference window.

``anchor_obs`` preserves the legacy position + orientation interface. The
independent ``anchor_ori_obs`` enables orientation without measured position.
Missing fields in old checkpoints mean False, preserving their input schema.
"""


def anchor_goal_dim(anchor_obs: bool = False, anchor_ori_obs: bool = False) -> int:
    return 9 if anchor_obs else 6 if anchor_ori_obs else 0

"""Extra anchor features appended once after the future-reference window.

``anchor_obs`` preserves the legacy position + orientation goal suffix.
``anchor_ori_obs`` does not change the goal: its rot6d is a separate proprio
history term, so a no-orientation run and an orientation run do not share
an observation width. Missing fields in old checkpoints mean False.
"""


def anchor_goal_dim(anchor_obs: bool = False, anchor_ori_obs: bool = False) -> int:
    del anchor_ori_obs  # Orientation feedback is not a goal suffix.
    return 9 if anchor_obs else 0

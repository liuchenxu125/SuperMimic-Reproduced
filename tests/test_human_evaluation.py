"""Full-clip actor routing uses real torso feedback without updating norms."""

from types import SimpleNamespace

import torch

from supermimic.human.evaluation import JointEvaluationActor
from supermimic.human.features import fixed_human_masks
from supermimic.human.model import HumanEncoder
from supermimic.policy.nets import Policy, PolicyCfg


def test_grouped_evaluation_matches_each_interface_and_preserves_normalizers():
    cfg = PolicyCfg(
        num_joints=27,
        goal_dim=690,
        obs_term_scale=(1.0, 5.0, 2.0, 1.0, 1.0, 1.0),
        encoder_hidden=(12, 8),
        decoder_hidden=(12, 8),
    )
    policy = Policy(cfg).eval()
    human = HumanEncoder(cfg).eval()
    frames = torch.tensor([4, 3, 2, 1])
    state = torch.zeros(4, 67)
    state[:, 3] = 1.0

    class Windows:
        def future(self, frame, length):
            poses = torch.zeros(len(frame), length, 24, 7)
            poses[..., 0] = frame[:, None, None] * 0.01
            poses[..., 2] = 1.0
            poses[..., 3] = 1.0
            return poses

    windows = Windows()
    actor = JointEvaluationActor(
        policy,
        human,
        windows,
        lambda s: s[:, 3:7],
        SimpleNamespace(read=lambda: state),
        SimpleNamespace(frame=frames),
        ["motion", "human_full", "human_five"],
    )
    obs = {"policy": torch.randn(4, 930) * 0.1, "goal": torch.randn(4, 690) * 0.1}
    modes = torch.tensor([0, 2, 1, 0])
    before = {k: v.clone() for k, v in human.named_buffers()}
    result = actor(obs, modes)
    for i, mode in enumerate(modes.tolist()):
        if mode == 0:
            expected = policy(obs["policy"][i : i + 1], obs["goal"][i : i + 1])
        else:
            poses = windows.future(frames[i : i + 1], 10)
            masks = fixed_human_masks(1, "full" if mode == 1 else "five", 10, torch.device("cpu"))
            goal = human.pack(
                poses[..., :3], poses[..., 3:], state[i : i + 1, 3:7], *masks, update=False
            )
            expected = policy.decoder(human(goal), obs["policy"][i : i + 1])
        torch.testing.assert_close(result[i], expected[0])
    assert all(torch.equal(before[k], v) for k, v in human.named_buffers())
    assert not human.training and not policy.training


def test_motion_evaluation_does_not_read_human_inputs():
    cfg = PolicyCfg(num_joints=27, goal_dim=690, encoder_hidden=(8,), decoder_hidden=(8,))
    policy = Policy(cfg).eval()
    human = HumanEncoder(cfg).eval()
    actor = JointEvaluationActor(policy, human, None, None, None, None, ["motion"])
    obs = {"policy": torch.zeros(2, 870), "goal": torch.zeros(2, 690)}
    torch.testing.assert_close(
        actor(obs, torch.zeros(2, dtype=torch.long)), policy(obs["policy"], obs["goal"])
    )

"""Replay chronology, reset boundaries, terminal targets and observation histories."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer


def append(buf, t, *, first=False, done=False, offset=0, ids=None):
    n = buf.N if ids is None else len(ids)
    state = torch.full((n, 71), float(offset + t))
    action = torch.full((n, 29), float(t + 1))
    buf.add_step(
        state,
        action,
        action + 0.125,
        torch.full((n, 33), float(offset)),
        torch.ones(n),
        torch.full((n,), done),
        torch.zeros(n, dtype=torch.long),
        torch.full((n,), t, dtype=torch.long),
        torch.full((n,), first),
        env_ids=ids,
    )
    return state, action


@pytest.mark.parametrize("terminated", [True, False])
def test_terminal_state_is_target_without_reset_crossing(terminated):
    buf = ReplayBuffer(2, 32, "cpu")
    for episode in range(3):
        for t in range(4):
            append(buf, t, first=t == 0, done=terminated and t == 3, offset=100 * episode)
        buf.add_terminal(torch.arange(2), torch.full((2, 71), float(100 * episode + 4)))
    windows = buf.sample_windows(512, 4)
    assert windows.states.shape == (5, 512, 71)
    assert torch.equal(windows.states[1:] - windows.states[:-1], torch.ones(4, 512, 71))
    assert torch.equal(windows.frame, torch.arange(4)[:, None].expand(4, 512))
    assert torch.equal(windows.done[-1], torch.full((512,), float(terminated)))
    assert not windows.done[:-1].any()
    assert torch.equal(windows.a_exec - windows.a_clean, torch.full((4, 512, 29), 0.125))
    assert torch.equal(windows.xi[:, 0], windows.states[0, :, 0])


def test_ring_wrap_and_overwritten_history():
    buf = ReplayBuffer(1, 7, "cpu")
    for t in range(25):
        append(buf, t, first=t == 0)
    windows = buf.sample_windows(256, 3)
    assert windows.states.min() >= 18 and windows.states.max() <= 24
    assert torch.equal(windows.states[1:] - windows.states[:-1], torch.ones(3, 256, 71))
    starts = buf.sample_starts(256, hist=10)
    assert torch.equal(starts.hist_states[-1], starts.state)
    expected = torch.maximum(
        starts.state[None, :, 0] - torch.arange(9, -1, -1)[:, None], torch.tensor(18)
    )
    assert torch.equal(starts.hist_states[:, :, 0], expected)
    # At state t, previous action is a_{t-1} = t, including the oldest retained row.
    assert torch.equal(starts.hist_actions[:, :, 0], expected)


def test_episode_history_repeats_reset_state_and_zero_previous_action():
    buf = ReplayBuffer(1, 10, "cpu")
    append(buf, 0, first=True)
    buf.add_terminal(torch.tensor([0]), torch.ones(1, 71))
    for t in range(3):
        append(buf, t, first=t == 0, offset=100)
    starts = buf.sample_starts(512, hist=5)
    assert not ((starts.state[:, 0] == 1) & (starts.xi[:, 0] == 0)).any()
    new = starts.xi[:, 0] == 100
    assert new.any()
    h = starts.hist_states[:, new, 0]
    expected = torch.maximum(
        starts.state[new, 0][None] - torch.arange(4, -1, -1)[:, None], torch.tensor(100)
    )
    assert torch.equal(h, expected)
    assert torch.equal(starts.hist_actions[:, new, 0], expected - 100)


def test_partial_env_writes_copy_inputs_and_fail_clearly_when_no_windows():
    buf = ReplayBuffer(3, 5, "cpu")
    with pytest.raises(RuntimeError, match="empty"):
        buf.sample_starts(1)
    ids = torch.tensor([2])
    state, action = append(buf, 0, first=True, ids=ids)
    state.fill_(99)
    action.fill_(99)
    starts = buf.sample_starts(10)
    assert not starts.state.any()  # Caller mutations must not corrupt replay.
    assert buf.size == 1
    with pytest.raises(RuntimeError, match="contiguous"):
        buf.sample_windows(1, 2)
    with pytest.raises(ValueError, match="Window length"):
        buf.sample_windows(1, 5)
    with pytest.raises(ValueError, match="positive"):
        buf.sample_starts(0)


@pytest.mark.parametrize("auto_reset", [True, False])
@pytest.mark.parametrize("terminated", [True, False])
def test_recorder_captures_terminal_before_actions_and_context_reset(
    monkeypatch, auto_reset, terminated
):
    from supermimic.data.recorder import ReplayRecorder, ReplayRecorderCfg
    from supermimic.envs import sim_context, state_io

    class FakeStateIO:
        def __init__(self, env):
            self.env = env

        def read(self):
            return self.env.state.clone()

    class FakeContext(FakeStateIO):
        def __init__(self, env, *args):
            super().__init__(env)

        def read(self):
            return self.env.xi.clone()

    monkeypatch.setattr(state_io, "StateIO", FakeStateIO)
    monkeypatch.setattr(sim_context, "SimContext", FakeContext)
    command = SimpleNamespace(
        frame=torch.zeros(2, dtype=torch.long),
        library=SimpleNamespace(frame_clip=torch.zeros(50, dtype=torch.long)),
    )
    action = SimpleNamespace(executed_raw=torch.zeros(2, 29))
    manager = SimpleNamespace(action=torch.zeros(2, 29), get_term=lambda _: action)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        cfg=SimpleNamespace(auto_reset=auto_reset),
        state=torch.zeros(2, 71),
        xi=torch.ones(2, 33),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        action_manager=manager,
        reward_buf=torch.ones(2),
        reset_terminated=torch.tensor([terminated, False]),
        reset_buf=torch.tensor([True, False]),
    )
    rec = ReplayRecorder(ReplayRecorderCfg(params={"capacity": 16, "foot_geom_regex": ".*"}), env)
    rec.record_post_reset(torch.arange(2))
    manager.action.fill_(2)
    action.executed_raw.fill_(2.125)
    env.state.fill_(1)  # True post-action state, before reset.
    if auto_reset:
        rec.record_pre_reset(torch.tensor([0]))
    else:
        rec.record_post_step()
    env.state[0] = 100  # Auto-reset destroys the terminal state and action.
    env.xi[0] = 100
    manager.action[0] = 0
    action.executed_raw[0] = 0
    command.frame[:] = torch.tensor([20, 1])
    rec.record_post_reset(torch.tensor([0]))
    if auto_reset:
        rec.record_post_step()
    windows = rec.buffer.sample_windows(64, 1)
    assert torch.equal(windows.states[0], torch.zeros(64, 71))
    assert torch.equal(windows.states[1], torch.ones(64, 71))
    assert torch.equal(windows.a_clean, torch.full((1, 64, 29), 2.0))
    assert torch.equal(windows.a_exec, torch.full((1, 64, 29), 2.125))
    assert torch.equal(windows.xi, torch.ones(64, 33))
    # Only env 0 has a successor row so far; timeouts are not physical terminals.
    assert torch.equal(windows.done, torch.full((1, 64), float(terminated)))
    assert rec._first[0] and not rec._first[1]

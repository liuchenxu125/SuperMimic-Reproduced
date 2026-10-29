"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * full clip length). Unplayed steps after failure
contribute zero; the survived-prefix score is retained as a separate diagnostic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.policy.nets import Policy, PolicyCfg


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    norm_return_survived: float | None = None
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)


def clip_result(name, frames, steps, reward_sum, failed, max_step_reward=0.1, terms=None):
    """The current command clock executes L steps for an L-frame full clip.

    Reset advances the target to frame 1; the final frame is scored again on
    the clip-end timeout. Check that contract instead of hiding early timeouts.
    """
    if frames < 1 or steps < 1 or steps > frames or (not failed and steps != frames):
        raise ValueError(f"{name}: invalid full-clip evaluation length {steps}/{frames}")
    return ClipResult(
        name=name,
        frames=frames,
        steps=steps,
        norm_return=reward_sum / (max_step_reward * frames),
        norm_return_survived=reward_sum / (max_step_reward * steps),
        failed=failed,
        fail_step=steps if failed else -1,
        terms=terms,
    )


def load_policy_from_checkpoint(path: str | Path | BinaryIO, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    cfg = dict(ck["config"])
    cfg["_checkpoint"] = {
        key: ck.get(key) for key in ("iteration", "iteration_unit", "env_steps", "num_updates")
    }
    cfg["_checkpoint"]["iteration_unit"] = ck.get("iteration_unit", "env_step")
    return policy, cfg


@torch.no_grad()
def run_benchmark(
    policy: Policy,
    npz_files: list[str],
    device: str = "cuda:0",
    num_envs: int | None = None,
    future_frames: int = 10,
    verbose: bool = True,
    anchor_obs: bool = False,
    robot: str = "g1",
    anchor_ori_obs: bool = False,
    actuator_delay: bool = False,
    actuator_delay_mode: str = "episode",
    collision_model: str = "mesh",
) -> list[ClipResult]:
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        robot=robot,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        anchor_ori_obs=anchor_ori_obs,
        actuator_delay=actuator_delay,
        actuator_delay_mode=actuator_delay_mode,
        collision_model=collision_model,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = clip_result(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            reward_sum=float(ret_sum[e]),
                            max_step_reward=max_r,
                            failed=bool(term[e]),
                            terms={
                                n: float(term_sum[e, i]) / max(st, 1)
                                for i, n in enumerate(term_names)
                            },
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Weight by full reference lengths, including unplayed suffixes after failure.
    tot_frames = sum(r.frames for r in results)
    w_ret = sum(r.norm_return * r.frames for r in results) / max(tot_frames, 1)
    survived = sum(
        r.norm_return_survived if r.norm_return_survived is not None else r.norm_return
        for r in results
    ) / max(n, 1)
    out = {
        "clips": n,
        "failed": failed,
        "mean_norm_return": mean_ret,
        "step_weighted_norm_return": w_ret,
        "frame_weighted_norm_return": w_ret,
        "mean_norm_return_survived": survived,
    }
    if results and results[0].terms:
        tot_steps = sum(r.steps for r in results)
        out["terms"] = {
            k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1)
            for k in results[0].terms
        }
    return out


def save_results(
    results: list[ClipResult], path: str | Path, meta: dict | None = None,
    *, evaluation: dict | None = None, shard: dict | None = None,
) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "normalization": "full_clip",
                "checkpoint": meta or {},
                "summary": summarize(results),
                "clips": [asdict(r) for r in results],
                **({"evaluation": evaluation} if evaluation is not None else {}),
                **({"shard": shard} if shard is not None else {}),
            },
            indent=1,
        )
    )

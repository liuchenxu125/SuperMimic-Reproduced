#!/usr/bin/env python3
"""Full-clip MuJoCo benchmark of motion / human-full / human-five interfaces."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--interfaces",
        nargs="+",
        choices=["motion", "human_full", "human_five"],
        default=["motion", "human_full", "human_five"],
    )
    p.add_argument("--num-envs", type=int, default=1536)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--clips", nargs="+")
    p.add_argument(
        "--actuator-delay",
        action="store_true",
        help="Optional separate sim2real robustness protocol",
    )
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if len(set(a.interfaces)) != len(a.interfaces) or a.num_envs < 1 or a.limit < 0:
        p.error("Need unique interfaces, positive env count and nonnegative limit")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/supermimic-eval-mpl")
    import torch

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import ClipResult, clip_result, save_results, summarize
    from supermimic.human.data import read_pairs
    from supermimic.human.evaluation import JointEvaluationActor
    from supermimic.human.joint_data import HumanWindows
    from supermimic.human.joint_student import TorsoFrame
    from supermimic.human.joint_training import FORMAT, robot_contract
    from supermimic.human.model import HumanEncoder
    from supermimic.human.teacher import policy_config
    from supermimic.policy.nets import Policy
    from supermimic.training.precision import configure_precision

    torch.set_num_threads(1)
    if a.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(a.device))
    with a.checkpoint.open("rb") as f:
        checkpoint_sha = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        ck = torch.load(f, map_location="cpu", weights_only=False)
    if ck.get("joint_format") != FORMAT:
        raise ValueError("Expected motion+human joint checkpoint")
    meta, model = robot_contract()
    if meta != ck["robot_contract"]:
        raise ValueError("Current robot assets/motor parameters differ from training checkpoint")
    cfg = ck["config"]
    pcfg = policy_config(cfg["policy"])
    configure_precision(cfg.get("tf32", True))
    names = set(ck["pair_split"]["train"]) | set(ck["pair_split"]["validation"])
    pairs = read_pairs(ck["joint_config"]["pairs"])
    pairs = sorted([pair for pair in pairs if pair.name in names], key=lambda pair: pair.name)
    if a.clips:
        requested = set(a.clips)
        if not requested <= {pair.name for pair in pairs}:
            raise ValueError("Unknown requested pair")
        pairs = [pair for pair in pairs if pair.name in requested]
    if a.limit:
        pairs = pairs[: a.limit]
    files = [str(Path(cfg["data"]["npz_dir"]) / (pair.name + ".npz")) for pair in pairs]
    if not pairs or any(not Path(f).is_file() for f in files):
        raise FileNotFoundError("Missing checkpoint staged robot clips")
    # Verify the selected source metadata against what was used in training.
    saved = {record["name"]: record for record in ck["pair_identity"]["files"]}
    for pair in pairs:
        record = saved[pair.name]
        for path, old in zip((pair.robot_npz, pair.human_npz), record["files"]):
            stat = path.stat()
            if stat.st_size != old["size"] or stat.st_mtime_ns != old["mtime_ns"]:
                raise ValueError(f"Training pair changed: {pair.name}, {path}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "checkpoint_sha256": checkpoint_sha,
        "iteration": ck["iteration"],
        "interfaces": a.interfaces,
        "names_sha256": hashlib.sha256(json.dumps([x.name for x in pairs]).encode()).hexdigest(),
        "clips": len(pairs),
        "normalization": "full_clip",
        "actuator_delay": a.actuator_delay,
        "reset_frame": 0,
        "observation_noise": False,
        "action_noise": False,
        "domain_randomization": False,
        "human_reference": "paired SOMA uniform, full 10-frame preview; real robot torso",
        "note": "In-dataset deterministic full-clip tracking; not PICO live deployment or held-out generalization",
    }
    signature_path = a.output_dir / "evaluation.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output directory belongs to a different evaluation")
    signature_path.write_text(json.dumps(signature, indent=2))
    journal = a.output_dir / "clips.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            key = (record["interface"], record["clip"]["name"])
            if key in done:
                raise ValueError("Duplicate result in evaluation journal")
            done[key] = record["clip"]
    job_records = [
        (i, m)
        for i, pair in enumerate(pairs)
        for m, name in enumerate(a.interfaces)
        if (name, pair.name) not in done
    ]
    print(
        f"[EVAL] checkpoint={ck['iteration']}; {len(pairs)} clips x {len(a.interfaces)} interfaces; "
        f"{len(done)} done, {len(job_records)} pending",
        flush=True,
    )
    if not job_records:
        write_reports(a, pairs, done, signature, ClipResult, save_results, summarize)
        return
    env = None
    started = time.monotonic()
    try:
        n = min(a.num_envs, len(job_records))
        args = SuperMimicEnvCfgArgs(
            npz_files=files,
            robot=cfg["robot"],
            num_envs=n,
            future_frames=cfg["env"]["future_frames"],
            history_length=pcfg.history,
            episode_length_s=1.0e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=0,
            anchor_obs=cfg["env"]["anchor_obs"],
            anchor_ori_obs=cfg["env"]["anchor_ori_obs"],
            actuator_delay=a.actuator_delay,
            actuator_delay_mode=cfg["env"].get("actuator_delay_mode", "episode"),
            collision_model=cfg["env"]["collision_model"],
            compact_library=True,
            library_device="cpu",
            library_transfer="pinned",
        )
        env = make_env(args, device=a.device, tf32=cfg.get("tf32", True))
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        if [clip.name for clip in lib.clips] != [pair.name for pair in pairs]:
            raise ValueError("Evaluation library ordering differs from paired data")
        policy = Policy(pcfg).to(a.device).eval()
        policy.load_state_dict(ck["policy"])
        human = HumanEncoder(pcfg).to(a.device).eval()
        human.load_state_dict(ck["human"])
        windows = (
            HumanWindows(pairs, meta, lib, cache_clips=0)
            if any(x != "motion" for x in a.interfaces)
            else None
        )
        actor = JointEvaluationActor(
            policy, human, windows, TorsoFrame(model, a.device), StateIO(env), cmd, a.interfaces
        )
        jobs = torch.tensor(job_records, dtype=torch.long, device=a.device)
        current = torch.arange(n, device=a.device)
        nxt = current + n
        cmd.forced_frame[:] = lib.clip_start[jobs[current, 0]]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[jobs[nxt.clamp_max(len(jobs) - 1), 0]]
        active = torch.ones(n, dtype=torch.bool, device=a.device)
        reward_sum = torch.zeros(n, device=a.device)
        steps = torch.zeros(n, dtype=torch.long, device=a.device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(n, len(term_names), device=a.device)
        initial_done = len(done)
        last_progress = started
        with journal.open("a", buffering=1) as output, torch.inference_mode():
            while len(done) < len(pairs) * len(a.interfaces):
                modes = jobs[current.clamp_max(len(jobs) - 1), 1]
                actions = actor(obs, modes)
                obs, reward, term, tout, _ = env.step(actions)
                reward_sum += reward
                steps += 1
                term_sum += env.reward_manager.term_rewards
                ended = (term | tout) & active
                if ended.any():
                    ids = ended.nonzero().flatten()
                    for e in ids.tolist():
                        clip_index, mode = job_records[int(current[e])]
                        name = a.interfaces[mode]
                        clip = lib.clips[clip_index]
                        st = int(steps[e])
                        result = clip_result(
                            clip.name,
                            clip.length,
                            st,
                            float(reward_sum[e]),
                            bool(term[e]),
                            max_step_reward=0.1,
                            terms={
                                key: float(term_sum[e, i]) / st for i, key in enumerate(term_names)
                            },
                        )
                        key = (name, clip.name)
                        if key in done:
                            raise RuntimeError("Evaluation queue repeated a completed job")
                        done[key] = asdict(result)
                        output.write(
                            json.dumps({"interface": name, "clip": done[key]}, ensure_ascii=False)
                            + "\n"
                        )
                    current[ids] = nxt[ids]
                    nxt[ids] += n
                    active[ids[current[ids] >= len(jobs)]] = False
                    cmd.forced_frame[ids] = lib.clip_start[
                        jobs[nxt[ids].clamp_max(len(jobs) - 1), 0]
                    ]
                    reward_sum[ids] = 0
                    steps[ids] = 0
                    term_sum[ids] = 0
                now = time.monotonic()
                if now - last_progress >= 30 or len(done) == len(pairs) * len(a.interfaces):
                    progress = {
                        name: {
                            "evaluated": sum(k[0] == name for k in done),
                            "completed": sum(
                                k[0] == name and not v["failed"] for k, v in done.items()
                            ),
                        }
                        for name in a.interfaces
                    }
                    elapsed = now - started
                    rate = (len(done) - initial_done) / max(elapsed, 1.0e-6)
                    report = {
                        "done": len(done),
                        "total": len(pairs) * len(a.interfaces),
                        "seconds": elapsed,
                        "jobs_per_second": rate,
                        "interfaces": progress,
                    }
                    (a.output_dir / "progress.json").write_text(json.dumps(report, indent=2))
                    print("[EVAL] " + json.dumps(report), flush=True)
                    last_progress = now
        write_reports(a, pairs, done, signature, ClipResult, save_results, summarize)
    finally:
        if env is not None:
            env.close()


def write_reports(a, pairs, done, signature, result_type, save_results, summarize):
    summaries = {}
    for name in a.interfaces:
        results = [result_type(**done[(name, pair.name)]) for pair in pairs]
        save_results(
            results,
            a.output_dir / (name + ".json"),
            meta={"iteration": signature["iteration"], "sha256": signature["checkpoint_sha256"]},
            evaluation={**signature, "interface": name},
        )
        summary = summarize(results)
        summary["completed"] = summary["clips"] - summary["failed"]
        summary["completion_rate"] = summary["completed"] / summary["clips"]
        summary["failed_within_1s"] = sum(x.failed and x.fail_step <= 50 for x in results)
        summaries[name] = summary
    (a.output_dir / "summary.json").write_text(
        json.dumps({"evaluation": signature, "interfaces": summaries}, indent=2)
    )
    print("[EVAL COMPLETE] " + json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()

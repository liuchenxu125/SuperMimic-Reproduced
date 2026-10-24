"""Test 640-goal C++/ROS interfaces with a TEMPORARY random actor, not deployable weights.

This does not train a policy or claim motion-tracking performance. It verifies
the input contract and that only IMU + joints are needed by the new ROS path.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
import yaml

from supermimic.deployment.export import ActorOnly
from supermimic.deployment.sim2sim import History, Reference, observation, quat_multiply, rotation
from supermimic.policy.nets import Policy, PolicyCfg


def parity(probe, bundle, actor, reference, q0, scale, shifted=False):
    process = subprocess.Popen(
        [str(probe.resolve()), str(bundle)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    history = History(27, 10)
    previous = np.zeros(27, dtype=np.float32)
    errors = {"observation": 0.0, "goal": 0.0, "actor": 0.0, "target": 0.0}
    outputs = []
    try:
        for t in range(64):
            qp = np.r_[reference.root_pos[t], reference.root_quat[t], reference.joint_pos[t]]
            if shifted:
                qp[:3] = np.nan  # Unavailable position must never enter proprio inference.
                dq = np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)])
                qp[3:7] = quat_multiply(dq, qp[3:7])
            omega = rotation(reference.root_quat[t]).T @ reference.root_ang[t]
            qv = np.r_[np.zeros(3), omega, reference.joint_vel[t]]
            frame = observation(qp, qv, previous, q0)
            if t == 0:
                history.reset(frame)
            else:
                history.append(frame)
            obs = history.packed()
            goal = reference.goal(t + 1, qp[:3], qp[3:7], False)
            state = np.r_[qp[7:], qv[6:], qp[:3], qp[3:7], omega].astype("<f8")
            process.stdin.write(state.tobytes())
            process.stdin.flush()
            raw = process.stdout.read(1564 * 4)
            if len(raw) != 1564 * 4:
                raise RuntimeError("C++ probe failed")
            row = np.frombuffer(raw, dtype="<f4")
            with torch.inference_mode():
                action = actor(torch.from_numpy(obs[None]), torch.from_numpy(goal[None]))[0].numpy()
            expected = [obs, goal, action, q0 + scale * row[1510:1537]]
            actual = [row[:870], row[870:1510], row[1510:1537], row[1537:]]
            for name, a, b in zip(errors, actual, expected):
                errors[name] = max(errors[name], float(np.max(np.abs(a - b))))
                np.testing.assert_allclose(a, b, atol=2e-6, rtol=3e-6)
            previous = row[1510:1537].copy()
            outputs.append(row.copy())
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait(timeout=5) != 0:
            raise RuntimeError("C++ probe exited with error")
    return errors, np.stack(outputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--probe", type=Path, default=Path("rl_controller/build/supermimic_ros/supermimic_probe")
    )
    p.add_argument(
        "--runner", type=Path, default=Path("rl_controller/build/supermimic_ros/supermimic_runner")
    )
    p.add_argument("--with-ros", action="store_true")
    p.add_argument(
        "--report", type=Path, default=Path("outputs/casbot02_hardware/proprio_interface_test.json")
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = root / "rl_controller/config/model/supermimic_walk4"
    meta = json.loads((original / "manifest.json").read_text())
    meta.update(
        goal_dim=640,
        anchor_obs=False,
        test_only=True,
        checkpoint_sha256="random-interface-fixture",
        checkpoint_iteration=0,
    )
    torch.set_num_threads(2)
    torch.manual_seed(123)
    actor = ActorOnly(
        Policy(
            PolicyCfg(num_joints=27, goal_dim=640, encoder_hidden=(32,), decoder_hidden=(32,))
        ).eval()
    )
    q0 = np.array([m["q0"] for m in meta["motors"]])
    scale = np.array([m["scale"] for m in meta["motors"]])
    reference = Reference(
        root / "outputs/casbot02_sim2sim/motions/walk4_subject1.npz", meta["joint_names"], 10, 0.02
    )
    with tempfile.TemporaryDirectory(prefix="supermimic_640_interface_") as temp:
        bundle = Path(temp)
        shutil.copy(original / "motion.bin", bundle / "motion.bin")
        torch.onnx.export(
            actor,
            (torch.zeros(1, 870), torch.zeros(1, 640)),
            str(bundle / "actor.onnx"),
            input_names=["obs_hist", "goal"],
            output_names=["action"],
            opset_version=17,
            dynamo=False,
        )
        (bundle / "manifest.json").write_text(json.dumps(meta))
        errors, baseline = parity(args.probe, bundle, actor, reference, q0, scale)
        _, shifted = parity(args.probe, bundle, actor, reference, q0, scale, True)
        np.testing.assert_allclose(baseline, shifted, atol=2e-6, rtol=3e-6)
        report = {
            "test_only_random_actor": True,
            "trained_weights_created": False,
            "goal_dim": 640,
            "cpp_frames_checked": 128,
            "max_abs_errors": errors,
            "position_and_global_yaw_invariant": True,
            "unavailable_position_nan_accepted": True,
        }
        # A malformed external ONNX must fail startup, before a controller can
        # publish even preparation commands. Keep this fixture in /tmp only.
        broken = bundle / "nonfinite_actor"
        broken.mkdir()
        shutil.copy(bundle / "manifest.json", broken / "manifest.json")
        shutil.copy(bundle / "motion.bin", broken / "motion.bin")
        graph = onnx.load(bundle / "actor.onnx")
        parameter = graph.graph.initializer[-1]
        values = onnx.numpy_helper.to_array(parameter).copy()
        values.reshape(-1)[0] = np.nan
        parameter.CopyFrom(onnx.numpy_helper.from_array(values, parameter.name))
        onnx.save(graph, broken / "actor.onnx")
        rejected = subprocess.run(
            [str(args.probe.resolve()), str(broken)],
            check=False,
            input=b"",
            capture_output=True,
            timeout=10,
        )
        if rejected.returncode == 0 or b"Nonfinite actor output" not in rejected.stderr:
            raise RuntimeError("C++ did not reject nonfinite ONNX: " + rejected.stderr.decode())
        report["nonfinite_onnx_rejected_at_startup"] = True
        if args.with_ros:
            config = yaml.safe_load(
                (root / "rl_controller/config/supermimic_walk4_proprio.yaml").read_text()
            )
            config["bundle"] = str(bundle)
            path = bundle / "test.yaml"
            path.write_text(yaml.safe_dump(config))
            # Fixed script paths; all ROS topics are isolated synthetic test topics.
            msg_setup = (
                root / "rl_controller/build/supermimic_msgs/share/crb_ros_msg/local_setup.bash"
            )
            if not msg_setup.exists():
                msg_setup = Path("/workspace/hl_motion/share/crb_ros_msg/local_setup.bash")
            command = [
                "bash",
                "-c",
                (
                    'source /opt/ros/humble/setup.bash && source "$1" && '
                    'exec /usr/bin/python3 "$2" --runner "$3" --config "$4"'
                ),
                "supermimic-interface-test",
                str(msg_setup),
                str(root / "rl_controller/tests/test_supermimic_ros.py"),
                str(args.runner.resolve()),
                str(path),
            ]
            env = os.environ.copy()
            env["ROS_LOG_DIR"] = "/tmp/supermimic_ros_logs"
            result = subprocess.run(command, check=False, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            report["ros"] = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith('{"dry_run"')
            ]
            assert len(report["ros"]) == 2
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

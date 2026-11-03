from types import SimpleNamespace

import mujoco
import numpy as np

from supermimic.data.bones_seed import (
    casbot_foot_model,
    casbot_foot_sole_height,
    feasibility_arrays,
)
from supermimic.data.motion_convert import forward_kinematics, resample_motion
from supermimic.robots import get_robot


def standing_arrays(height_offset=0):
    model, _, height = casbot_foot_model()
    defaults = get_robot("casbot02").motor_parameters()["default_joint_pos"]
    raw = np.zeros((50, 34))
    raw[:, 2], raw[:, 6] = height + height_offset, 1
    raw[:, 7:] = [defaults[model.joint(j).name] for j in range(1, model.njnt)]
    motion = resample_motion(raw, 50, 50)
    body = forward_kinematics(model, motion)
    return {
        "robot": np.array("casbot02"),
        "fps": np.array([50]),
        "body_names": np.array(body.body_names),
        "body_pos_w": body.pos_w,
        "body_quat_w": body.quat_w,
        "body_lin_vel_w": body.lin_vel_w,
        "joint_pos": motion.joint_pos,
        "joint_vel": motion.joint_vel,
    }, raw


def test_casbot_feasibility_uses_its_own_feet_and_stature():
    arrays, _ = standing_arrays()
    assert feasibility_arrays(arrays, "casbot02") is None
    arrays, _ = standing_arrays(0.03)
    assert feasibility_arrays(arrays, "casbot02") == "floating_feet"
    arrays, _ = standing_arrays(0.30)
    assert feasibility_arrays(arrays, "casbot02") == "elevated_support"
    arrays, _ = standing_arrays(-0.25)
    assert feasibility_arrays(arrays, "casbot02") == "ground_penetration"


def test_casbot_sole_geometry_matches_native_mujoco():
    model, geoms, _ = casbot_foot_model()
    arrays, raw = standing_arrays()
    # Add nontrivial root and ankle rotations; verify the analytic capsule bottom.
    from scipy.spatial.transform import Rotation

    raw[:, 3:7] = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat()
    raw[:, 7:] += np.random.default_rng(12).normal(0, 0.05, (50, 27))
    motion = resample_motion(raw, 50, 50)
    body = forward_kinematics(model, motion)
    arrays.update(body_pos_w=body.pos_w, body_quat_w=body.quat_w)
    actual = casbot_foot_sole_height(arrays, np.ones(50, dtype=bool))
    data, expected = mujoco.MjData(model), []
    vertices = {}
    for geom in geoms:
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices[geom] = model.mesh_vert[start:start + count]
    for i in range(50):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = (
            motion.root_pos[i],
            motion.root_quat[i],
            motion.joint_pos[i],
        )
        mujoco.mj_kinematics(model, data)
        lows = []
        for geom in geoms:
            rotation = data.geom_xmat[geom].reshape(3, 3)
            lows.append((data.geom_xpos[geom, 2] + vertices[geom] @ rotation.T[:, 2]).min())
        expected.append(min(lows))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_csv_pipeline_filters_before_writing_and_resumes(tmp_path):
    import json

    from supermimic.data.bones_preparation import prepare_csv_dataset

    source, output = tmp_path / "csv", tmp_path / "npz"
    source.mkdir()
    _, raw = standing_arrays()
    for name, z_offset in [("walk", 0), ("floating", 0.03), ("chair", 0)]:
        clip = raw.copy()
        clip[:, 2] += z_offset
        np.savetxt(source / f"{name}.csv", clip, delimiter=",")
    invalid = raw.copy()
    invalid[:, 3:7] = 0
    np.savetxt(source / "invalid.csv", invalid, delimiter=",")
    args = SimpleNamespace(
        robot="casbot02",
        output_dir=output,
        input_fps=50,
        csv_dir=source,
        pattern="*.csv",
        limit=None,
        tracked_bodies_only=True,
        workers=1,
        progress_every=10,
        out=None,
    )
    prepare_csv_dataset(args)
    assert [p.stem for p in output.glob("*.npz")] == ["walk"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete_with_invalid_sources"
    assert any(reason.startswith("invalid:ValueError:") for reason in summary["reasons"])
    with np.load(output / "walk.npz") as clip:
        assert clip["joint_pos"].shape == (50, 27)
        assert tuple(clip["body_names"]) == get_robot("casbot02").tracked_bodies
        assert clip["body_pos_w"].dtype == np.float32
    before = (output / "walk.npz").stat().st_mtime_ns
    prepare_csv_dataset(args)
    assert (output / "walk.npz").stat().st_mtime_ns == before

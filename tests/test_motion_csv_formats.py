"""Units, joint order and quaternion conventions for both retarget CSV formats."""

import csv

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    convert_csv,
    csv_format,
    forward_kinematics,
    load_csv,
    resample_motion,
)
from supermimic.robots import get_robot


def bones_csv(path, *, reverse_joints=False):
    n = 7
    euler = np.column_stack((np.arange(n) * 3.0, np.arange(n) * 2.0, 30.0 + np.arange(n)))
    joints = np.arange(n * 29).reshape(n, 29) / 10
    pos_cm = np.column_stack((np.arange(n) * 2.0, np.arange(n) * -1.0, np.full(n, 80.0)))
    names = list(G1_JOINT_NAMES)
    if reverse_joints:
        names.reverse()
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Frame",
                *[f"root_translate{a}" for a in "XYZ"],
                *[f"root_rotate{a}" for a in "XYZ"],
                *[f"{s}_dof" for s in names],
            ]
        )
        writer.writerows([i, *pos_cm[i], *euler[i], *joints[i]] for i in range(n))
    return np.column_stack(
        (
            pos_cm / 100.0,
            Rotation.from_euler("xyz", euler, degrees=True).as_quat(),
            np.deg2rad(joints),
        )
    )


def test_bones_units_equal_headerless_motion_and_export(tmp_path):
    bones = tmp_path / "bones.csv"
    raw = bones_csv(bones)
    lafan = tmp_path / "lafan.csv"
    np.savetxt(lafan, raw, delimiter=",")
    assert csv_format(bones) == "bones" and csv_format(lafan) == "lafan1"
    np.testing.assert_allclose(load_csv(bones), load_csv(lafan), atol=1e-14)
    for source in (bones, lafan):
        convert_csv(source, source.with_suffix(".npz"), 30, 50)
    with np.load(bones.with_suffix(".npz")) as a, np.load(lafan.with_suffix(".npz")) as b:
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            np.testing.assert_allclose(a[key], b[key], atol=1e-6)


def test_bones_joint_column_order_is_not_silently_reinterpreted(tmp_path):
    path = tmp_path / "bad.csv"
    bones_csv(path, reverse_joints=True)
    with pytest.raises(ValueError, match="joint order"):
        load_csv(path)


@pytest.mark.parametrize("robot", ["g1", "casbot02"])
def test_resampling_and_fk_follow_physical_frames(robot):
    model = get_robot(robot).model()
    n = model.nq - 7
    times = np.arange(61) / 30.0
    raw = np.zeros((len(times), 7 + n))
    raw[:, :3] = np.column_stack((times * 0.2, times * -0.1, 0.8 + times * 0.01))
    raw[:, 3:7] = Rotation.from_euler(
        "xyz", np.column_stack((0.05 * times, -0.08 * times, 0.3 * times))
    ).as_quat()
    raw[::2, 3:7] *= -1  # q and -q must not introduce angular-velocity spikes.
    raw[:, 7:] = 0.04 * times[:, None]
    motion = resample_motion(raw, 30, 50)
    t = np.arange(motion.num_frames) / 50
    exact = Slerp(times, Rotation.from_quat(raw[:, 3:7]))(t).as_quat()
    np.testing.assert_allclose(
        np.abs((motion.root_quat[:, [1, 2, 3, 0]] * exact).sum(-1)), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        motion.root_lin_vel_w, np.tile([0.2, -0.1, 0.01], (len(t), 1)), atol=1e-12
    )
    np.testing.assert_allclose(motion.joint_vel, 0.04, atol=1e-12)
    body = forward_kinematics(model, motion)
    # MuJoCo's independent object-velocity query checks every link origin.
    data = mujoco.MjData(model)
    i = 51
    data.qpos[:] = np.r_[motion.root_pos[i], motion.root_quat[i], motion.joint_pos[i]]
    root_rotation = Rotation.from_quat(exact[i])
    data.qvel[:] = np.r_[
        motion.root_lin_vel_w[i],
        root_rotation.inv().apply(motion.root_ang_vel_w[i]),
        motion.joint_vel[i],
    ]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(body.pos_w[i], data.xpos[1:], atol=1e-12)
    for link in range(1, model.nbody):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, link, velocity, 0)
        np.testing.assert_allclose(body.lin_vel_w[i, link - 1], velocity[3:], atol=1e-12)
        np.testing.assert_allclose(body.ang_vel_w[i, link - 1], velocity[:3], atol=1e-12)

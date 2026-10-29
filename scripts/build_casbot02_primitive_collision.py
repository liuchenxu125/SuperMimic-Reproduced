"""Build a G1-style primitive collision MJCF, preserving the source mesh model.

Visual meshes, body frames, joints, explicit inertias and actuators are preserved.
Capsules approximate the anatomical segments; seven sole capsules per foot use
that foot's original mesh footprint bounds and lowest point. This is an alternate
contact model, not a bit-identical replacement for mesh collision.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from supermimic.robots.profiles import CASBOT_ASSETS


def build():
    source = CASBOT_ASSETS / "xml/casbot02.xml"
    target = source.with_name("casbot02_primitives.xml")
    tree = ET.parse(source)
    root = tree.getroot()
    model = mujoco.MjModel.from_xml_path(str(source))
    # The current source uses explicit zero masks for visuals, and inherits
    # nonzero masks on every collision geom. Verify before touching the tree.
    removed = 0
    for body in root.find("worldbody").iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            body.remove(geom)
            removed += 1
    assert removed == sum(bool(model.geom_contype[g] or model.geom_conaffinity[g])
                          for g in range(model.ngeom))
    default = ET.SubElement(root.find("default"), "default", {"class": "primitive_collision"})
    ET.SubElement(default, "geom", {"type": "capsule", "group": "3", "rgba": "0.2 0.6 0.2 0.35",
                                       "contype": "1", "conaffinity": "1", "condim": "1", "density": "0"})
    bodies = {b.get("name"): b for b in root.find("worldbody").iter("body")}
    entries = []

    def add(body, name, radius, endpoints=None, position=None, foot=False):
        attrs = dict(name=name, **{"class": "primitive_collision"}, size=str(radius))
        if endpoints is not None:
            attrs["fromto"] = " ".join(f"{v:.9g}" for v in endpoints)
        else:
            attrs.update(type="sphere", pos=" ".join(map(str, position)))
        if foot:
            attrs.update(condim="3", priority="1", friction="0.9 0.2 0.2")
        ET.SubElement(bodies[body], "geom", attrs)
        entries.append({"body": body, **attrs})

    add("base_link", "pelvis_collision", .07, [0, 0, .025, 0, 0, .105])
    add("waist_yaw_link", "torso_collision", .11, [.015, 0, .055, .015, 0, .255])
    add("head_pitch_link", "head_collision", .095, position=[.02, 0, .075])
    foot_bounds = {}
    for side in ("left", "right"):
        add(f"{side}_leg_pelvic_yaw_link", f"{side}_thigh_collision", .06,
            [-.005, 0, -.045, -.02, 0, -.28])
        add(f"{side}_leg_knee_pitch_link", f"{side}_shin_collision", .045,
            [.005, 0, -.045, .005, 0, -.395])
        add(f"{side}_shoulder_yaw_link", f"{side}_upper_arm_collision", .04,
            [0, 0, .005, 0, 0, -.065])
        add(f"{side}_elbow_pitch_link", f"{side}_elbow_collision", .035,
            [0, 0, .005, 0, 0, -.055])
        add(f"{side}_wrist_yaw_link", f"{side}_forearm_collision", .035,
            [0, 0, -.02, 0, 0, -.135])
        add(f"{side}_wrist_roll_link", f"{side}_hand_collision", .03,
            [0, 0, -.005, 0, 0, -.135])
        geom = model.geom(f"{side}_foot_collision")
        mesh = int(model.geom_dataid[geom.id])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, geom.quat)
        points = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + geom.pos
        lo, hi = points.min(0), points.max(0)
        foot_bounds[side] = {"min": lo.tolist(), "max": hi.tolist()}
        radius = .01
        mid_y = (lo[1] + hi[1]) / 2
        half_y = (hi[1] - lo[1]) / 2
        for i, y in enumerate(np.linspace(lo[1] + radius, hi[1] - radius, 7), 1):
            edge = abs((y - mid_y) / half_y)**2
            x0, x1 = lo[0] + radius + .025 * edge, hi[0] - radius - .025 * edge
            z = lo[2] + radius
            add(f"{side}_leg_ankle_roll_link", f"{side}_foot{i}_collision", radius,
                [x0, y, z, x1, y, z], foot=True)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    exclusions = [
        ("base_link", "left_leg_pelvic_yaw_link"),
        ("base_link", "right_leg_pelvic_yaw_link"),
        ("left_elbow_pitch_link", "left_wrist_roll_link"),
        ("right_elbow_pitch_link", "right_wrist_roll_link"),
        ("left_wrist_yaw_link", "left_wrist_roll_link"),
        ("right_wrist_yaw_link", "right_wrist_roll_link"),
    ]
    for a, b in exclusions:
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    ET.indent(tree, space="  ")
    tree.write(target, encoding="unicode")
    with target.open("a") as f:
        f.write("\n")
    candidate = mujoco.MjModel.from_xml_path(str(target))
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                  "jnt_pos", "jnt_axis", "jnt_range", "dof_armature", "dof_damping", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(model, field), getattr(candidate, field), err_msg=field)
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "removed_collision_meshes": removed, "primitive_geoms": entries,
                  "foot_mesh_bounds": foot_bounds, "exclusions": exclusions,
                  "source": "Hand-sized anatomical capsules following mjlab G1 collision structure; sole dimensions measured from Casbot02 source meshes."}
    (CASBOT_ASSETS / "primitive_collision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{target}: {removed} collision meshes -> {len(entries)} primitives; exact inertial/joint parity")


if __name__ == "__main__":
    build()

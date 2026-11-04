"""Build a G1-style primitive collision MJCF, preserving the source mesh model.

Visual meshes, body frames, joints, explicit inertias and actuators are preserved.
Capsules approximate the anatomical segments, including the pelvic-pitch link.
Each foot keeps the source mesh collision from casbot02.xml. This is an alternate
contact model, not a bit-identical replacement for full-body mesh collision.
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
    preserved_feet = []
    for body in root.find("worldbody").iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            if geom.get("name", "").endswith("_foot_collision"):
                preserved_feet.append((body.get("name"), dict(geom.attrib)))
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

    def visual_points(body_name, *, single=True):
        body_id = model.body(body_name).id
        geoms = [g for g in range(model.ngeom)
                 if model.geom_bodyid[g] == body_id and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and not (model.geom_contype[g] or model.geom_conaffinity[g])]
        if single and len(geoms) != 1:
            raise ValueError(f"{body_name}: expected one visual mesh")
        if not geoms:
            raise ValueError(f"{body_name}: expected a visual mesh")
        clouds = []
        for geom in geoms:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            clouds.append(model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + model.geom_pos[geom])
        return np.concatenate(clouds)

    def cover_radius(points, endpoints):
        """Smallest millimetre radius whose cylinder covers the link along this segment."""
        start = np.asarray(endpoints[:3], float)
        end = np.asarray(endpoints[3:], float)
        direction = end - start
        scale = float(direction @ direction)
        along = ((points - start) @ direction) / scale
        inside = (along >= 0.0) & (along <= 1.0)
        chosen = points[inside] if np.any(inside) else points
        axis = direction / np.sqrt(scale)
        offset = chosen - start
        radial = np.linalg.norm(offset - (offset @ axis)[:, None] * axis, axis=1)
        return float(np.ceil((float(radial.max()) + 0.001) * 1000 - 1e-6) / 1000)

    def hip_segment(points):
        center = points.mean(0)
        axis = np.linalg.svd(points - center, full_matrices=False)[2][0]
        projection = (points - center) @ axis
        start = center + axis * np.percentile(projection, 2)
        end = center + axis * np.percentile(projection, 98)
        direction = end - start
        scale = float(direction @ direction)
        along = np.clip(((points - start) @ direction) / scale, 0, 1)
        distance = np.linalg.norm(points - (start + along[:, None] * direction), axis=1)
        # Cover the outer shell, not only the 99th percentile of the mesh.
        radius = float(np.ceil((np.percentile(distance, 99) + 0.001) * 1000) / 1000)
        # The fitted outer end reaches the hanging arm at the zero pose. Pull
        # that end medially; the radius still covers the hip shell.
        start, end = np.array(start, float), np.array(end, float)
        lateral = start if abs(start[1]) >= abs(end[1]) else end
        medial = end if lateral is start else start
        direction = lateral - medial
        length = float(np.linalg.norm(direction))
        lateral -= direction / length * min(0.05, length * 0.45)
        return np.concatenate((medial, lateral)), radius

    add("base_link", "pelvis_collision", .07, [0, 0, .025, 0, 0, .105])
    add("waist_yaw_link", "torso_collision", .11, [.015, 0, .055, .015, 0, .255])
    add("head_pitch_link", "head_collision", .095, position=[.02, 0, .075])
    foot_bounds = {}
    for side in ("left", "right"):
        hip_points = visual_points(f"{side}_leg_pelvic_pitch_link")
        segment, _old_radius = hip_segment(hip_points)
        add(f"{side}_leg_pelvic_pitch_link", f"{side}_hip_collision",
            cover_radius(hip_points, segment), segment)
        thigh = [-.005, 0, -.045, -.02, 0, -.28]
        add(f"{side}_leg_pelvic_yaw_link", f"{side}_thigh_collision",
            round(cover_radius(visual_points(f"{side}_leg_pelvic_yaw_link", single=False), thigh) - 0.005, 3), thigh)
        shin = np.array([.005, 0, -.045, .005, 0, -.395])
        shin_mid = (shin[:3] + shin[3:]) / 2
        shin_upper = np.concatenate((shin[:3], shin_mid))
        shin_lower = np.concatenate((shin_mid, shin[3:]))
        shin_radius = round(cover_radius(visual_points(f"{side}_leg_knee_pitch_link", single=False), shin) - 0.017, 3)
        add(f"{side}_leg_knee_pitch_link", f"{side}_shin_upper_collision", shin_radius, shin_upper)
        add(f"{side}_leg_knee_pitch_link", f"{side}_shin_lower_collision",
            round(shin_radius - 0.015, 3), shin_lower)
        upper_arm = [0, 0, .005, 0, 0, -.065]
        add(f"{side}_shoulder_yaw_link", f"{side}_upper_arm_collision",
            cover_radius(visual_points(f"{side}_shoulder_yaw_link", single=False), upper_arm), upper_arm)
        elbow = [0, 0, .005, 0, 0, -.055]
        add(f"{side}_elbow_pitch_link", f"{side}_elbow_collision",
            round(cover_radius(visual_points(f"{side}_elbow_pitch_link", single=False), elbow) - 0.014, 3), elbow)
        forearm = [0, 0, -.02, 0, 0, -.135]
        add(f"{side}_wrist_yaw_link", f"{side}_forearm_collision",
            round(cover_radius(visual_points(f"{side}_wrist_yaw_link", single=False), forearm) - 0.015, 3), forearm)
        hand = [0, 0, -.005, 0, 0, -.135]
        add(f"{side}_wrist_roll_link", f"{side}_hand_collision",
            round(cover_radius(visual_points(f"{side}_wrist_roll_link", single=False), hand) - 0.013, 3), hand)
        points = visual_points(f"{side}_leg_ankle_roll_link")
        lo, hi = points.min(0), points.max(0)
        foot_bounds[side] = {"min": lo.tolist(), "max": hi.tolist()}
    if [name for _, attrs in preserved_feet for name in (attrs["name"],)] != [
            "left_foot_collision", "right_foot_collision"]:
        raise ValueError(f"Expected the source foot meshes, got {preserved_feet}")
    for body_name, attrs in preserved_feet:
        ET.SubElement(bodies[body_name], "geom", attrs)
        entries.append({"body": body_name, **attrs})
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    exclusions = [
        ("base_link", "left_leg_pelvic_yaw_link"),
        ("base_link", "right_leg_pelvic_yaw_link"),
        ("left_leg_pelvic_pitch_link", "left_leg_pelvic_yaw_link"),
        ("right_leg_pelvic_pitch_link", "right_leg_pelvic_yaw_link"),
        ("left_leg_pelvic_pitch_link", "right_leg_pelvic_pitch_link"),
        ("left_leg_pelvic_pitch_link", "waist_yaw_link"),
        ("right_leg_pelvic_pitch_link", "waist_yaw_link"),
        ("left_leg_knee_pitch_link", "left_leg_ankle_roll_link"),
        ("right_leg_knee_pitch_link", "right_leg_ankle_roll_link"),
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
                  "source": "Anatomical capsules plus the source Casbot02 foot meshes. Hip capsules cover the pelvic-pitch visual mesh."}
    (CASBOT_ASSETS / "primitive_collision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{target}: {removed} collision meshes -> {len(entries)} primitives; exact inertial/joint parity")


if __name__ == "__main__":
    build()

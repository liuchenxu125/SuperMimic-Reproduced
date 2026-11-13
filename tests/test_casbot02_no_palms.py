"""Removing fixed palms must preserve wrist joints and cover their real shells."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from supermimic.robots import get_robot
from supermimic.robots.profiles import CASBOT_ASSETS


@pytest.mark.parametrize("variant", ["mesh", "primitive"])
def test_palms_removed_but_wrist_articulation_and_inertia_retained(variant):
    model = get_robot("casbot02").model(collision_model=variant)
    assert (model.nq, model.nv, model.nbody) == (34, 33, 31)
    meshes = {model.mesh(i).name for i in range(model.nmesh)}
    assert not meshes & {"left_base_link", "right_base_link"}
    for side, mass in [("left", .11076), ("right", .10176)]:
        assert side + "_wrist_roll_link" in meshes
        assert model.body(side + "_wrist_roll_link").mass[0] == mass
        for number in (5, 6, 7):
            assert model.joint(f"upper_{side}_{number}_joint").id > 0


def test_wrist_capsule_contains_bare_wrist_mesh_and_stops_before_old_palm():
    model = get_robot("casbot02").model(collision_model="primitive")
    xml = ET.parse(CASBOT_ASSETS / "xml/casbot02_primitives.xml")
    for side in ["left", "right"]:
        name = side + "_wrist_roll_link"
        body = model.body(name).id
        visuals = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                   and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        assert len(visuals) == 1
        g = visuals[0]
        k = int(model.geom_dataid[g])
        vertices = model.mesh_vert[model.mesh_vertadr[k]:model.mesh_vertadr[k] + model.mesh_vertnum[k]]
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, model.geom_quat[g])
        points = vertices @ matrix.reshape(3, 3).T + model.geom_pos[g]
        element = next(g for g in xml.getroot().findall(".//geom") if g.get("name") == side + "_wrist_collision")
        segment = np.fromstring(element.get("fromto"), sep=" ").reshape(2, 3)
        radius = float(element.get("size"))
        direction = segment[1] - segment[0]
        t = np.clip((points - segment[0]) @ direction / (direction @ direction), 0, 1)
        distances = np.linalg.norm(points - (segment[0] + t[:, None] * direction), axis=1)
        assert distances.max() <= radius - .001 + 2e-8
        assert segment[:, 2].min() - radius > -.09  # Old palm capsule reached -.172 m.
        assert np.linalg.norm(direction) + 2 * radius < .15
        assert not any(g.get("name") == side + "_hand_collision" for g in xml.getroot().findall(".//geom"))

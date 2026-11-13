"""Exercise geometry gradients, dense avoidance and refusal to alter fixed samples."""
import mujoco
import numpy as np
import pytest

from supermimic.data.collision_repair import ExteriorMeshChecker, dense_poses, repair_joints


@pytest.fixture
def checker():
    vertices = " ".join(str(v) for x in (-.06, .06) for y in (-.06, .06)
                        for z in (-.06, .06) for v in (x, y, z))
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><asset><mesh name="cube" vertex="{vertices}"/></asset>
    <worldbody><body name="base"><freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body name="left" pos="-.1 0 0">
        <joint name="left_slide" type="slide" axis="1 0 0" range="-.2 .2"/>
        <geom type="mesh" mesh="cube" group="1" contype="0" conaffinity="0"/>
      </body><body name="right" pos=".1 0 0">
        <joint name="right_slide" type="slide" axis="0 1 0" range="-.2 .2"/>
        <geom type="mesh" mesh="cube" group="1" contype="0" conaffinity="0"/>
      </body>
    </body></worldbody></mujoco>""")
    return ExteriorMeshChecker(model, skip_graph_distance=0)


def test_signed_distance_gradient_matches_finite_difference(checker):
    q = np.array([.1, .01])
    value, jac = checker.distances(q, [0], .002)
    numerical = np.empty(2)
    for j in range(2):
        changed = q.copy()
        changed[j] += 1e-5
        after, _ = checker.distances(changed, [0], .002)
        numerical[j] = (after[0] - value[0]) / 1e-5
    np.testing.assert_allclose(jac[0], numerical, atol=2e-4, rtol=2e-4)


def test_repair_clears_dense_path_and_preserves_source_endpoints(checker):
    q = np.array([[0., 0.], [.1, 0.], [0., 0.]])
    assert checker.scan(q, 4)
    fixed, report = repair_joints(checker, q, 50, 25, substeps=4, max_rounds=2,
                                  progress=lambda *args, **kwargs: None)
    assert not checker.scan(fixed.astype(float), 4)
    np.testing.assert_array_equal(fixed[[0, 2]], q[[0, 2]])
    assert report["final_dense_collision_frames"] == 0
    assert report["max_correction_rad"] < .15
    assert fixed.dtype == np.float32


def test_fixed_colliding_source_frame_is_not_silently_modified(checker):
    q = np.array([[0., 0.], [.1, 0.], [0., 0.]])
    with pytest.raises(RuntimeError, match="fixed source"):
        repair_joints(checker, q, 50, 50, progress=lambda *args, **kwargs: None)


def test_local_refinement_uses_global_source_sample_phase(checker):
    q = np.array([[0., 0.], [.1, 0.], [0., 0.]])
    # A cropped window starts one frame into the clip. Its middle frame is now
    # an original 25 Hz source knot and must not be treated as adjustable.
    with pytest.raises(RuntimeError, match="fixed source"):
        repair_joints(checker, q, 50, 25, frame_offset=1, progress=lambda *args, **kwargs: None)


def test_preexisting_limit_violation_requires_explicit_opt_in(checker):
    q = np.array([[-.21, 0.], [0., 0.], [0., 0.]])
    with pytest.raises(ValueError, match="clamp-joints"):
        repair_joints(checker, q, 50, 25)


def test_dense_samples_include_both_endpoints():
    q = np.array([[0., 2.], [4., 6.]])
    sampled = dense_poses(q, 4)
    np.testing.assert_allclose(sampled, [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]])

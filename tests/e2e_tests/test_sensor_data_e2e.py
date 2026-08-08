"""The sensor modalities of both views hold plausible data.

The normal view is the dataset's ground truth rig; the perturbated view is the
same tick captured from a laterally and rotationally offset rig. They must be
recognisably the same scene — same tick, same calibration, same statistics —
and unmistakably different images, because a view that silently fell back to
the normal rig would train the policy on an augmentation that is not there.
"""

import numpy as np
import pytest
from py123d.api.scene.scene_filter import SceneFilter

from lead.log_reader import SceneLoader
from tests.e2e_tests.conftest import sample_indices

# Both views decode full images per sample, so keep the pass small.
MAX_FRAMES = 6
# CARLA's lidar reaches this far; a point beyond it is a frame or unit error.
MAX_LIDAR_RANGE_M = 200.0
# The rig offsets the collection perturbs by.
MAX_LATERAL_OFFSET_M = 2.0
MAX_YAW_OFFSET_DEG = 20.0
# Share of camera images that may be a flat colour. A camera pointed at an
# unlit wall at night renders black legitimately; a broken rig renders black
# everywhere, and one camera of a six-camera rig is already a sixth.
MAX_FLAT_IMAGE_FRACTION = 0.1


def _loader(data_root: str, perturbation_probability: float) -> SceneLoader:
    """A loader over one sensor view, with the cheap modalities only.

    Args:
        data_root: The dataset root.
        perturbation_probability: 0.0 for the normal view, 1.0 for the
            perturbated one.

    Returns:
        The loader.
    """
    return SceneLoader(
        data_root,
        SceneFilter(shuffle=False, history_num_iterations=0, future_num_iterations=0),
        perturbation_probability=perturbation_probability,
        read_radar_sweeps=False,
        read_map_api=False,
    )


@pytest.fixture(scope="module")
def normal(data_root: str) -> SceneLoader:
    """The normal-view loader.

    Args:
        data_root: The dataset root.

    Returns:
        The loader.
    """
    loader = _loader(data_root, 0.0)
    assert len(loader) > 0, f"No scenes under {data_root}"
    return loader


@pytest.fixture(scope="module")
def perturbated(data_root: str) -> SceneLoader:
    """The perturbated-view loader, or skip when the split is absent.

    Args:
        data_root: The dataset root.

    Returns:
        The loader.
    """
    loader = _loader(data_root, 1.0)
    if not loader.any_perturbated_sensor_views:
        pytest.skip("Dataset has no perturbated split")
    return loader


@pytest.mark.e2e
def test_normal_view_cameras_are_decoded_images(normal) -> None:
    """Every camera yields a full uint8 image with its calibration.

    A camera that failed to render comes back a constant colour, but so does one
    pointed at an unlit wall at night, so the flat ones are counted rather than
    forbidden: a broken rig shows up as a large share of them, one dark frame
    does not.
    """
    flat: list[str] = []
    total = 0
    for index in sample_indices(len(normal), MAX_FRAMES):
        frame = normal[index]
        assert frame.cameras, f"frame {index} has no cameras"
        for camera in frame.cameras:
            image = camera.image
            assert image.dtype == np.uint8
            assert image.ndim == 3 and image.shape[2] == 3
            assert image.shape[0] > 1 and image.shape[1] > 1
            assert np.isfinite(camera.camera_to_global_se3.array).all()
            total += 1
            if image.std() <= 1.0:
                flat.append(f"frame {index} {camera.metadata.camera_id}")

    assert total, "no cameras were inspected"
    assert len(flat) <= MAX_FLAT_IMAGE_FRACTION * total, (
        f"{len(flat)}/{total} camera images are flat, which is more than one "
        f"dark scene explains: {flat[:10]}"
    )


@pytest.mark.e2e
def test_normal_view_lidar_is_a_plausible_sweep(normal) -> None:
    """The sweep is finite, in range, and spread over all three axes."""
    for index in sample_indices(len(normal), MAX_FRAMES):
        frame = normal[index]
        assert frame.lidar_sweeps is not None
        points = frame.lidar_sweeps[0].point_cloud_3d
        assert len(points) > 0, f"frame {index} has an empty sweep"
        coordinates = np.asarray(points)[:, :3]
        assert np.isfinite(coordinates).all()
        assert np.linalg.norm(coordinates[:, :2], axis=1).max() < MAX_LIDAR_RANGE_M
        # A sweep collapsed onto a plane means the frame or the decode is wrong.
        assert (coordinates.std(axis=0) > 0.1).all(), (
            f"frame {index}: the sweep is degenerate on some axis"
        )


@pytest.mark.e2e
def test_normal_view_carries_the_derived_modalities(normal) -> None:
    """Semantic, instance and depth cameras line up with the RGB cameras."""
    frame = normal[0]
    for name in ("semantic_cameras", "instance_cameras", "depth_cameras"):
        cameras = getattr(frame, name)
        if cameras is None:
            continue
        assert len(cameras) == len(frame.cameras), f"{name}: camera count differs"
        for derived, rgb in zip(cameras, frame.cameras, strict=False):
            assert derived.timestamp.time_us == rgb.timestamp.time_us
            assert derived.image.shape[:2] == rgb.image.shape[:2]

    if frame.semantic_cameras is not None:
        # Semantic pixels are raw CARLA classes, so a small integer set.
        classes = np.unique(frame.semantic_cameras[0].image)
        assert len(classes) > 1, "the semantic camera holds a single class"
        assert classes.max() < 256


@pytest.mark.e2e
def test_the_perturbated_view_is_the_same_tick_from_a_shifted_rig(
    normal,
    perturbated,
) -> None:
    """Same scene, same calibration, different pixels — and a real rig offset."""
    assert len(normal) == len(perturbated)
    paired = 0
    for index in sample_indices(len(perturbated), MAX_FRAMES):
        perturbated_frame = perturbated[index]
        if perturbated_frame.rig_perturbation is None:
            continue
        paired += 1
        normal_frame = normal[index]
        assert normal_frame.rig_perturbation is None

        offset = perturbated_frame.rig_perturbation
        assert 0.0 < abs(offset.lateral_translation_m) <= MAX_LATERAL_OFFSET_M
        assert abs(offset.yaw_rotation_deg) <= MAX_YAW_OFFSET_DEG

        assert len(perturbated_frame.cameras) == len(normal_frame.cameras)
        for shifted, straight in zip(
            perturbated_frame.cameras,
            normal_frame.cameras,
            strict=False,
        ):
            # Same tick and same camera: only the rig pose may differ.
            assert shifted.timestamp.time_us == straight.timestamp.time_us
            assert shifted.metadata.camera_id == straight.metadata.camera_id
            assert shifted.image.shape == straight.image.shape
            assert shifted.image.dtype == straight.image.dtype
            assert not np.array_equal(shifted.image, straight.image), (
                f"frame {index}: the perturbated camera rendered the normal view"
            )
            # The same scene from a shifted rig: similar overall brightness.
            assert (
                abs(float(shifted.image.mean()) - float(straight.image.mean())) < 40.0
            )
    assert paired > 0, "perturbated split present but no frame paired"


@pytest.mark.e2e
def test_the_two_views_share_the_privileged_stream(normal, perturbated) -> None:
    """The world is one; only the rig moved, so the log-only fields agree."""
    for index in sample_indices(len(perturbated), MAX_FRAMES):
        perturbated_frame = perturbated[index]
        if perturbated_frame.rig_perturbation is None:
            continue
        normal_frame = normal[index]

        assert perturbated_frame.ego_state is not None
        assert normal_frame.ego_state is not None
        # Ego state and detections are global-frame, so the rig cannot change them.
        assert (
            perturbated_frame.ego_state.timestamp.time_us
            == normal_frame.ego_state.timestamp.time_us
        )
        assert perturbated_frame.driving_meta is not None
        np.testing.assert_allclose(
            np.asarray(perturbated_frame.driving_meta["route"], dtype=np.float64),
            np.asarray(normal_frame.driving_meta["route"], dtype=np.float64),
        )

        # Navigation is ego-frame, so it must move with the rig, not stay put.
        assert perturbated_frame.target_point is not None
        assert normal_frame.target_point is not None
        assert not np.allclose(
            perturbated_frame.target_point,
            normal_frame.target_point,
        )
        # Moving the rig by at most a couple of metres cannot move it far.
        assert (
            np.linalg.norm(
                perturbated_frame.target_point - normal_frame.target_point,
            )
            < MAX_LIDAR_RANGE_M
        )


@pytest.mark.e2e
def test_the_perturbated_lidar_sees_the_same_world(normal, perturbated) -> None:
    """The sweep is re-expressed, not re-simulated: same size world, same tick."""
    for index in sample_indices(len(perturbated), MAX_FRAMES):
        perturbated_frame = perturbated[index]
        if perturbated_frame.rig_perturbation is None:
            continue
        normal_frame = normal[index]
        assert perturbated_frame.lidar_sweeps is not None
        assert normal_frame.lidar_sweeps is not None

        shifted = perturbated_frame.lidar_sweeps[0]
        straight = normal_frame.lidar_sweeps[0]
        assert shifted.timestamp.time_us == straight.timestamp.time_us
        shifted_points = np.asarray(shifted.point_cloud_3d)[:, :3]
        straight_points = np.asarray(straight.point_cloud_3d)[:, :3]
        assert np.isfinite(shifted_points).all()
        # Same scene, so the swept extent agrees to about the rig offset.
        for axis in range(2):
            assert (
                abs(shifted_points[:, axis].max() - straight_points[:, axis].max())
                < MAX_LIDAR_RANGE_M
            )

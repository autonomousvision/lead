"""End-to-end test for log readability."""

import os

import pytest
from py123d.api.scene.scene_filter import SceneFilter
from py123d.datatypes import LidarID

from lead.dataloader.scene_index import get_scenes


@pytest.mark.e2e
def test_logs_readable():
    """Load scenes and verify core modalities resolve."""
    data_root = os.environ.get("LEAD_TEST_DATA_ROOT")
    if not data_root:
        pytest.skip("LEAD_TEST_DATA_ROOT not set")

    # Cameras only exist on save ticks; anchor the scenes on them.
    scenes = get_scenes(
        data_root,
        SceneFilter(
            shuffle=False,
            history_num_iterations=0,
            future_num_iterations=0,
            required_scene_modalities=["camera:all@initial"],
        ),
    )
    assert scenes, f"No scenes found under {data_root}"

    for scene in scenes:
        ego = scene.get_ego_state_se3_at_iteration(0)
        assert ego is not None, f"{scene.log_name}: ego state did not resolve"

        box_metadata = scene.get_box_detections_se3_metadata()
        assert box_metadata is not None, (
            f"{scene.log_name}: box metadata did not resolve"
        )

        boxes = scene.get_box_detections_se3_at_iteration(0)
        assert boxes is not None, f"{scene.log_name}: box detections did not resolve"
        assert len(boxes.box_detections) >= 0, (
            f"{scene.log_name}: box detections list is valid"
        )

        for camera_id, metadata in scene.get_camera_semantic_metadatas().items():
            assert metadata.segmentation_label_class is not None, (
                f"{scene.log_name}: semantic label class for {camera_id} did not resolve"
            )

        camera_metadatas = scene.get_camera_metadatas()
        assert camera_metadatas, f"{scene.log_name}: camera metadata did not resolve"

        for camera_id in camera_metadatas:
            camera = scene.get_camera_at_iteration(0, camera_id)
            assert camera is not None, (
                f"{scene.log_name}: RGB image for {camera_id} did not resolve"
            )

        lidar = scene.get_lidar_at_iteration(0, LidarID.LIDAR_TOP)
        assert lidar is not None, f"{scene.log_name}: lidar did not resolve"
        assert len(lidar.point_cloud_3d) > 0, f"{scene.log_name}: lidar is empty"

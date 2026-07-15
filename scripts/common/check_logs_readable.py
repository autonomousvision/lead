#!/usr/bin/env python
"""Read collected py123d logs back and check the core modalities resolve."""

import sys
from pathlib import Path

from py123d.api.scene.scene_filter import SceneFilter

from lead.dataloader.scene_index import get_scenes


def main() -> None:
    """Load every scene under the given data root and check the core modalities resolve."""
    data_root = Path(sys.argv[1])
    scenes = get_scenes(data_root, SceneFilter(shuffle=False))
    assert scenes, f"No scenes found under {data_root}."

    for scene in scenes:
        ego = scene.get_ego_state_se3_at_iteration(0)
        assert ego is not None, f"{scene.log_name}: ego state did not resolve."

        box_metadata = scene.get_box_detections_se3_metadata()
        assert box_metadata is not None, (
            f"{scene.log_name}: box detection metadata did not resolve."
        )
        boxes = scene.get_box_detections_se3_at_iteration(0)
        assert boxes is not None, (
            f"{scene.log_name}: box detections did not resolve at iteration 0."
        )

        for camera_id, metadata in scene.get_camera_semantic_metadatas().items():
            assert metadata.segmentation_label_class is not None, (
                f"{scene.log_name}: semantic label class for {camera_id} did not resolve."
            )

        print(
            f"{scene.log_name}: ego, {len(boxes.box_detections)} box detections, camera semantics OK",
        )

    print(f"{len(scenes)} scene(s) readable.")


if __name__ == "__main__":
    main()

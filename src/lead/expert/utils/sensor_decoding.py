"""Decode CARLA's packed depth and instance-segmentation camera streams."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.constants import CarlaSemanticSegmentationClass


def convert_depth(data: jt.UInt8[npt.NDArray, "h w 3"]) -> jt.Float[npt.NDArray, "h w"]:
    """Compute normalized depth from a CARLA depth map.

    Converts CARLA's RGB-encoded depth values to actual depth in meters.

    Args:
        data: CARLA depth map as RGB array of shape (height, width, 3).

    Returns:
        Depth values in meters as array of shape (height, width).
    """
    # Accumulate per channel to avoid an astype(float32) copy of the whole 3D array.

    scale = 1000.0 / 16777215.0  # 1000 / (256**3 - 1)

    # R * 65536 * scale + G * 256 * scale + B * scale
    # Initialize with the most significant byte (R)
    depth = data[:, :, 0].astype(np.float32)
    depth *= 65536.0 * scale

    # Add Middle byte (G)
    depth += data[:, :, 1].astype(np.float32) * (256.0 * scale)

    # Add Least significant byte (B)
    depth += data[:, :, 2].astype(np.float32) * scale

    return depth


def convert_instance_segmentation(
    data: jt.UInt8[npt.NDArray, "H W 3"],
) -> jt.Int32[npt.NDArray, "H W 2"]:
    """
    Args:
        data: Instance segmentation map from CARLA of shape (H, W, 3) with values in range [0, 255].
              R channel = semantic ID
              G+B*256 = instance ID
    Returns:
        data: Instance segmentation of shape (H, W, 2) with channels: [semantic_id, instance_id]
    """
    semantic_id = data[:, :, 0]
    semantic_id[semantic_id >= len(CarlaSemanticSegmentationClass)] = (
        0  # Carla semantic segmentation has some bug which give invalid semantic class.
    )
    instance_id = data[:, :, 1].astype(np.int32) + (data[:, :, 2].astype(np.int32) << 8)
    return np.stack([semantic_id, instance_id], axis=-1)

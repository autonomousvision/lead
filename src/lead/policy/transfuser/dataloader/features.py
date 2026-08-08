"""TransFuser featurization: one scene of raw driving data into the model inputs."""

import typing
from collections.abc import Mapping

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.api.point_cloud_transforms import (
    lidar_sweep_to_carla_ego_frame,
    radar_returns_to_carla_ego_frame,
    split_radar_by_sensor,
)
from lead.common.sensors import ransac
from lead.config import LeadConfig
from lead.log_reader import SceneData, view_geometry
from lead.log_reader.carla_decoding import carla_forward_speed
from lead.policy.transfuser.dataloader.point_cloud import accumulate_lidar_points
from lead.policy.transfuser.dataloader.sample import TransfuserOutputs


def build_camera_features(
    scene_data: SceneData,
    lead_config: LeadConfig,
) -> TransfuserOutputs:
    """The camera and navigation model inputs of one scene.

    Args:
        scene_data: One tick of raw driving data in the ego view frame.
        lead_config: Root config tree.

    Returns:
        The rgb tensor, the target points, the ego speed, and the town.
    """
    data: TransfuserOutputs = {}
    transfuser_config = lead_config.policy.transfuser

    # RGB: concatenate the per-camera images side by side, left to right.
    stitched_camera_image = np.concatenate(
        [camera.image for camera in scene_data.cameras],
        axis=1,
    )
    data["rgb"] = np.transpose(stitched_camera_image, (2, 0, 1))

    for key in ("previous_target_point", "target_point", "next_target_point"):
        point = getattr(scene_data, key)
        if point is not None:
            data[key] = point
    if transfuser_config.use_velocity:
        assert scene_data.ego_state is not None
        data["speed"] = carla_forward_speed(scene_data.ego_state)
    assert scene_data.log_metadata is not None
    assert scene_data.log_metadata.location is not None
    data["town"] = scene_data.log_metadata.location
    return data


def build_lidar_raster(
    scene_data: SceneData,
    lead_config: LeadConfig,
) -> TransfuserOutputs:
    """The BEV lidar raster of one scene's sweep window.

    The 123D sweeps are decoded into the CARLA ego frame and accumulated into
    the anchor tick.

    Args:
        scene_data: One tick of raw driving data in the ego view frame.
        lead_config: Root config tree.

    Returns:
        The rasterized lidar tensor, or an empty dict without lidar sweeps.
    """
    if scene_data.lidar_sweeps is None:
        return {}
    assert scene_data.past_ego_positions is not None
    assert scene_data.past_ego_yaws is not None
    lidar_points = accumulate_lidar_points(
        {
            age: lidar_sweep_to_carla_ego_frame(lidar)
            for age, lidar in scene_data.lidar_sweeps.items()
        },
        (
            None
            if scene_data.radar_sweeps is None
            else {
                age: radar_returns_to_carla_ego_frame(radar)[:, :3].astype(np.float64)
                for age, radar in scene_data.radar_sweeps.items()
            }
        ),
        scene_data.past_ego_positions,
        scene_data.past_ego_yaws,
        lead_config,
    )
    if scene_data.rig_perturbation is not None:
        lidar_points = view_geometry.to_view_frame_point_cloud(
            lidar_points,
            scene_data.rig_perturbation.lateral_translation_m,
            scene_data.rig_perturbation.yaw_rotation_deg,
        )
    return {
        "rasterized_lidar": rasterize_lidar_bev(
            lead_config=lead_config,
            lidar_points_ego_frame=lidar_points,
        )[None],
    }


def build_radar_features(
    scene_data: SceneData,
    lead_config: LeadConfig,
) -> dict[str, jt.Float[npt.NDArray, "_ 5"]]:
    """The radar model inputs of one scene's anchor tick.

    The merged returns are split per sensor, filtered and padded to a fixed size.

    Args:
        scene_data: One tick of raw driving data in the ego view frame.
        lead_config: Root config tree.

    Returns:
        The per-sensor and concatenated radar tensors, or an empty dict
        without radar sweeps.
    """
    if scene_data.radar_sweeps is None:
        return {}
    data: dict[str, jt.Float[npt.NDArray, "_ 5"]] = {}
    radar_points_per_sensor = _preprocess_radar_input(
        lead_config,
        {
            f"radar{i + 1}": radar
            for i, radar in enumerate(split_radar_by_sensor(scene_data.radar_sweeps[0]))
        },
    )
    for i, radar_points in enumerate(radar_points_per_sensor):
        data[f"radar{i + 1}"] = radar_points
    data["radar"] = np.concatenate(radar_points_per_sensor, axis=0)
    return data


def build_features(
    scene_data: SceneData,
    lead_config: LeadConfig,
) -> Mapping[str, typing.Any]:
    """Turn one scene into every model input, as the inference path consumes them.

    A pure function of the scene data, so training and inference share one
    featurization.

    Args:
        scene_data: One tick of raw driving data in the ego view frame.
        lead_config: Root config tree.

    Returns:
        The model inputs of one unbatched sample.
    """
    data: dict[str, typing.Any] = dict(build_camera_features(scene_data, lead_config))
    data.update(build_lidar_raster(scene_data, lead_config))
    data.update(build_radar_features(scene_data, lead_config))
    return data


def rasterize_lidar_bev(
    lead_config: LeadConfig,
    lidar_points_ego_frame: jt.Float[npt.NDArray, "n_points 3"],
    remove_ground_plane: bool = False,
) -> jt.Float32[npt.NDArray, "bev_h bev_w"]:
    """Rasterize an ego-frame point cloud into the BEV density grid.

    Args:
        lead_config: Root config tree.
        lidar_points_ego_frame: Point cloud in the CARLA ego frame.
        remove_ground_plane: Whether to drop ground points before splatting.

    Returns:
        The normalized BEV point-density grid.
    """
    transfuser_config = lead_config.policy.transfuser

    def bev_point_density(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(
            transfuser_config.bev_min_x_meter,
            transfuser_config.bev_max_x_meter,
            (transfuser_config.bev_max_x_meter - transfuser_config.bev_min_x_meter)
            * int(transfuser_config.bev_pixels_per_meter)
            + 1,
        )
        ybins = np.linspace(
            transfuser_config.bev_min_y_meter,
            transfuser_config.bev_max_y_meter,
            (transfuser_config.bev_max_y_meter - transfuser_config.bev_min_y_meter)
            * int(transfuser_config.bev_pixels_per_meter)
            + 1,
        )
        points_per_cell = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        max_points = transfuser_config.max_lidar_points_per_bev_pixel
        points_per_cell[points_per_cell > max_points] = max_points
        normalized_point_density = points_per_cell / max_points
        # Transpose swaps axes: carla is x front, y right, whereas the image is
        # y front, x right (x height channel, y width channel).
        return normalized_point_density.T

    # Remove points above the vehicle
    heights = lidar_points_ego_frame[..., 2]
    lidar_points_ego_frame = lidar_points_ego_frame[
        (heights <= transfuser_config.lidar_max_height_meter)
        & (transfuser_config.lidar_min_height_meter <= heights)
    ]
    if remove_ground_plane:
        is_ground_mask = ransac.remove_ground(
            lidar_points_ego_frame,
            lead_config.expert,
        )
        lidar_points_ego_frame = lidar_points_ego_frame[~is_ground_mask]
    return bev_point_density(lidar_points_ego_frame).astype(np.float32)


def _preprocess_radar_input(
    lead_config: LeadConfig,
    radar_points_by_sensor: dict[str, jt.Float16[npt.NDArray, "_ 4"]],
) -> list[jt.Float32[npt.NDArray, "_ 5"]]:
    """Preprocess radar input data for model inference."""
    if not lead_config.expert.sensor_rig.use_radars:
        return []

    transfuser_config = lead_config.policy.transfuser

    def filter_and_pad_radars(radar_points):
        # Filter points within spatial bounds
        x_mask = (radar_points[:, 0] >= transfuser_config.bev_min_x_meter) & (
            radar_points[:, 0] <= transfuser_config.bev_max_x_meter
        )
        y_mask = (radar_points[:, 1] >= transfuser_config.bev_min_y_meter) & (
            radar_points[:, 1] <= transfuser_config.bev_max_y_meter
        )
        valid_mask = x_mask & y_mask
        filtered_arr = radar_points[valid_mask]

        # Pad the filtered array
        n = filtered_arr.shape[0]
        if n >= transfuser_config.num_radar_points_per_sensor:
            return filtered_arr[: transfuser_config.num_radar_points_per_sensor]
        out = np.zeros(
            (transfuser_config.num_radar_points_per_sensor, filtered_arr.shape[1]),
            dtype=np.float32,
        )
        out[:n] = filtered_arr.astype(np.float32)
        return out

    radar_points_per_sensor = []
    for i in range(1, lead_config.expert.sensor_rig.num_radar_sensors + 1):
        padded_radar = filter_and_pad_radars(radar_points_by_sensor[f"radar{i}"])

        # Add sensor identity column (0-indexed)
        sensor_index_column = np.full(
            (padded_radar.shape[0], 1),
            float(i - 1),
            dtype=np.float32,
        )
        radar_with_id = np.concatenate([padded_radar, sensor_index_column], axis=1)
        radar_points_per_sensor.append(radar_with_id)
    return radar_points_per_sensor

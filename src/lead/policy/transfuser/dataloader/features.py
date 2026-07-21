"""TransFuser featurization: one frame of raw driving data into the model inputs."""

import typing

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.common.sensors import ransac
from lead.config import LeadConfig
from lead.dataloader import Frame, view_geometry
from lead.dataloader.box_decoding import carla_forward_speed
from lead.dataloader.sensor_decoding import (
    lidar_sweep_to_carla_ego_frame,
    radar_to_carla_ego_frame,
    split_radar_by_sensor,
)
from lead.policy.transfuser.dataloader.point_cloud import accumulate_lidar_points


def build_features(
    frame: Frame,
    lead_config: LeadConfig,
    image_augmenter: typing.Any | None = None,
) -> dict[str, typing.Any]:
    """Turn one frame into the inputs the model consumes at training and inference.

    Args:
        frame: One tick of raw driving data in the ego view frame.
        lead_config: Root config tree.
        image_augmenter: Color augmenter applied to the stitched camera image;
            None to leave the image untouched.

    Returns:
        The model inputs of one unbatched sample.
    """
    data: dict[str, typing.Any] = {}
    config = lead_config.policy.transfuser

    # RGB: concatenate the per-camera images side by side, left to right.
    image = np.concatenate([camera.image for camera in frame.cameras], axis=1)
    if image_augmenter is not None:
        image = image_augmenter(image=image)
        assert isinstance(image, np.ndarray)
    data["rgb"] = np.transpose(image, (2, 0, 1))

    # LiDAR BEV: accumulate the sweep history into the anchor frame. The 123D
    # sweeps are decoded into the CARLA ego frame.
    if frame.lidar_sweeps is not None:
        assert frame.past_positions is not None
        assert frame.past_yaws is not None
        lidar_points = accumulate_lidar_points(
            {
                age: lidar_sweep_to_carla_ego_frame(lidar)
                for age, lidar in frame.lidar_sweeps.items()
            },
            (
                None
                if frame.radar_sweeps is None
                else {
                    age: radar_to_carla_ego_frame(radar)[:, :3].astype(np.float64)
                    for age, radar in frame.radar_sweeps.items()
                }
            ),
            frame.past_positions,
            frame.past_yaws,
            lead_config,
        )
        if frame.perturbation is not None:
            lidar_points = view_geometry.to_view_frame_lidar(
                lidar_points,
                frame.perturbation.translation,
                frame.perturbation.rotation,
            )
        points = lidar_points[
            lidar_points[:, 3] < lead_config.training.data.training_used_lidar_steps
        ]
        data["rasterized_lidar"] = np.array(
            _rasterize_lidar(lead_config=lead_config, lidar=points[:, :3]),
        ).squeeze()[None]

    # Radars: the anchor tick's merged returns split per sensor, filtered and
    # padded to a fixed size.
    if frame.radar_sweeps is not None:
        radar_list = _preprocess_radar_input(
            lead_config,
            {
                f"radar{i + 1}": radar
                for i, radar in enumerate(split_radar_by_sensor(frame.radar_sweeps[0]))
            },
        )
        for i, arr in enumerate(radar_list):
            data[f"radar{i + 1}"] = arr
        data["radar"] = np.concatenate(radar_list, axis=0)

    # Navigation and ego state.
    for key in ("target_point_previous", "target_point", "target_point_next"):
        point = getattr(frame, key)
        if point is not None:
            data[key] = point
    if config.use_velocity:
        assert frame.ego_state is not None
        data["speed"] = carla_forward_speed(frame.ego_state)
    assert frame.log_metadata is not None
    data["town"] = frame.log_metadata.location

    return data


def _rasterize_lidar(
    lead_config: LeadConfig,
    lidar: jt.Float[npt.NDArray, "N 3"],
    remove_ground_plane: bool = False,
) -> jt.Float[npt.NDArray, "H W"]:
    """
    Convert LiDAR point cloud into pseudo-image.

    Args:
        lead_config: Root config tree.
        lidar: LiDAR point cloud.
        remove_ground_plane: whether to remove ground plane points.
    Returns:
        Sparse pseudo-image.
    """
    data_config = lead_config.expert.data_collection

    def splat_points(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(
            data_config.min_x_meter,
            data_config.max_x_meter,
            (data_config.max_x_meter - data_config.min_x_meter)
            * int(data_config.pixels_per_meter)
            + 1,
        )
        ybins = np.linspace(
            data_config.min_y_meter,
            data_config.max_y_meter,
            (data_config.max_y_meter - data_config.min_y_meter)
            * int(data_config.pixels_per_meter)
            + 1,
        )
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > lead_config.training.data.hist_max_per_pixel] = (
            lead_config.training.data.hist_max_per_pixel
        )
        overhead_splat = hist / lead_config.training.data.hist_max_per_pixel
        # Transpose swaps axes: carla is x front, y right, whereas the image is
        # y front, x right (x height channel, y width channel).
        return overhead_splat.T

    # Remove points above the vehicle
    features = splat_points(lidar)
    lidar = lidar[
        (lidar[..., 2] <= lead_config.policy.transfuser.max_height_lidar)
        & (lead_config.policy.transfuser.min_height_lidar <= lidar[..., 2])
    ]
    if remove_ground_plane:
        is_ground_mask = ransac.remove_ground(lidar, lead_config.expert)
        above = lidar[~is_ground_mask]
        features = splat_points(above)
    else:
        features = np.stack([splat_points(point_cloud=lidar)], axis=-1)
    return (features).squeeze().astype(np.float32)


def _preprocess_radar_input(
    lead_config: LeadConfig,
    radar_data_dict: dict,
) -> list[jt.Float[npt.NDArray, "N 5"]]:
    """Preprocess radar input data for model inference.

    Args:
        lead_config: Root config tree.
        radar_data_dict: Dictionary containing radar data from sensors (e.g., {"radar1": array, "radar2": array, ...}).

    Returns:
        List of preprocessed radar data with sensor ID as last column.
    """
    if not lead_config.expert.sensor_rig.use_radars:
        return []

    config = lead_config.policy.transfuser
    data_config = lead_config.expert.data_collection

    def filter_and_pad_radars(arr):
        # Filter points within spatial bounds
        x_mask = (arr[:, 0] >= data_config.min_x_meter) & (
            arr[:, 0] <= data_config.max_x_meter
        )
        y_mask = (arr[:, 1] >= data_config.min_y_meter) & (
            arr[:, 1] <= data_config.max_y_meter
        )
        valid_mask = x_mask & y_mask
        filtered_arr = arr[valid_mask]

        # Pad the filtered array
        n = filtered_arr.shape[0]
        if n >= config.num_radar_points_per_sensor:
            return filtered_arr[: config.num_radar_points_per_sensor]
        out = np.zeros(
            (config.num_radar_points_per_sensor, filtered_arr.shape[1]),
            dtype=np.float32,
        )
        out[:n] = filtered_arr.astype(np.float32)
        return out

    radar_list = []
    for i in range(1, lead_config.expert.sensor_rig.num_radar_sensors + 1):
        padded_radar = filter_and_pad_radars(radar_data_dict[f"radar{i}"])

        # Add sensor identity column (0-indexed)
        sensor_id = np.full((padded_radar.shape[0], 1), float(i - 1), dtype=np.float32)
        radar_with_id = np.concatenate([padded_radar, sensor_id], axis=1)
        radar_list.append(radar_with_id)
    return radar_list

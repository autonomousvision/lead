"""TransFuser training dataset: sample parts over the generic scene data, with a
per-part opt-in cache store."""

import logging
import typing
from functools import partial

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE3

from lead.api.abstract_dataset import AbstractPolicyDataset
from lead.api.driving_meta import DrivingMeta
from lead.api.py123d_log_api import (
    BOX_ATTRIBUTES_KEY,
    LOCALIZED_EGO_STATE_KEY,
    se3_matrix_to_localized_pose,
)
from lead.api.training_sample import SamplePart
from lead.common import constants, geometry
from lead.config import LeadConfig, TransfuserConfig
from lead.log_reader import SceneData, SceneLoadingSpec, carla_decoding, view_geometry
from lead.log_reader.scene_loader import SceneLoader
from lead.policy.transfuser.dataloader import route_smoothing
from lead.policy.transfuser.dataloader.features import (
    build_camera_features,
    build_lidar_raster,
    build_radar_features,
)
from lead.policy.transfuser.dataloader.label_builders import (
    build_depth_target,
    build_labels,
)
from lead.policy.transfuser.dataloader.sample import (
    TransfuserOutputs,
    TransfuserTrainingSample,
)

LOG = logging.getLogger(__name__)


# Driving-driving_meta keys copied into the batch item verbatim.
_VERBATIM_DRIVING_META_KEYS = [
    "current_active_scenario_type",
    "previous_active_scenario_type",
    "changed_route",
    "stop_sign_hazard",
    "walker_hazard",
    "light_hazard",
    "vehicle_hazard",
    "lane_type_str",
    "does_emergency_brake_for_pedestrians",
    "construction_obstacle_two_ways_stuck",
    "accident_two_ways_stuck",
    "parked_obstacle_two_ways_stuck",
    "vehicle_opens_door_two_ways_stuck",
    "vehicle_opened_door",
    "vehicle_door_side",
    "ego_lane_id",
    "rear_danger_8",
    "rear_danger_16",
    "brake_cutin",
    "weather_setting",
    "jpeg_storage_quality",
    "emergency_brake_for_special_vehicle",
    "visual_visibility",
    "num_parking_vehicles_in_proximity",
    "slower_bad_visibility",
    "slower_clutterness",
    "over_head_traffic_light",
    "europe_traffic_light",
    "stop_sign_close",
    "num_dangerous_adversarial",
    "num_safe_adversarial",
    "num_ignored_adversarial",
    "rear_adversarial_id",
]

# Driving-driving_meta keys coerced to float in the batch item (None → inf).
_FLOAT_COERCED_DRIVING_META_KEYS = [
    "steer",
    "throttle",
    "brake",
    "dist_to_construction_site",
    "dist_to_accident_site",
    "dist_to_parked_obstacle",
    "dist_to_vehicle_opens_door",
    "dist_to_cutin_vehicle",
    "dist_to_pedestrian",
    "dist_to_biker",
    "distance_to_next_junction",
    "signed_dist_to_lane_change",
    "speed_limit",
    "distance_to_intersection_index_ego",
    "ego_lane_width",
    "route_left_length",
    "distance_ego_to_route",
    "target_speed_limit",
    "target_speed",
    "traffic_light_height",
]


# Distance of the first smoothed route point ahead of the ego, in meters.
_SMOOTHED_ROUTE_FIRST_POINT_DISTANCE_M = 2.5

# policy.transfuser fields a cacheable part reads to decide a cached tensor's
# shape or content; see TransfuserDataset.cache_finger_print. Head
# toggles (use_semantic, detect_boxes, ...) are deliberately excluded: they
# decide what gets consumed from the store, never what gets written into it.
_CACHE_FINGER_PRINT_FIELDS = (
    # BEV raster geometry and LiDAR preprocessing (features.py, point_cloud.py).
    "bev_pixels_per_meter",
    "bev_min_x_meter",
    "bev_max_x_meter",
    "bev_min_y_meter",
    "bev_max_y_meter",
    "max_lidar_points_per_bev_pixel",
    "lidar_max_height_meter",
    "lidar_min_height_meter",
    "accumulate_lidar_sweeps",
    "past_lidar_tick_ages",
    "past_radar_tick_ages",
    "remove_lidar_ground_points",
    "merge_radar_into_lidar",
    "duplicate_radar_near_ego",
    "duplicate_radar_radius_meter",
    "duplicate_radar_repeat_count",
    # BEV-semantic labels (label_builders.py, bev_raster.py).
    "bev_downsample_factor",
    "pedestrian_bev_extent_scale",
    "pedestrian_bev_min_half_extent_meter",
    "lane_marker_width_meter",
    # Bounding-box labels (label_builders.py).
    "max_num_boxes",
    "num_box_classes",
    "num_yaw_bins",
    "detected_static_prop_type_ids",
    "open_door_extra_width_meter",
    "min_box_center_z_meter",
    "max_box_center_z_meter",
    # Radar-detection labels (label_builders.py).
    "num_radar_queries",
    # Stitched geometry of the semantic labels (label_builders.py).
    "final_image_width",
    "final_image_height",
)


class TransfuserDataset(AbstractPolicyDataset):
    """Training dataset producing the TransFuser model inputs and labels."""

    sample_class = TransfuserTrainingSample

    def __init__(self, lead_config: LeadConfig, scene_loader: SceneLoader) -> None:
        """Construct the dataset over the 123D scenes, opening the cache store if configured.

        Args:
            lead_config: Root config tree.
            scene_loader: Loader over the scenes to train on, built by the
                policy (see ``Transfuser.build_scene_loader``).
        """
        super().__init__(lead_config, scene_loader, lead_config.policy.transfuser)

    def prepare_read(self) -> None:
        super().prepare_read()
        # Disable threading: the DataLoader already splits across workers.
        cv2.setNumThreads(0)

    @property
    def cache_finger_print(self) -> dict[str, str]:
        """Inherited, see superclass."""
        transfuser_config = self.lead_config.policy.transfuser
        finger_print = {
            name: str(getattr(transfuser_config, name))
            for name in _CACHE_FINGER_PRINT_FIELDS
        }
        # Which cameras feed the stitched semantic labels, in stitch order.
        finger_print["input_cameras"] = str(
            [camera.name for camera in transfuser_config.input_cameras],
        )
        # The rig decides whether radar reaches the lidar raster and whether
        # the radar targets exist at all.
        finger_print["use_radars"] = str(self.lead_config.expert.sensor_rig.use_radars)
        return finger_print

    def get_sample_parts(self) -> dict[str, SamplePart]:
        """The policy's sample parts, config-gated exactly like the featurization.

        Returns:
            The parts, keyed by part name.
        """
        lead_config = self.lead_config
        config = lead_config.policy.transfuser
        use_radars = lead_config.expert.sensor_rig.use_radars
        anchor_only = (0,)

        parts: dict[str, SamplePart] = {
            "camera_features": SamplePart(
                reads=SceneLoadingSpec(rgb_tick_ages=config.past_rgb_tick_ages),
                builds=partial(build_camera_features, lead_config=lead_config),
            ),
            "meta_features": SamplePart(
                reads=SceneLoadingSpec(
                    ego_pose_tick_ages=config.past_ego_pose_tick_ages,
                ),
                builds=self._build_meta_features,
            ),
        }
        if use_radars:
            parts["radar_features"] = SamplePart(
                reads=SceneLoadingSpec(radar_tick_ages=anchor_only),
                builds=partial(build_radar_features, lead_config=lead_config),
            )
        if not config.LTF:
            parts["lidar_raster"] = SamplePart(
                reads=SceneLoadingSpec(
                    lidar_tick_ages=config.past_lidar_tick_ages,
                    radar_tick_ages=config.past_radar_tick_ages if use_radars else (),
                ),
                builds=partial(build_lidar_raster, lead_config=lead_config),
                # The splat holds exactly k / max_lidar_points_per_bev_pixel.
                caches={
                    "rasterized_lidar": {
                        "name": "png",
                        "quantization_scale": config.max_lidar_points_per_bev_pixel,
                    },
                },
            )
        if config.needs_planning_targets:
            parts["planning_targets"] = SamplePart(
                reads=SceneLoadingSpec(
                    future_iterations=config.future_ego_pose_iterations,
                ),
                # Never cached: positions are cheap to read live, and caching
                # them would couple the store's coverage to the planning head.
                builds=self._build_planning_targets,
            )
        if (
            config.detect_boxes
            or config.use_bev_semantic
            or config.use_semantic
            or config.use_radar_detection
        ):
            parts["privileged_targets"] = SamplePart(
                reads=SceneLoadingSpec(
                    read_semantic_cameras=config.use_semantic,
                    read_map_api=config.use_bev_semantic,
                    radar_tick_ages=anchor_only if use_radars else (),
                ),
                builds=self._build_privileged_targets,
                caches=_privileged_codecs(config, use_radars=use_radars),
            )
        if config.use_depth:
            parts["depth_target"] = SamplePart(
                reads=SceneLoadingSpec(read_depth_cameras=True),
                builds=build_depth_target,
            )
        return parts

    # --- Builders ---
    def _build_meta_features(self, scene_data: SceneData) -> dict[str, typing.Any]:
        """Identity fields, driving_meta lifts and scenario ids of one scene."""
        driving_meta = scene_data.driving_meta
        assert driving_meta is not None
        assert scene_data.log_metadata is not None
        assert scene_data.scene_metadata is not None
        rig_perturbation = scene_data.rig_perturbation

        meta_entries: dict[str, typing.Any] = {
            "perturbate_sensor": rig_perturbation is not None,
            "perturbation_translation": (
                rig_perturbation.lateral_translation_m
                if rig_perturbation is not None
                else 0.0
            ),
            "perturbation_rotation": (
                rig_perturbation.yaw_rotation_deg
                if rig_perturbation is not None
                else 0.0
            ),
            "route_number": scene_data.log_metadata.log_name,
            "frame_number": scene_data.scene_metadata.initial_idx,
            "scenario_type_dir": str(scene_data.log_metadata.split).split("/")[-1],
            "town": scene_data.log_metadata.location,
        }

        for meta_key in _VERBATIM_DRIVING_META_KEYS:
            if meta_key == "vehicle_door_side" and driving_meta.get(meta_key) is None:
                meta_entries[meta_key] = "NA"
            elif meta_key == "vehicle_door_side":
                door_side = driving_meta[meta_key]
                meta_entries[meta_key] = (
                    door_side[0] if isinstance(door_side, list) else door_side
                )
            else:
                meta_entries[meta_key] = driving_meta[meta_key]

        for meta_key in _FLOAT_COERCED_DRIVING_META_KEYS:
            if meta_key in driving_meta and driving_meta[meta_key] is None:
                meta_entries[meta_key] = np.inf
            elif meta_key in driving_meta:
                meta_entries[meta_key] = float(driving_meta[meta_key])
        assert scene_data.ego_state is not None
        meta_entries["speed"] = carla_decoding.carla_forward_speed(scene_data.ego_state)

        for meta_key in [
            "current_active_scenario_type",
            "previous_active_scenario_type",
        ]:
            meta_entries[meta_key] = (
                "NA" if driving_meta.get(meta_key) is None else driving_meta[meta_key]
            )
        if meta_entries["current_active_scenario_type"] != "NA":
            meta_entries["scenario_type"] = meta_entries["current_active_scenario_type"]
        elif meta_entries["previous_active_scenario_type"] != "NA":
            meta_entries["scenario_type"] = meta_entries[
                "previous_active_scenario_type"
            ]
        else:
            meta_entries["scenario_type"] = "NA"
        meta_entries["scenario_type_id"] = constants.SCENARIO_TYPES.index(
            meta_entries["scenario_type"],
        )
        return meta_entries

    def _build_planning_targets(self, scene_data: SceneData) -> TransfuserOutputs:
        """The ego's future waypoints and the smoothed route of one scene."""
        driving_meta = scene_data.driving_meta
        assert driving_meta is not None
        planning_entries: TransfuserOutputs = {}
        self._add_future_waypoints(
            planning_entries,
            scene_data,
            driving_meta,
            scene_data.future_driving_metas or {},
            scene_data.future_ego_states or {},
        )
        self._add_route(planning_entries, scene_data, driving_meta)
        return planning_entries

    def _build_privileged_targets(self, scene_data: SceneData) -> TransfuserOutputs:
        """The detection, segmentation and BEV targets of one scene.

        Decodes the tick's boxes into the view frame and builds the labels
        from them.
        """
        lead_config = self.lead_config
        driving_meta = scene_data.driving_meta
        assert driving_meta is not None

        boxes: list[dict] | None = None
        if scene_data.box_detections is not None:
            assert scene_data.ego_state is not None
            boxes = carla_decoding.box_detections_to_carla_ego_frame(
                scene_data.box_detections,
                scene_data.ego_state,
                driving_meta[BOX_ATTRIBUTES_KEY],
            )
            if scene_data.rig_perturbation is not None:
                boxes = view_geometry.to_view_frame_boxes(
                    boxes,
                    scene_data.rig_perturbation.lateral_translation_m,
                    scene_data.rig_perturbation.yaw_rotation_deg,
                )
        return build_labels(scene_data, boxes, lead_config)

    def _add_future_waypoints(
        self,
        planning_entries: TransfuserOutputs,
        scene_data: SceneData,
        driving_meta: DrivingMeta,
        future_driving_metas: dict[int, DrivingMeta | None],
        future_ego_states: dict[int, EgoStateSE3 | None],
    ) -> None:
        """Add the ego's future poses as the planning labels of the planning_entries."""
        assert scene_data.ego_state is not None
        ego_position, _ = carla_decoding.carla_ego_pose(scene_data.ego_state)
        ego_yaw = _localized_yaw(driving_meta)

        future_iterations = (
            self.lead_config.policy.transfuser.future_ego_pose_iterations
        )
        future_waypoints: list[jt.Float[npt.NDArray, " 2"]] = []
        future_yaws: list[float] = []
        for future_iteration in future_iterations:
            future_driving_meta: DrivingMeta | None = future_driving_metas.get(
                future_iteration,
            )
            future_state = future_ego_states.get(future_iteration)
            # Skipping would shift every later waypoint into an earlier slot,
            # silently mislabelling its time horizon.
            if future_driving_meta is None or future_state is None:
                raise ValueError(
                    f"Missing future iteration {future_iteration} of "
                    f"{future_iterations}; the scene filter must "
                    f"only enumerate scenes whose full future is available.",
                )
            future_position, _ = carla_decoding.carla_ego_pose(future_state)
            future_waypoints.append(
                geometry.to_local_frame_2d(future_position, ego_position, ego_yaw),
            )
            future_yaws.append(
                geometry.normalize_angle_rad(
                    _localized_yaw(future_driving_meta) - ego_yaw,
                ),
            )

        waypoints: jt.Float[npt.NDArray, "n 2"] = np.array(future_waypoints).reshape(
            -1,
            2,
        )
        yaws: jt.Float[npt.NDArray, " n"] = np.array(future_yaws).reshape(-1)
        if scene_data.rig_perturbation is not None:
            waypoints = view_geometry.to_view_frame_points(
                waypoints,
                scene_data.rig_perturbation.lateral_translation_m,
                scene_data.rig_perturbation.yaw_rotation_deg,
            )
            yaws = view_geometry.to_view_frame_yaws(
                yaws,
                scene_data.rig_perturbation.yaw_rotation_deg,
            )
        planning_entries["future_waypoints"] = waypoints
        planning_entries["future_yaws"] = yaws

    def _add_route(
        self,
        planning_entries: TransfuserOutputs,
        scene_data: SceneData,
        driving_meta: DrivingMeta,
    ) -> None:
        """Add the route ahead of the ego, smoothed and in the view frame."""
        lead_config: LeadConfig = self.lead_config
        transfuser_config = lead_config.policy.transfuser

        # The route is stored in the global frame; convert to the ego frame.
        assert scene_data.ego_state is not None
        ego_position, _ = carla_decoding.carla_ego_pose(scene_data.ego_state)
        ego_yaw = _localized_yaw(driving_meta)
        route: jt.Float[npt.NDArray, "n 2"] = np.array(
            [
                geometry.to_local_frame_2d(
                    np.array(point),
                    ego_position,
                    ego_yaw,
                )
                for point in driving_meta["route"][
                    : transfuser_config.num_route_points_smoothing
                ]
            ],
        )
        if transfuser_config.smooth_route:
            route = route_smoothing.smooth_route(
                lead_config,
                route,
                target_first_distance=_SMOOTHED_ROUTE_FIRST_POINT_DISTANCE_M,
            )
        route = route[: transfuser_config.num_route_points_prediction]
        if scene_data.rig_perturbation is not None:
            route = view_geometry.to_view_frame_points(
                route,
                scene_data.rig_perturbation.lateral_translation_m,
                scene_data.rig_perturbation.yaw_rotation_deg,
            )
        planning_entries["route"] = route


# --- Module-level helpers ---
def _privileged_codecs(
    transfuser_config: TransfuserConfig,
    *,
    use_radars: bool,
) -> dict[str, str | dict]:
    """Codecs of the privileged targets each enabled head contributes.

    Args:
        transfuser_config: The ``policy.transfuser`` config section.
        use_radars: Whether the sensor rig records radars.

    Returns:
        The tensor-name to codec-spec map of the privileged part.
    """
    codecs: dict[str, str | dict] = {}
    if transfuser_config.detect_boxes:
        codecs |= {
            "center_net_heatmap": "zlib",
            "center_net_wh": "zlib",
            "center_net_offset": "zlib",
            "center_net_yaw_class": "zlib",
            "center_net_yaw_res": "zlib",
            "center_net_velocity": "zlib",
            "center_net_brake": "zlib",
            "center_net_pixel_weight": "zlib",
            "center_net_bounding_boxes": "zlib",
            "center_net_avg_factor": "raw",
        }
    if use_radars:
        codecs["radar_detections"] = "raw"
    if transfuser_config.use_semantic:
        codecs["semantic"] = {"name": "png", "quantization_scale": 1}
    if transfuser_config.use_bev_semantic:
        codecs["bev_semantic"] = {"name": "png", "quantization_scale": 1}
    return codecs


def _localized_yaw(driving_meta: DrivingMeta) -> float:
    """The tick's localized ego yaw, read from its SE(3) pose matrix."""
    _, yaw = se3_matrix_to_localized_pose(
        np.asarray(driving_meta[LOCALIZED_EGO_STATE_KEY], dtype=np.float64),
    )
    return yaw

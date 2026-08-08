"""LEAD's per-tick driving state: what the expert knew and decided.

The ``driving_meta`` custom py123d modality of every LEAD log. Reading it needs
neither the simulator nor the expert, so this is the description a py123d-only
consumer works from.

Key types are those of the writer (``ExpertAgent.save_meta``). Values that pass
through the modality's serializer arrive as whatever it reconstructs, so the
array-valued keys are typed as bare arrays and the readers wrap them in
``np.asarray`` for that reason.
"""

import typing

import numpy as np


class RequiredDrivingMeta(typing.TypedDict):
    """The driving-meta keys every reader indexes directly.

    A log missing one of these cannot be loaded, so they are required even
    though the rest of the stream is version-dependent.
    """

    localized_ego_state_se3: np.ndarray
    route: np.ndarray
    target_point_indices: dict[str, int]
    stop_sign_hazard: bool
    scenario_obstacles_ids: list[int]
    vehicle_opened_door: bool
    vehicle_door_side: str | list[str] | None
    europe_traffic_light: bool
    over_head_traffic_light: bool
    box_attributes: dict[str, dict[str, typing.Any]]


class DrivingMeta(RequiredDrivingMeta, total=False):
    """One tick of LEAD's driving-meta stream.

    Every key is optional: the stream has grown over dataset versions, and a
    reader must tolerate a log written before a key existed.
    """

    # --- Localization and route ---
    # The GNSS+IMU-localized ego pose — the pose the policy observes — as a 4x4
    # SE(3) matrix in the global frame.
    # The dense route ahead in the global frame, and the pre-reroute original.
    route_original: np.ndarray
    changed_route: bool
    # Index of the current target point per route-planner pop distance, keyed by
    # the pop distance as a string.
    route_left_length: float | None
    distance_ego_to_route: float | None

    # --- Expert controls and speed ---
    steer: float | None
    throttle: float | None
    brake: bool
    target_speed: float | None
    target_speed_limit: float | None
    speed_limit: float | None
    last_encountered_speed_limit_sign: float | None
    speed_reduced_by_obj_type: str | None
    speed_reduced_by_obj_id: int | None
    speed_reduced_by_obj_distance: float | None

    # --- Hazards the expert reacted to ---
    vehicle_hazard: bool
    vehicle_affecting_id: int | None
    walker_hazard: bool
    walker_affecting_id: int | None
    walker_close: bool
    walker_close_id: int | None
    light_hazard: bool
    stop_sign_close: bool
    does_emergency_brake_for_pedestrians: bool
    emergency_brake_for_special_vehicle: bool
    brake_cutin: bool
    rear_danger_8: bool
    rear_danger_16: bool

    # --- Scenario state ---
    scenario: str | None
    current_active_scenario_type: str | None
    previous_active_scenario_type: str | None
    scenario_actors_ids: list[int]
    scenario_obstacles_convex_hull: np.ndarray
    cut_in_actors_ids: list[int]
    # "left" or "right"; some scenarios record it as a one-element list.
    construction_obstacle_two_ways_stuck: bool
    accident_two_ways_stuck: bool
    parked_obstacle_two_ways_stuck: bool
    vehicle_opens_door_two_ways_stuck: bool

    # --- Distances to scenario features, in meters ---
    dist_to_construction_site: float | None
    dist_to_accident_site: float | None
    dist_to_parked_obstacle: float | None
    dist_to_vehicle_opens_door: float | None
    dist_to_cutin_vehicle: float | None
    dist_to_pedestrian: float | None
    dist_to_biker: float | None
    dist_to_junction: float | None
    distance_to_next_junction: float | None
    distance_to_intersection_index_ego: float | None
    signed_dist_to_lane_change: float | None

    # --- Road graph at the ego's waypoint ---
    ego_lane_id: int
    ego_lane_width: float | None
    target_lane_width: float | None
    road_id: int
    lane_id: int
    is_junction: bool
    junction_id: int
    next_road_ids: list[int]
    next_next_road_ids_ego: list[int]
    lane_change_str: str
    lane_type_str: str
    left_lane_marking_color_str: str
    left_lane_marking_type_str: str
    right_lane_marking_color_str: str
    right_lane_marking_type_str: str

    # --- Traffic lights ---
    traffic_light_height: float | None

    # --- Environment and capture ---
    weather_setting: str
    weather_parameters: dict[str, float]
    jpeg_storage_quality: int
    visual_visibility: int
    slower_bad_visibility: bool
    slower_clutterness: bool
    num_parking_vehicles_in_proximity: int

    # --- Adversarial actors ---
    num_dangerous_adversarial: int
    num_safe_adversarial: int
    num_ignored_adversarial: int
    # -1 when there is no rear adversarial actor.
    rear_adversarial_id: int

    # --- Per-box fields the box-detections stream has no slot for, by track token ---

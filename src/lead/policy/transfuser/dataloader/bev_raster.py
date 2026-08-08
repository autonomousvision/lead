"""Load-time BEV semantic rasterization from the 123D map.

Draws the static ``BEVSemanticClass`` map classes at training
resolution from 123D map queries; dynamic classes are overlaid afterwards from
the box detections.
"""

import itertools
import math

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api.map.map_api import MapAPI
from py123d.datatypes import MapLayer
from py123d.datatypes.map_objects.map_layer_types import StopZoneType
from py123d.geometry import Point2D, PoseSE2

from lead.config import LeadConfig
from lead.config.policy.transfuser.label_classes import BEVSemanticClass


def _query_radius_meter(lead_config: LeadConfig) -> float:
    """Radius around the ego covering every grid corner."""
    config = lead_config.policy.transfuser
    return math.sqrt(
        max(abs(config.bev_min_x_meter), abs(config.bev_max_x_meter)) ** 2
        + max(abs(config.bev_min_y_meter), abs(config.bev_max_y_meter)) ** 2,
    )


def _world_to_pixel(
    points: jt.Float[npt.NDArray, "n 2"],
    center_se2: PoseSE2,
    lead_config: LeadConfig,
) -> jt.Int32[npt.NDArray, "n 2"]:
    """Transform world-frame 2D points into BEV pixel coordinates."""
    config = lead_config.policy.transfuser
    cos_yaw = math.cos(center_se2.yaw)
    sin_yaw = math.sin(center_se2.yaw)
    dx = points[:, 0] - center_se2.x
    dy = points[:, 1] - center_se2.y
    # Ego frame: x forward, y left (ISO 8855)
    x_ego = cos_yaw * dx + sin_yaw * dy
    y_ego = -sin_yaw * dx + cos_yaw * dy
    # Grid: columns along x (back → front), rows along -y (left → right).
    cols = (x_ego - config.bev_min_x_meter) * config.bev_pixels_per_meter
    rows = (-y_ego - config.bev_min_y_meter) * config.bev_pixels_per_meter
    return np.stack([cols, rows], axis=1).astype(np.int32)


def rasterize_bev_semantic_map(
    map_api: MapAPI,
    center_se2: PoseSE2,
    stop_sign_hazard: bool,
    lead_config: LeadConfig,
) -> jt.UInt8[npt.NDArray, "rows cols"]:
    """Draw the static map part of the BEV semantic grid.

    Args:
        map_api: Queryable 123D map of the town.
        center_se2: Global pose anchoring the grid (the view frame's origin).
        stop_sign_hazard: Whether a stop sign currently affects the ego; stop
            zones are only drawn while it does.
        lead_config: Root config tree with the planning area geometry.

    Returns:
        BEV semantic class grid of shape
        (``lidar_height_pixel``, ``lidar_width_pixel``).
    """
    config = lead_config.policy.transfuser
    grid = np.zeros(
        (config.lidar_height_pixel, config.lidar_width_pixel),
        dtype=np.uint8,
    )

    query_point = Point2D(x=center_se2.x, y=center_se2.y)
    radius = _query_radius_meter(lead_config)

    # Road surface: literal layer arguments so the typed per-layer map query
    # resolves each result to its concrete object class.
    for map_object in itertools.chain(
        map_api.get_layer_objects_in_radius(query_point, radius, MapLayer.LANE),
        map_api.get_layer_objects_in_radius(query_point, radius, MapLayer.INTERSECTION),
        map_api.get_layer_objects_in_radius(
            query_point,
            radius,
            MapLayer.GENERIC_DRIVABLE,
        ),
    ):
        exterior = np.array(map_object.shapely_polygon.exterior.coords)[:, :2]
        cv2.fillPoly(
            grid,
            [_world_to_pixel(exterior, center_se2, lead_config)],
            (int(BEVSemanticClass.ROAD),),
        )

    # Lane markings
    for road_line in map_api.get_layer_objects_in_radius(
        query_point,
        radius,
        MapLayer.ROAD_LINE,
    ):
        polyline = np.array(road_line.polyline_2d.array)[:, :2]
        cv2.polylines(
            grid,
            [_world_to_pixel(polyline, center_se2, lead_config)],
            isClosed=False,
            color=(int(BEVSemanticClass.LANE_MARKERS),),
            thickness=config.lane_marker_thickness_pixel,
        )

    # Stop-sign stop zones, drawn only while a stop sign affects the ego
    if stop_sign_hazard:
        for stop_zone in map_api.get_layer_objects_in_radius(
            query_point,
            radius,
            MapLayer.STOP_ZONE,
        ):
            if stop_zone.stop_zone_type != StopZoneType.STOP_SIGN:
                continue
            exterior = np.array(stop_zone.shapely_polygon.exterior.coords)[:, :2]
            cv2.fillPoly(
                grid,
                [_world_to_pixel(exterior, center_se2, lead_config)],
                (int(BEVSemanticClass.STOP_SIGNS),),
            )

    return grid

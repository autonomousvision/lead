"""Load-time BEV semantic rasterization from the 123D map.

Draws the static ``BEVSemanticClass`` map classes at training
resolution from 123D map queries; dynamic classes are overlaid afterwards from
the box detections.
"""

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
from lead.policy.transfuser.labels import BEVSemanticClass

# Map layers rasterized as drivable road.
_ROAD_LAYERS = [MapLayer.LANE, MapLayer.INTERSECTION, MapLayer.GENERIC_DRIVABLE]


def _query_radius_meter(lead_config: LeadConfig) -> float:
    """Radius around the ego covering every grid corner.

    Args:
        lead_config: Root config tree with the planning area geometry.

    Returns:
        Query radius in meters.
    """
    data_config = lead_config.expert.data_collection
    return math.sqrt(
        max(abs(data_config.min_x_meter), abs(data_config.max_x_meter)) ** 2
        + max(abs(data_config.min_y_meter), abs(data_config.max_y_meter)) ** 2,
    )


def world_to_pixel(
    points: jt.Float[npt.NDArray, "n 2"],
    center_se2: PoseSE2,
    lead_config: LeadConfig,
) -> jt.Int32[npt.NDArray, "n 2"]:
    """Transform world-frame 2D points into BEV pixel coordinates.

    Args:
        points: World-frame (ISO 8855) xy points.
        center_se2: Global pose anchoring the grid.
        lead_config: Root config tree with the planning area geometry.

    Returns:
        Integer (col, row) pixel coordinates for OpenCV drawing.
    """
    data_config = lead_config.expert.data_collection
    cos_yaw = math.cos(center_se2.yaw)
    sin_yaw = math.sin(center_se2.yaw)
    dx = points[:, 0] - center_se2.x
    dy = points[:, 1] - center_se2.y
    # Ego frame: x forward, y left (ISO 8855)
    x_ego = cos_yaw * dx + sin_yaw * dy
    y_ego = -sin_yaw * dx + cos_yaw * dy
    # Grid: columns along x (back → front), rows along -y (left → right).
    cols = (x_ego - data_config.min_x_meter) * data_config.pixels_per_meter
    rows = (-y_ego - data_config.min_y_meter) * data_config.pixels_per_meter
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
    data_config = lead_config.expert.data_collection
    grid = np.zeros(
        (data_config.lidar_height_pixel, data_config.lidar_width_pixel),
        dtype=np.uint8,
    )

    query_point = Point2D(x=center_se2.x, y=center_se2.y)
    radius = _query_radius_meter(lead_config)

    # Road surface
    road_objects = map_api.get_map_objects_in_radius(
        query_point,
        radius,
        list[str | MapLayer](_ROAD_LAYERS),
    )
    for layer in _ROAD_LAYERS:
        for map_object in road_objects.get(layer, []):
            exterior = np.array(map_object.shapely_polygon.exterior.coords)[:, :2]
            cv2.fillPoly(
                grid,
                [world_to_pixel(exterior, center_se2, lead_config)],
                int(BEVSemanticClass.ROAD),
            )

    # Lane markings
    line_objects = map_api.get_map_objects_in_radius(
        query_point,
        radius,
        [MapLayer.ROAD_LINE],
    )
    for road_line in line_objects.get(MapLayer.ROAD_LINE, []):
        polyline = np.array(road_line.polyline_2d.array)[:, :2]
        cv2.polylines(
            grid,
            [world_to_pixel(polyline, center_se2, lead_config)],
            isClosed=False,
            color=int(BEVSemanticClass.LANE_MARKERS),
            thickness=1,
        )

    # Stop-sign stop zones, drawn only while a stop sign affects the ego
    if stop_sign_hazard:
        stop_objects = map_api.get_map_objects_in_radius(
            query_point,
            radius,
            [MapLayer.STOP_ZONE],
        )
        for stop_zone in stop_objects.get(MapLayer.STOP_ZONE, []):
            if stop_zone.stop_zone_type != StopZoneType.STOP_SIGN:
                continue
            exterior = np.array(stop_zone.shapely_polygon.exterior.coords)[:, :2]
            cv2.fillPoly(
                grid,
                [world_to_pixel(exterior, center_se2, lead_config)],
                int(BEVSemanticClass.STOP_SIGNS),
            )

    return grid

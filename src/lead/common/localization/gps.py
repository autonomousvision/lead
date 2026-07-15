"""Ego pose estimation from the GNSS and IMU sensors."""

import logging
import math
import re
import typing

import carla
import numpy as np
import numpy.typing as npt
from agents.navigation.local_planner import RoadOption
from scipy.optimize import fsolve

from lead.common.geometry import normalize_angle

LOG = logging.getLogger(__name__)


def convert_gps_to_carla(
    gps: npt.NDArray,
    lat_ref: float,
    lon_ref: float,
) -> npt.NDArray:
    """
    Converts GPS signal into the CARLA coordinate frame.

    Args:
        gps: GPS from GNSS sensor
        lat_ref: Latitude reference point of the map.
        lon_ref: Longitude reference point of the map.
    Returns:
        npt.NDArray: CARLA coordinates of the specific map in meters.
    """
    EARTH_RADIUS_EQUA = 6378137.0  # Constant from CARLA leaderboard GPS simulation

    lat, lon, _ = gps
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
    mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
    y = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
        - my
    )
    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    gps = np.array([x, y, gps[2]])

    return gps


def convert_gnss_to_carla(
    gnss: npt.NDArray,
    lat_ref: float,
    lon_ref: float,
    mirrors_latitude: bool,
) -> npt.NDArray:
    """
    Converts a GNSS sensor reading into the CARLA coordinate frame.

    Since CARLA 0.9.16 the GNSS sensor mirrors the latitude axis with respect to
    the convention the leaderboard uses for the route plan, so the y coordinate
    has to be flipped back; 0.9.15 and earlier do not need this.

    Args:
        gnss: GPS reading from the GNSS sensor.
        lat_ref: Latitude reference point of the map.
        lon_ref: Longitude reference point of the map.
        mirrors_latitude: Whether the connected CARLA version mirrors the GNSS
            latitude axis (true from 0.9.16 onwards). See `gnss_mirrors_latitude`.
    Returns:
        npt.NDArray: CARLA coordinates of the specific map in meters.
    """
    position = convert_gps_to_carla(gnss, lat_ref, lon_ref)
    if mirrors_latitude:
        position[1] *= -1.0

    return position


_GNSS_MIRRORS_LATITUDE_BY_VERSION = {
    "0.9.15": False,
    "0.9.16": True,
}


def gnss_mirrors_latitude(client: carla.Client) -> bool:
    """
    Whether the connected CARLA server's GNSS sensor mirrors the latitude axis.

    The behavior changed in CARLA 0.9.16; see `convert_gnss_to_carla`. Only the
    versions this codebase is validated against are recognized; anything else
    raises rather than silently guessing.

    Args:
        client: Connected CARLA client, used to query the server version.
    Returns:
        bool: True if the server is version 0.9.16, False if 0.9.15.
    Raises:
        ValueError: If the server is not 0.9.15 or 0.9.16.
    """
    version = client.get_server_version()
    match = re.match(r"\d+\.\d+\.\d+", version)
    version_key = match.group() if match else version
    if version_key not in _GNSS_MIRRORS_LATITUDE_BY_VERSION:
        raise ValueError(
            f"Unsupported CARLA server version {version!r}; expected one of "
            f"{sorted(_GNSS_MIRRORS_LATITUDE_BY_VERSION)}.",
        )
    return _GNSS_MIRRORS_LATITUDE_BY_VERSION[version_key]


def find_gps_ref(
    global_plan_world_coord: list[tuple[carla.Transform, RoadOption]],
    global_plan: list[tuple[dict[str, float], RoadOption]],
) -> tuple[float, float]:
    """Estimate the GPS lat/lon reference, which the CARLA leaderboard does not expose.

    The reference is solved from the route plan, which the leaderboard provides
    in both GPS and CARLA world coordinates.

    Args:
        global_plan_world_coord: The global plan in CARLA world coordinates.
        global_plan: The global plan in GPS coordinates. The dicts have keys 'lat', 'lon', 'z'.

    Returns:
        tuple[float, float]: lat_ref, lon_ref
    """
    try:
        locx, locy = (
            global_plan_world_coord[0][0].location.x,
            global_plan_world_coord[0][0].location.y,
        )
        lon, lat = global_plan[0][0]["lon"], global_plan[0][0]["lat"]
        earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation

        def equations(variables):
            x, y = variables
            eq1 = (
                lon * math.cos(x * math.pi / 180.0)
                - (locx * 180.0) / (math.pi * earth_radius_equa)
                - math.cos(x * math.pi / 180.0) * y
            )
            eq2 = (
                math.log(math.tan((lat + 90.0) * math.pi / 360.0))
                * earth_radius_equa
                * math.cos(x * math.pi / 180.0)
                + locy
                - math.cos(x * math.pi / 180.0)
                * earth_radius_equa
                * math.log(math.tan((90.0 + x) * math.pi / 360.0))
            )
            return [eq1, eq2]

        initial_guess = [0.0, 0.0]
        # scipy's fsolve overloads can't see through the `equations` closure to
        # infer a plain ndarray result; the runtime return is always one.
        solution = typing.cast("npt.NDArray", fsolve(equations, initial_guess))
        return float(solution[0]), float(solution[1])
    except Exception as e:
        LOG.warning(e)
        return 0.0, 0.0


def preprocess_compass(compass: float) -> float:
    """
    Checks the compass for Nans and rotates it into the default CARLA coordinate system with range [-pi,pi].

    Args:
        compass: compass value provided by the IMU, in radian

    Returns:
        float: yaw of the car in radian in the CARLA coordinate system.
    """
    if math.isnan(compass):  # simulation bug
        compass = 0.0
    # The minus 90.0 degree is because the compass sensor uses a different coordinate system then CARLA
    return normalize_angle(compass - np.deg2rad(90.0))

"""Ego pose estimation from the GNSS and IMU sensors."""

import logging
import math
import re

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from agents.navigation.local_planner import RoadOption

from lead.common.geometry import normalize_angle_rad

LOG = logging.getLogger(__name__)


def convert_gps_to_carla(
    gps: jt.Float[npt.NDArray, " 3"],
    lat_ref: float,
    lon_ref: float,
) -> jt.Float[npt.NDArray, " 3"]:
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
    return np.array([x, y, gps[2]])


def convert_tmerc_gnss_to_carla(
    gnss: jt.Float[npt.NDArray, " 3"],
    lat_ref: float,
    lon_ref: float,
) -> jt.Float[npt.NDArray, " 3"]:
    """
    Converts a transverse-Mercator GNSS reading into the CARLA coordinate frame.

    Exact inverse of CARLA 0.9.16's GNSS sensor (`GeoLocation::Transform`), a
    spherical transverse Mercator centered on the map's geo-reference with a
    hardcoded scale factor of 0.9996 and no latitude flip.

    Args:
        gnss: GPS reading from the GNSS sensor.
        lat_ref: Latitude reference point of the map.
        lon_ref: Longitude reference point of the map.
    Returns:
        npt.NDArray: CARLA coordinates of the specific map in meters.
    """
    EARTH_RADIUS_EQUA = 6378137.0  # Constant from CARLA leaderboard GPS simulation
    K0 = 0.9996  # Scale factor hardcoded in CARLA's GeoLocation

    lat, lon, _ = gnss
    phi = math.radians(lat)
    delta_lambda = math.radians(lon - lon_ref)
    b = math.cos(phi) * math.sin(delta_lambda)
    x = 0.5 * K0 * EARTH_RADIUS_EQUA * math.log((1 + b) / (1 - b))
    y = (
        K0
        * EARTH_RADIUS_EQUA
        * (math.atan(math.tan(phi) / math.cos(delta_lambda)) - math.radians(lat_ref))
    )

    return np.array([x, y, gnss[2]])


def convert_gnss_to_carla(
    gnss: jt.Float[npt.NDArray, " 3"],
    lat_ref: float,
    lon_ref: float,
    uses_transverse_mercator: bool,
) -> jt.Float[npt.NDArray, " 3"]:
    """
    Converts a GNSS sensor reading into the CARLA coordinate frame.

    CARLA 0.9.15's GNSS sensor applies the equatorial Mercator projection the
    leaderboard also uses for the route plan; CARLA 0.9.16's applies a
    transverse Mercator, so each reading is inverted with its own projection.

    Args:
        gnss: GPS reading from the GNSS sensor.
        lat_ref: Latitude reference point of the map.
        lon_ref: Longitude reference point of the map.
        uses_transverse_mercator: Whether the connected CARLA version projects
            with a transverse Mercator (true from 0.9.16 onwards). See
            `gnss_uses_transverse_mercator`.
    Returns:
        npt.NDArray: CARLA coordinates of the specific map in meters.
    """
    if uses_transverse_mercator:
        return convert_tmerc_gnss_to_carla(gnss, lat_ref, lon_ref)
    return convert_gps_to_carla(gnss, lat_ref, lon_ref)


_GNSS_USES_TRANSVERSE_MERCATOR_BY_VERSION = {
    "0.9.15": False,
    "0.9.16": True,
}


def gnss_uses_transverse_mercator(client: carla.Client) -> bool:
    """
    Whether the connected CARLA server's GNSS sensor projects with a transverse Mercator.

    The projection changed in CARLA 0.9.16; see `convert_gnss_to_carla`. Only
    the versions this codebase is validated against are recognized; anything
    else raises rather than silently guessing.

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
    if version_key not in _GNSS_USES_TRANSVERSE_MERCATOR_BY_VERSION:
        raise ValueError(
            f"Unsupported CARLA server version {version!r}; expected one of "
            f"{sorted(_GNSS_USES_TRANSVERSE_MERCATOR_BY_VERSION)}.",
        )
    return _GNSS_USES_TRANSVERSE_MERCATOR_BY_VERSION[version_key]


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

        def mercator_ordinate(lat_deg: float) -> float:
            return math.log(math.tan((90.0 + lat_deg) * math.pi / 360.0))

        # The projection satisfies ordinate(lat_ref) = ordinate(lat) +
        # locy / (R * cos(lat_ref)); the fixed-point iteration contracts by
        # ~locy/R per step, so a few steps reach machine precision.
        lat_ref = lat
        for _ in range(20):
            ordinate = mercator_ordinate(lat) + locy / (
                earth_radius_equa * math.cos(math.radians(lat_ref))
            )
            lat_ref = 360.0 * math.atan(math.exp(ordinate)) / math.pi - 90.0
        lon_ref = lon - locx * 180.0 / (
            math.pi * earth_radius_equa * math.cos(math.radians(lat_ref))
        )
        return lat_ref, lon_ref
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
    return normalize_angle_rad(compass - np.deg2rad(90.0))

"""Sampling of sensor pose perturbation parameters for data collection."""

import logging
import typing

import numpy as np

from lead.common import constants

if typing.TYPE_CHECKING:
    from lead.config import ExpertConfig

LOG = logging.getLogger(__name__)


def sample_sensor_perturbation_parameters(
    config: "ExpertConfig",
    max_speed_limit_route: float,
    min_lane_width_route: float,
) -> tuple[float, float]:
    """
    Sample sensor perturbation parameters (translation and rotation) based on the route's speed limit and lane width.
    Args:
        config: Configuration object containing perturbation parameters.
        max_speed_limit_route: Maximum speed limit along the route (in m/s).
        min_lane_width_route: Minimum lane width along the route (in meters).
    Returns:
        tuple[float, float]: A tuple containing the perturbation translation (in meters) and perturbation rotation (in degrees).
    """
    safety_translation_perturbation_gap = (
        config.perturbation.default_safety_translation_perturbation_penalty
    )
    if max_speed_limit_route < constants.URBAN_MAX_SPEED_LIMIT:
        safety_translation_perturbation_gap = (
            config.perturbation.urban_safety_translation_perturbation_penalty
        )
    lateral_gap = max(
        min_lane_width_route / 2.0
        - config.simulation.ego_extent_y / 2
        - safety_translation_perturbation_gap,
        0.1,
    )
    tmax = min(config.perturbation.camera_translation_perturbation_max, lateral_gap)

    # Pick perturbation translation shift
    perturbation_translation = np.random.choice([-1, 1]) * np.random.uniform(
        low=config.perturbation.camera_translation_perturbation_min,
        high=tmax,
    )

    # Next, pick rotation perturbation ranges, depending on translation perturbation.
    # Interpolate perturbation rotation depends on translation perturbation to avoid unrealistic configurations.
    neg_range = (
        -config.perturbation.camera_rotation_perturbation_min,
        config.perturbation.camera_rotation_perturbation_max,
    )  # for t <= -1
    pos_range = (
        -config.perturbation.camera_rotation_perturbation_max,
        config.perturbation.camera_rotation_perturbation_min,
    )  # for t >= +1
    if (
        perturbation_translation
        <= -config.perturbation.camera_translation_perturbation_max
    ):
        rmin, rmax = neg_range
    elif (
        perturbation_translation
        >= config.perturbation.camera_translation_perturbation_max
    ):
        rmin, rmax = pos_range
    else:
        alpha = (
            perturbation_translation
            + config.perturbation.camera_translation_perturbation_max
        ) / (
            2.0 * config.perturbation.camera_translation_perturbation_max
        )  # maps to [0,1]
        rmin = -(-neg_range[0] * (1 - alpha) + -pos_range[0] * alpha)
        rmax = neg_range[1] * (1 - alpha) + pos_range[1] * alpha

    beta = tmax / config.perturbation.camera_translation_perturbation_max  # in [0,1]
    rmin *= beta
    rmax *= beta

    # Rejection sampling of rotation perturbation to avoid useless small rotation angles
    perturbation_rotation = np.random.uniform(low=rmin, high=rmax)
    while abs(perturbation_rotation) < config.perturbation.camera_rotation_epsilon:
        perturbation_rotation = np.random.uniform(low=rmin, high=rmax)

    LOG.info(f"Translation perturbation range: [-{tmax:.2f}m, {tmax:.2f}m].")
    LOG.info(f"Sampled translation: {perturbation_translation:.2f}m.")
    LOG.info(
        f"rotation range: [{rmin:.2f}°, {rmax:.2f}°], sampled rotation: {perturbation_rotation:.2f}°",
    )

    return perturbation_translation, perturbation_rotation

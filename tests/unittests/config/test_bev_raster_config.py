"""Tests of the BEV raster knobs: meters in, pixels out."""

import pytest

from lead.config import load_lead_config


def transfuser(overrides: dict | None = None):
    """The TransFuser config section under ``policy.transfuser`` overrides.

    Args:
        overrides: ``policy.transfuser`` overrides to apply.

    Returns:
        The config section.
    """
    return load_lead_config(
        {"policy": {"transfuser": dict(overrides or {})}},
    ).policy.transfuser


def test_the_default_width_is_the_one_pixel_stroke_it_replaced():
    # 0.25 m at the default 4 px/m; the raster must not change under the rename.
    config = transfuser()
    assert config.bev_pixels_per_meter == 4.0
    assert config.lane_marker_thickness_pixel == 1


@pytest.mark.parametrize(
    ("pixels_per_meter", "expected_thickness"),
    [(4.0, 1), (8.0, 2), (16.0, 4)],
)
def test_the_stroke_holds_its_width_in_meters(pixels_per_meter, expected_thickness):
    """Doubling the resolution doubles the stroke, so the marking stays 0.25 m."""
    config = transfuser({"bev_pixels_per_meter": pixels_per_meter})
    assert config.lane_marker_thickness_pixel == expected_thickness


def test_a_wider_marking_draws_thicker():
    assert transfuser({"lane_marker_width_meter": 1.0}).lane_marker_thickness_pixel == 4


def test_a_marking_thinner_than_a_pixel_still_draws():
    # cv2 renders nothing at thickness 0, which would drop the class entirely.
    config = transfuser({"lane_marker_width_meter": 0.01})
    assert config.lane_marker_thickness_pixel == 1


def test_the_width_is_in_the_cache_finger_print():
    """It changes ``bev_semantic``, which is a cached tensor."""
    from lead.policy.transfuser.dataloader.dataset import _CACHE_FINGER_PRINT_FIELDS

    assert "lane_marker_width_meter" in _CACHE_FINGER_PRINT_FIELDS


def test_the_sweep_windows_are_in_the_cache_finger_print():
    """They change ``rasterized_lidar``, so a changed window must refuse the store."""
    from lead.policy.transfuser.dataloader.dataset import _CACHE_FINGER_PRINT_FIELDS

    assert "past_lidar_tick_ages" in _CACHE_FINGER_PRINT_FIELDS
    assert "past_radar_tick_ages" in _CACHE_FINGER_PRINT_FIELDS


def test_no_planning_knob_is_in_the_cache_finger_print():
    """Positions are never cached, so no planning knob may invalidate the store."""
    from lead.policy.transfuser.dataloader.dataset import _CACHE_FINGER_PRINT_FIELDS

    assert "future_ego_pose_iterations" not in _CACHE_FINGER_PRINT_FIELDS
    assert "num_route_points_smoothing" not in _CACHE_FINGER_PRINT_FIELDS

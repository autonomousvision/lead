import numpy as np
import pytest

from lead.config import load_lead_config
from lead.policy.transfuser.dataloader.features import rasterize_lidar_bev


@pytest.fixture
def config():
    """Fixture providing the config tree."""
    return load_lead_config()


class TestRasterizeLidar:
    """Tests for LiDAR point cloud rasterization."""

    def test_rasterize_empty_point_cloud(self, config):
        """Test rasterization with empty point cloud."""
        lidar = np.array([]).reshape(0, 3)
        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        # Should return a valid grid even with no points
        expected_height = int(
            (
                config.policy.transfuser.bev_max_y_meter
                - config.policy.transfuser.bev_min_y_meter
            )
            * config.policy.transfuser.bev_pixels_per_meter,
        )
        expected_width = int(
            (
                config.policy.transfuser.bev_max_x_meter
                - config.policy.transfuser.bev_min_x_meter
            )
            * config.policy.transfuser.bev_pixels_per_meter,
        )
        assert result.shape == (expected_height, expected_width)
        assert np.all(result == 0.0)

    def test_rasterize_single_point(self, config):
        """Test rasterization with single point in center."""
        # Point at origin, within height bounds
        lidar = np.array([[0.0, 0.0, 0.0]])
        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        expected_height = int(
            (
                config.policy.transfuser.bev_max_y_meter
                - config.policy.transfuser.bev_min_y_meter
            )
            * config.policy.transfuser.bev_pixels_per_meter,
        )
        expected_width = int(
            (
                config.policy.transfuser.bev_max_x_meter
                - config.policy.transfuser.bev_min_x_meter
            )
            * config.policy.transfuser.bev_pixels_per_meter,
        )
        assert result.shape == (expected_height, expected_width)
        # At least one bin should be non-zero
        assert np.sum(result) > 0.0

    def test_rasterize_multiple_points(self, config):
        """Test rasterization with multiple points."""
        # Create points scattered within bounds
        np.random.seed(42)
        n_points = 100
        x = np.random.uniform(
            config.policy.transfuser.bev_min_x_meter,
            config.policy.transfuser.bev_max_x_meter,
            n_points,
        )
        y = np.random.uniform(
            config.policy.transfuser.bev_min_y_meter,
            config.policy.transfuser.bev_max_y_meter,
            n_points,
        )
        z = np.random.uniform(
            config.policy.transfuser.lidar_min_height_meter,
            config.policy.transfuser.lidar_max_height_meter,
            n_points,
        )
        lidar = np.column_stack([x, y, z])

        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        # Should produce non-zero output
        assert np.sum(result) > 0.0
        # All values should be in [0, 1] range (normalized)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_height_filtering(self, config):
        """Test that points outside height bounds are filtered."""
        transfuser_config = config.policy.transfuser
        in_bounds = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 1.0]])
        out_of_bounds = np.array(
            [
                [1.0, 1.0, transfuser_config.lidar_max_height_meter + 1.0],
                [2.0, 2.0, transfuser_config.lidar_min_height_meter - 1.0],
            ],
        )

        result = rasterize_lidar_bev(
            config,
            np.vstack([in_bounds, out_of_bounds]),
            remove_ground_plane=False,
        )

        # The out-of-height points lie inside the x-y bounds, so they can only
        # vanish through the height filter.
        np.testing.assert_array_equal(
            result,
            rasterize_lidar_bev(config, in_bounds, remove_ground_plane=False),
        )
        assert np.sum(result) == pytest.approx(
            len(in_bounds) / transfuser_config.max_lidar_points_per_bev_pixel,
        )

    def test_histogram_saturation(self, config):
        """Test that histogram bins saturate at max_lidar_points_per_bev_pixel."""
        # Create many points in the same location to exceed max_lidar_points_per_bev_pixel
        n_points = config.policy.transfuser.max_lidar_points_per_bev_pixel * 3
        lidar = np.tile([[0.0, 0.0, 0.0]], (n_points, 1))

        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        # Maximum value should be clamped to 1.0 (normalized max_lidar_points_per_bev_pixel)
        assert np.max(result) <= 1.0

    def test_output_shape_matches_config(self, config):
        """Test that output shape matches configuration parameters."""
        lidar = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]])

        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        expected_height = int(
            (
                config.policy.transfuser.bev_max_y_meter
                - config.policy.transfuser.bev_min_y_meter
            )
            * config.policy.transfuser.bev_pixels_per_meter,
        )
        expected_width = int(
            (
                config.policy.transfuser.bev_max_x_meter
                - config.policy.transfuser.bev_min_x_meter
            )
            * config.policy.transfuser.bev_pixels_per_meter,
        )
        assert result.shape == (expected_height, expected_width)

    def test_output_dtype(self, config):
        """Test that output has correct dtype (float32)."""
        lidar = np.array([[0.0, 0.0, 0.0]])
        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        assert result.dtype == np.float32

    def test_points_outside_xy_bounds(self, config):
        """Test behavior with points outside x-y bounds."""
        # Create points outside the configured bounds
        lidar = np.array(
            [
                [
                    config.policy.transfuser.bev_max_x_meter + 10.0,
                    0.0,
                    0.0,
                ],  # Beyond x bound
                [
                    0.0,
                    config.policy.transfuser.bev_max_y_meter + 10.0,
                    0.0,
                ],  # Beyond y bound
                [
                    config.policy.transfuser.bev_min_x_meter - 10.0,
                    0.0,
                    0.0,
                ],  # Before x bound
                [
                    0.0,
                    config.policy.transfuser.bev_min_y_meter - 10.0,
                    0.0,
                ],  # Before y bound
            ],
        )

        # histogramdd drops points outside the bin edges: with every input point
        # out of bounds the grid must stay empty, not clamp them into edge bins.
        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)
        assert np.all(result == 0.0)

    def test_remove_ground_plane_option(self, config):
        """Test the flag strips most ground mass while keeping elevated points."""
        # Dense flat cloud: the RANSAC segments need >= 64 candidates each to fit.
        rng = np.random.default_rng(42)
        n_ground = 4096
        n_elevated = 512

        ground_points = np.column_stack(
            [
                rng.uniform(-20, 20, n_ground),
                rng.uniform(-20, 20, n_ground),
                rng.uniform(-0.05, 0.05, n_ground),
            ],
        )
        elevated_points = np.column_stack(
            [
                rng.uniform(-10, 10, n_elevated),
                rng.uniform(-10, 10, n_elevated),
                rng.uniform(1.5, 2.5, n_elevated),
            ],
        )
        lidar = np.vstack([ground_points, elevated_points])

        result_with_ground = rasterize_lidar_bev(
            config,
            lidar,
            remove_ground_plane=False,
        )
        result_without_ground = rasterize_lidar_bev(
            config,
            lidar,
            remove_ground_plane=True,
        )

        # Ground removal must strip most of the mass (ignoring the flag keeps
        # the masses equal) without touching the points far above the plane.
        assert np.sum(result_without_ground) < 0.5 * np.sum(result_with_ground)
        elevated_mass = np.sum(
            rasterize_lidar_bev(config, elevated_points, remove_ground_plane=False),
        )
        assert np.sum(result_without_ground) >= elevated_mass - 1e-6

    def test_coordinate_system_transpose(self, config):
        """Test that coordinate system is correctly transposed."""
        transfuser_config = config.policy.transfuser
        pixels_per_meter = transfuser_config.bev_pixels_per_meter
        num_x_bins = int(
            (transfuser_config.bev_max_x_meter - transfuser_config.bev_min_x_meter)
            * pixels_per_meter,
        )
        num_y_bins = int(
            (transfuser_config.bev_max_y_meter - transfuser_config.bev_min_y_meter)
            * pixels_per_meter,
        )
        # Asymmetric bin indices, otherwise a missing transpose is invisible.
        x_bin = num_x_bins // 3
        y_bin = (2 * num_y_bins) // 3
        assert x_bin != y_bin

        # A single point at the center of cell (x_bin, y_bin).
        x_coord = transfuser_config.bev_min_x_meter + (x_bin + 0.5) / pixels_per_meter
        y_coord = transfuser_config.bev_min_y_meter + (y_bin + 0.5) / pixels_per_meter
        lidar = np.array([[x_coord, y_coord, 0.0]])

        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        # CARLA is x-front, y-right while the image is row=y, column=x, so the
        # transposed grid must be hot at exactly [y_bin, x_bin].
        assert np.count_nonzero(result) == 1
        assert result[y_bin, x_bin] == pytest.approx(
            1.0 / transfuser_config.max_lidar_points_per_bev_pixel,
        )

    def test_normalization_range(self, config):
        """Test that output values are normalized between 0 and 1."""
        np.random.seed(42)
        n_points = 200
        lidar = np.column_stack(
            [
                np.random.uniform(
                    config.policy.transfuser.bev_min_x_meter,
                    config.policy.transfuser.bev_max_x_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.policy.transfuser.bev_min_y_meter,
                    config.policy.transfuser.bev_max_y_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.policy.transfuser.lidar_min_height_meter,
                    config.policy.transfuser.lidar_max_height_meter,
                    n_points,
                ),
            ],
        )

        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        # All values should be in [0, 1] range
        assert np.all(result >= 0.0), f"Found negative values: {result[result < 0.0]}"
        assert np.all(result <= 1.0), f"Found values > 1.0: {result[result > 1.0]}"

    def test_deterministic_output(self, config):
        """Test that same input produces same output (deterministic)."""
        lidar = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, -1.0, 1.0]])

        result1 = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)
        result2 = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        np.testing.assert_array_equal(result1, result2)

    def test_dense_point_cloud(self, config):
        """Test with a dense point cloud."""
        np.random.seed(42)
        n_points = 100000
        lidar = np.column_stack(
            [
                np.random.uniform(
                    config.policy.transfuser.bev_min_x_meter,
                    config.policy.transfuser.bev_max_x_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.policy.transfuser.bev_min_y_meter,
                    config.policy.transfuser.bev_max_y_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.policy.transfuser.lidar_min_height_meter,
                    config.policy.transfuser.lidar_max_height_meter,
                    n_points,
                ),
            ],
        )

        result = rasterize_lidar_bev(config, lidar, remove_ground_plane=False)

        # With dense point cloud, most bins should have some points
        non_zero_ratio = np.sum(result > 0) / result.size
        assert non_zero_ratio > 0.1  # At least 10% of bins should be filled
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

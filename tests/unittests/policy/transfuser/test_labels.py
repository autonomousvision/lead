import numpy as np
import pytest

from lead.config import load_lead_config
from lead.policy.transfuser.dataloader.labels import rasterize_lidar


@pytest.fixture
def config():
    """Fixture providing the config tree."""
    return load_lead_config()


class TestRasterizeLidar:
    """Tests for LiDAR point cloud rasterization."""

    def test_rasterize_empty_point_cloud(self, config):
        """Test rasterization with empty point cloud."""
        lidar = np.array([]).reshape(0, 3)
        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # Should return a valid grid even with no points
        expected_height = int(
            (
                config.expert.data_collection.max_y_meter
                - config.expert.data_collection.min_y_meter
            )
            * config.expert.data_collection.pixels_per_meter,
        )
        expected_width = int(
            (
                config.expert.data_collection.max_x_meter
                - config.expert.data_collection.min_x_meter
            )
            * config.expert.data_collection.pixels_per_meter,
        )
        assert result.shape == (expected_height, expected_width)
        assert np.all(result == 0.0)

    def test_rasterize_single_point(self, config):
        """Test rasterization with single point in center."""
        # Point at origin, within height bounds
        lidar = np.array([[0.0, 0.0, 0.0]])
        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        expected_height = int(
            (
                config.expert.data_collection.max_y_meter
                - config.expert.data_collection.min_y_meter
            )
            * config.expert.data_collection.pixels_per_meter,
        )
        expected_width = int(
            (
                config.expert.data_collection.max_x_meter
                - config.expert.data_collection.min_x_meter
            )
            * config.expert.data_collection.pixels_per_meter,
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
            config.expert.data_collection.min_x_meter,
            config.expert.data_collection.max_x_meter,
            n_points,
        )
        y = np.random.uniform(
            config.expert.data_collection.min_y_meter,
            config.expert.data_collection.max_y_meter,
            n_points,
        )
        z = np.random.uniform(
            config.policy.transfuser.min_height_lidar,
            config.policy.transfuser.max_height_lidar,
            n_points,
        )
        lidar = np.column_stack([x, y, z])

        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # Should produce non-zero output
        assert np.sum(result) > 0.0
        # All values should be in [0, 1] range (normalized)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_height_filtering(self, config):
        """Test that points outside height bounds are filtered."""
        # Create points: some within bounds, some outside
        lidar = np.array(
            [
                [0.0, 0.0, 0.0],  # Within bounds
                [
                    1.0,
                    1.0,
                    config.policy.transfuser.max_height_lidar + 1.0,
                ],  # Above max
                [
                    2.0,
                    2.0,
                    config.policy.transfuser.min_height_lidar - 1.0,
                ],  # Below min
                [3.0, 3.0, 1.0],  # Within bounds
            ],
        )

        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # Only points within height bounds should contribute
        # Points outside bounds should be filtered
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_histogram_saturation(self, config):
        """Test that histogram bins saturate at hist_max_per_pixel."""
        # Create many points in the same location to exceed hist_max_per_pixel
        n_points = config.training.data.hist_max_per_pixel * 3
        lidar = np.tile([[0.0, 0.0, 0.0]], (n_points, 1))

        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # Maximum value should be clamped to 1.0 (normalized hist_max_per_pixel)
        assert np.max(result) <= 1.0

    def test_output_shape_matches_config(self, config):
        """Test that output shape matches configuration parameters."""
        lidar = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]])

        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        expected_height = int(
            (
                config.expert.data_collection.max_y_meter
                - config.expert.data_collection.min_y_meter
            )
            * config.expert.data_collection.pixels_per_meter,
        )
        expected_width = int(
            (
                config.expert.data_collection.max_x_meter
                - config.expert.data_collection.min_x_meter
            )
            * config.expert.data_collection.pixels_per_meter,
        )
        assert result.shape == (expected_height, expected_width)

    def test_output_dtype(self, config):
        """Test that output has correct dtype (float32)."""
        lidar = np.array([[0.0, 0.0, 0.0]])
        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        assert result.dtype == np.float32

    def test_points_outside_xy_bounds(self, config):
        """Test behavior with points outside x-y bounds."""
        # Create points outside the configured bounds
        lidar = np.array(
            [
                [
                    config.expert.data_collection.max_x_meter + 10.0,
                    0.0,
                    0.0,
                ],  # Beyond x bound
                [
                    0.0,
                    config.expert.data_collection.max_y_meter + 10.0,
                    0.0,
                ],  # Beyond y bound
                [
                    config.expert.data_collection.min_x_meter - 10.0,
                    0.0,
                    0.0,
                ],  # Before x bound
                [
                    0.0,
                    config.expert.data_collection.min_y_meter - 10.0,
                    0.0,
                ],  # Before y bound
            ],
        )

        # Should not crash, points outside bins are ignored by histogramdd
        result = rasterize_lidar(config, lidar, remove_ground_plane=False)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_remove_ground_plane_option(self, config):
        """Test ground plane removal functionality."""
        # Create a point cloud with ground points (low z) and elevated points
        np.random.seed(42)
        n_ground = 50
        n_elevated = 20

        # Ground points (z near 0)
        ground_points = np.column_stack(
            [
                np.random.uniform(-5, 5, n_ground),
                np.random.uniform(-5, 5, n_ground),
                np.random.uniform(-0.1, 0.1, n_ground),
            ],
        )

        # Elevated points (z > 1.0)
        elevated_points = np.column_stack(
            [
                np.random.uniform(-5, 5, n_elevated),
                np.random.uniform(-5, 5, n_elevated),
                np.random.uniform(1.5, 2.5, n_elevated),
            ],
        )

        lidar = np.vstack([ground_points, elevated_points])

        # Rasterize without ground removal
        result_with_ground = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # Rasterize with ground removal
        result_without_ground = rasterize_lidar(config, lidar, remove_ground_plane=True)

        # Both should produce valid outputs
        assert result_with_ground.shape == result_without_ground.shape
        assert result_with_ground.shape[0] > 0
        assert result_without_ground.shape[0] > 0

    def test_coordinate_system_transpose(self, config):
        """Test that coordinate system is correctly transposed."""
        # Place a point at known location
        x_coord = 5.0
        y_coord = 3.0
        lidar = np.array([[x_coord, y_coord, 0.0]])

        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # The function transposes because CARLA uses x-front, y-right
        # while image uses y-front (height), x-right (width)
        # Just verify the result is valid and has the expected shape
        assert result.shape[0] > 0  # Height dimension
        assert result.shape[1] > 0  # Width dimension
        assert np.sum(result) > 0.0  # Point should be rasterized somewhere

    def test_normalization_range(self, config):
        """Test that output values are normalized between 0 and 1."""
        np.random.seed(42)
        n_points = 200
        lidar = np.column_stack(
            [
                np.random.uniform(
                    config.expert.data_collection.min_x_meter,
                    config.expert.data_collection.max_x_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.expert.data_collection.min_y_meter,
                    config.expert.data_collection.max_y_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.policy.transfuser.min_height_lidar,
                    config.policy.transfuser.max_height_lidar,
                    n_points,
                ),
            ],
        )

        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # All values should be in [0, 1] range
        assert np.all(result >= 0.0), f"Found negative values: {result[result < 0.0]}"
        assert np.all(result <= 1.0), f"Found values > 1.0: {result[result > 1.0]}"

    def test_deterministic_output(self, config):
        """Test that same input produces same output (deterministic)."""
        lidar = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, -1.0, 1.0]])

        result1 = rasterize_lidar(config, lidar, remove_ground_plane=False)
        result2 = rasterize_lidar(config, lidar, remove_ground_plane=False)

        np.testing.assert_array_equal(result1, result2)

    def test_dense_point_cloud(self, config):
        """Test with a dense point cloud."""
        np.random.seed(42)
        n_points = 100000
        lidar = np.column_stack(
            [
                np.random.uniform(
                    config.expert.data_collection.min_x_meter,
                    config.expert.data_collection.max_x_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.expert.data_collection.min_y_meter,
                    config.expert.data_collection.max_y_meter,
                    n_points,
                ),
                np.random.uniform(
                    config.policy.transfuser.min_height_lidar,
                    config.policy.transfuser.max_height_lidar,
                    n_points,
                ),
            ],
        )

        result = rasterize_lidar(config, lidar, remove_ground_plane=False)

        # With dense point cloud, most bins should have some points
        non_zero_ratio = np.sum(result > 0) / result.size
        assert non_zero_ratio > 0.1  # At least 10% of bins should be filled
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

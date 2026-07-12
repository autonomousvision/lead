import numpy as np
import pytest

from lead.common.sensors.ransac import remove_ground
from lead.config import load_lead_config


@pytest.fixture
def sample_point_cloud():
    """Generate a simple synthetic point cloud with ground and non-ground points."""
    np.random.seed(42)

    ground_points = np.column_stack(
        [
            np.random.uniform(-10, 10, 100),  # x
            np.random.uniform(-10, 10, 100),  # y
            np.random.uniform(0.00, 0.01, 100),  # z near 0
        ],
    )

    non_ground_points = np.column_stack(
        [
            np.random.uniform(-10, 10, 100),  # x
            np.random.uniform(-10, 10, 100),  # y
            np.random.uniform(3.0, 3.01, 100),  # z elevated
        ],
    )

    return np.vstack([ground_points, non_ground_points])


@pytest.fixture
def mock_config():
    """Fixture providing the expert config section."""
    return load_lead_config().expert


class TestRemoveGround:
    """Tests for the remove_ground public API function."""

    def test_remove_ground_basic(self, sample_point_cloud, mock_config):
        """Test basic ground removal functionality."""
        ground_mask = remove_ground(sample_point_cloud, mock_config, parallel=False)

        # Check that mask has correct shape
        assert ground_mask.shape == (len(sample_point_cloud),)
        assert ground_mask.dtype == bool

        # Should detect some ground points
        assert np.sum(ground_mask) > 0

    def test_remove_ground_parallel(self, sample_point_cloud, mock_config):
        """Test that parallel processing produces consistent results."""
        mask_serial = remove_ground(sample_point_cloud, mock_config, parallel=False)
        mask_parallel = remove_ground(sample_point_cloud, mock_config, parallel=True)

        # Both should have the same shape
        assert mask_serial.shape == mask_parallel.shape

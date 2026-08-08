import numpy as np
import pytest

from lead.common.sensors.ransac import (
    _fit_segment,
    _median_of_lowest,
    _percentile,
    generate_center_grid,
    remove_ground,
)
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
        ground_mask = remove_ground(sample_point_cloud, mock_config)

        # Check that mask has correct shape
        assert ground_mask.shape == (len(sample_point_cloud),)
        assert ground_mask.dtype == bool

        # Should detect some ground points
        assert np.sum(ground_mask) > 0

    def test_remove_ground_deterministic(self, sample_point_cloud, mock_config):
        """Test that repeated runs on the same cloud agree, despite the parallel centers."""
        first = remove_ground(sample_point_cloud, mock_config)
        second = remove_ground(sample_point_cloud, mock_config)

        assert np.array_equal(first, second)


class TestPercentile:
    """Tests for the hand-rolled percentile against numpy's reference."""

    @pytest.mark.parametrize("q", [0.0, 10.0, 25.0, 30.0, 50.0, 75.0, 99.0, 100.0])
    @pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 257])
    def test_matches_numpy_linear_percentile(self, q, size):
        """Test the partial-selection percentile equals np.percentile."""
        rng = np.random.default_rng(size * 1000 + int(q))
        values = rng.normal(size=size).astype(np.float64)
        assert np.isclose(
            _percentile(values, q),
            np.percentile(values, q, method="linear"),
        )

    def test_matches_numpy_on_duplicate_heavy_input(self):
        """Test ties do not perturb the interpolation."""
        values = np.array([1.0] * 10 + [2.0] * 10 + [3.0] * 5, dtype=np.float64)
        for q in (0.0, 25.0, 40.0, 50.0, 80.0, 100.0):
            assert np.isclose(
                _percentile(values, q),
                np.percentile(values, q, method="linear"),
            )

    def test_extremes_are_min_and_max(self):
        """Test q=0 and q=100 return the bounds."""
        values = np.array([5.0, -2.0, 9.0, 0.5], dtype=np.float64)
        assert np.isclose(_percentile(values, 0.0), -2.0)
        assert np.isclose(_percentile(values, 100.0), 9.0)

    def test_single_element(self):
        """Test a one-element array returns that element at any percentile."""
        values = np.array([3.25], dtype=np.float64)
        for q in (0.0, 50.0, 100.0):
            assert np.isclose(_percentile(values, q), 3.25)

    def test_does_not_mutate_input(self):
        """Test the partial sort does not reorder the caller's array."""
        values = np.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=np.float64)
        original = values.copy()
        _percentile(values, 30.0)
        assert np.array_equal(values, original)


class TestMedianOfLowest:
    """Tests for the median of the lowest N values."""

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 8, 9])
    def test_matches_sorted_reference(self, count):
        """Test the result equals the median of the smallest ``count`` values."""
        rng = np.random.default_rng(count)
        values = rng.normal(size=32).astype(np.float64)
        assert np.isclose(
            _median_of_lowest(values, count),
            np.median(np.sort(values)[:count]),
        )

    def test_count_of_one_is_the_minimum(self):
        """Test taking one value returns the array minimum."""
        values = np.array([4.0, -1.0, 7.0, 2.0], dtype=np.float64)
        assert np.isclose(_median_of_lowest(values, 1), -1.0)

    def test_count_equal_to_length_is_the_full_median(self):
        """Test using every value reduces to the plain median."""
        rng = np.random.default_rng(7)
        for size in (4, 5, 16, 17):
            values = rng.normal(size=size).astype(np.float64)
            assert np.isclose(
                _median_of_lowest(values, size),
                np.median(values),
            )

    def test_even_count_averages_the_two_middle_values(self):
        """Test the even branch interpolates between the middle pair."""
        values = np.array([10.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        # Lowest four are [1, 2, 3, 4]; median is (2 + 3) / 2.
        assert np.isclose(_median_of_lowest(values, 4), 2.5)

    def test_odd_count_takes_the_middle_value(self):
        """Test the odd branch selects the middle of the lowest values."""
        values = np.array([10.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        # Lowest three are [1, 2, 3]; median is 2.
        assert np.isclose(_median_of_lowest(values, 3), 2.0)

    def test_ignores_large_values(self):
        """Test values above the lowest ``count`` do not affect the result."""
        low = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with_outliers = np.concatenate([low, np.array([1e6, 1e7], dtype=np.float64)])
        assert np.isclose(
            _median_of_lowest(low, 3),
            _median_of_lowest(with_outliers, 3),
        )


class TestGenerateCenterGrid:
    """Tests for the ground-fitting center grid."""

    def test_origin_is_always_included(self):
        """Test the ego origin is appended to the grid."""
        grid = generate_center_grid(8.0, -20.0, 20.0, -20.0, 20.0)
        assert np.any(np.all(np.isclose(grid, [0.0, 0.0]), axis=1))

    def test_shape_is_pairs(self):
        """Test the grid is returned as (c, 2)."""
        grid = generate_center_grid(8.0, -20.0, 20.0, -20.0, 20.0)
        assert grid.ndim == 2
        assert grid.shape[1] == 2

    def test_spans_the_requested_bounds(self):
        """Test the grid reaches past both ends of the area."""
        grid = generate_center_grid(8.0, -20.0, 20.0, -16.0, 16.0)
        assert grid[:, 0].min() <= -20.0
        assert grid[:, 0].max() >= 20.0
        assert grid[:, 1].min() <= -16.0
        assert grid[:, 1].max() >= 16.0

    def test_finer_resolution_gives_more_centers(self):
        """Test halving the spacing increases the number of centers."""
        coarse = generate_center_grid(10.0, -20.0, 20.0, -20.0, 20.0)
        fine = generate_center_grid(5.0, -20.0, 20.0, -20.0, 20.0)
        assert len(fine) > len(coarse)

    def test_offset_uses_floor_division_on_a_float(self):
        """Test the start offset is ``resolution // 2``, which floors for odd values.

        With a resolution of 5.0 the offset is 2.0, not 2.5, because ``//`` is
        applied to a float. Pinned so the asymmetry stays deliberate.
        """
        grid = generate_center_grid(5.0, -10.0, 10.0, -10.0, 10.0)
        assert np.isclose(grid[:, 0].min(), -12.0)

    def test_even_resolution_offsets_by_exactly_half(self):
        """Test an even spacing offsets by half, where floor division is exact."""
        grid = generate_center_grid(8.0, -10.0, 10.0, -10.0, 10.0)
        assert np.isclose(grid[:, 0].min(), -14.0)


class TestFitSegment:
    """Tests for the iterative single-segment ground-plane fit."""

    def flat_ground_with_box(
        self,
        num_ground: int = 400,
        num_box: int = 120,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a flat ground plane at z=0 with a raised box of obstacle points.

        Args:
            num_ground: Number of ground points.
            num_box: Number of elevated points.

        Returns:
            The x, y, z and radial-distance arrays.
        """
        rng = np.random.default_rng(0)
        ground_xy = rng.uniform(-6.0, 6.0, size=(num_ground, 2))
        ground_z = rng.normal(0.0, 0.005, size=num_ground)
        box_xy = rng.uniform(2.0, 4.0, size=(num_box, 2))
        box_z = rng.uniform(1.5, 2.0, size=num_box)

        x = np.concatenate([ground_xy[:, 0], box_xy[:, 0]]).astype(np.float64)
        y = np.concatenate([ground_xy[:, 1], box_xy[:, 1]]).astype(np.float64)
        z = np.concatenate([ground_z, box_z]).astype(np.float64)
        distances = np.hypot(x, y)
        return x, y, z, distances

    def test_separates_ground_from_a_raised_box(self):
        """Test the fitted mask marks the plane and not the elevated points."""
        x, y, z, distances = self.flat_ground_with_box()
        ground = np.zeros(x.shape[0], dtype=np.bool_)
        _fit_segment(x, y, z, distances, ground, 3, 20, 0.4, 0.15, 10.0)

        num_ground = 400
        assert ground[:num_ground].mean() > 0.95
        assert ground[num_ground:].mean() < 0.05

    def test_returns_without_writing_when_too_few_points(self):
        """Test the early return leaves the mask untouched."""
        x = np.zeros(3, dtype=np.float64)
        y = np.zeros(3, dtype=np.float64)
        z = np.zeros(3, dtype=np.float64)
        distances = np.zeros(3, dtype=np.float64)
        ground = np.zeros(3, dtype=np.bool_)
        _fit_segment(x, y, z, distances, ground, 3, 20, 0.4, 0.15, 10.0)
        assert not ground.any()

    def test_returns_when_too_few_points_near_the_center(self):
        """Test a cloud entirely outside the center radius is left alone."""
        x, y, z, distances = self.flat_ground_with_box()
        ground = np.zeros(x.shape[0], dtype=np.bool_)
        _fit_segment(x, y, z, distances, ground, 3, 20, 0.4, 0.15, 0.001)
        assert not ground.any()

    def test_fits_a_tilted_plane(self):
        """Test a sloped ground plane is still recovered."""
        rng = np.random.default_rng(1)
        xy = rng.uniform(-6.0, 6.0, size=(500, 2))
        x = xy[:, 0].astype(np.float64)
        y = xy[:, 1].astype(np.float64)
        z = (0.05 * x + 0.02 * y + rng.normal(0.0, 0.005, size=500)).astype(np.float64)
        distances = np.hypot(x, y)
        ground = np.zeros(x.shape[0], dtype=np.bool_)
        _fit_segment(x, y, z, distances, ground, 5, 20, 0.4, 0.15, 10.0)
        assert ground.mean() > 0.95

import numpy as np
import pytest
import torch

from lead.common.geometry import angle_to_yaw_class
from lead.config import LeadConfig, load_lead_config
from lead.policy.transfuser.utils.ops import (
    bev_grid_sample,
    class2angle,
    gen_sineembed_for_position,
    normalize_imagenet,
    unit_normalize_bev_points,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@pytest.fixture
def lead_config():
    """Fixture providing the root config tree."""
    return load_lead_config()


@pytest.fixture
def bev_area(lead_config: LeadConfig) -> tuple[float, float, float, float]:
    """Fixture providing the BEV area bounds as (min_x, max_x, min_y, max_y).

    Args:
        lead_config: Root config tree.

    Returns:
        The BEV area bounds in meters.
    """
    data_config = lead_config.expert.data_collection
    return (
        float(data_config.min_x_meter),
        float(data_config.max_x_meter),
        float(data_config.min_y_meter),
        float(data_config.max_y_meter),
    )


class TestNormalizeImagenet:
    """Tests for ImageNet channel normalization."""

    def test_constant_image_matches_hand_computation(self):
        """Test each channel is normalized with its own mean and std."""
        image = torch.full((1, 3, 4, 5), 255.0)
        normalized = normalize_imagenet(image)
        for channel, (mean, std) in enumerate(
            zip(IMAGENET_MEAN, IMAGENET_STD, strict=True),
        ):
            assert torch.allclose(
                normalized[0, channel],
                torch.full((4, 5), (1.0 - mean) / std),
            )

    def test_zero_image_maps_to_negative_mean_over_std(self):
        """Test a black image maps to -mean/std per channel."""
        normalized = normalize_imagenet(torch.zeros((2, 3, 3, 3)))
        for channel, (mean, std) in enumerate(
            zip(IMAGENET_MEAN, IMAGENET_STD, strict=True),
        ):
            assert torch.allclose(
                normalized[:, channel],
                torch.full((2, 3, 3), -mean / std),
            )

    def test_does_not_mutate_input(self):
        """Test the caller's tensor is cloned before scaling."""
        image = torch.full((1, 3, 2, 2), 128.0)
        original = image.clone()
        normalize_imagenet(image)
        assert torch.equal(image, original)

    def test_preserves_shape(self):
        """Test the output keeps the input shape."""
        assert normalize_imagenet(torch.zeros((4, 3, 8, 16))).shape == (4, 3, 8, 16)

    def test_channels_are_scaled_independently(self):
        """Test a difference in one channel does not leak into the others."""
        image = torch.zeros((1, 3, 2, 2))
        image[:, 1] = 255.0
        normalized = normalize_imagenet(image)
        assert torch.allclose(
            normalized[0, 0],
            torch.full((2, 2), -IMAGENET_MEAN[0] / IMAGENET_STD[0]),
        )
        assert torch.allclose(
            normalized[0, 1],
            torch.full((2, 2), (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]),
        )


class TestGenSineembedForPosition:
    """Tests for the DAB-DETR sinusoidal position embedding."""

    def test_output_shape(self):
        """Test the embedding has one row per position and hidden_dim columns."""
        positions = torch.rand((5, 2))
        assert gen_sineembed_for_position(positions, hidden_dim=64).shape == (5, 64)

    @pytest.mark.parametrize("hidden_dim", [8, 32, 64, 128])
    def test_shape_follows_hidden_dim(self, hidden_dim):
        """Test the width tracks the requested hidden dimension."""
        positions = torch.rand((3, 2))
        embedded = gen_sineembed_for_position(positions, hidden_dim=hidden_dim)
        assert embedded.shape == (3, hidden_dim)

    def test_values_are_bounded(self):
        """Test the embedding is built from sines and cosines only."""
        embedded = gen_sineembed_for_position(torch.rand((16, 2)), hidden_dim=64)
        assert torch.all(embedded >= -1.0)
        assert torch.all(embedded <= 1.0)

    def test_is_deterministic(self):
        """Test the same input always embeds to the same value."""
        positions = torch.rand((4, 2))
        assert torch.equal(
            gen_sineembed_for_position(positions, hidden_dim=64),
            gen_sineembed_for_position(positions, hidden_dim=64),
        )

    def test_distinct_positions_embed_differently(self):
        """Test the embedding separates different positions."""
        positions = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        embedded = gen_sineembed_for_position(positions, hidden_dim=64)
        assert not torch.allclose(embedded[0], embedded[1])
        assert not torch.allclose(embedded[1], embedded[2])

    def test_accepts_the_unit_square_bounds(self):
        """Test the documented input range is accepted at both ends."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        assert torch.isfinite(gen_sineembed_for_position(positions)).all()

    @pytest.mark.parametrize("bad_value", [-0.01, 1.01])
    def test_rejects_positions_outside_the_unit_square(self, bad_value):
        """Test the range assertion fires for unnormalized input."""
        positions = torch.tensor([[bad_value, 0.5]])
        with pytest.raises(AssertionError):
            gen_sineembed_for_position(positions)

    def test_x_and_y_occupy_separate_halves(self):
        """Test the y embedding leads and the x embedding follows.

        The concatenation order is ``(pos_y, pos_x)``, so changing only x must
        leave the first half untouched.
        """
        base = torch.tensor([[0.25, 0.75]])
        changed_x = torch.tensor([[0.9, 0.75]])
        embedded_base = gen_sineembed_for_position(base, hidden_dim=64)
        embedded_changed = gen_sineembed_for_position(changed_x, hidden_dim=64)
        assert torch.allclose(embedded_base[:, :32], embedded_changed[:, :32])
        assert not torch.allclose(embedded_base[:, 32:], embedded_changed[:, 32:])


class TestClass2Angle:
    """Tests for decoding a discrete yaw class back to an angle."""

    def test_round_trip_with_angle_to_yaw_class(self, lead_config):
        """Test decoding inverts the encoder in lead.common.geometry."""
        num_yaw_bins = lead_config.policy.transfuser.num_yaw_bins
        for angle in np.linspace(0.0, 2 * np.pi, 73, endpoint=False):
            angle_cls, angle_res = angle_to_yaw_class(float(angle), num_yaw_bins)
            decoded = class2angle(
                torch.tensor([angle_cls]),
                torch.tensor([angle_res]),
                lead_config.policy.transfuser,
                limit_period=False,
            )
            assert np.isclose(float(decoded[0]) % (2 * np.pi), angle % (2 * np.pi))

    def test_zero_class_and_residual_is_zero_angle(self, lead_config):
        """Test the first bin center decodes to zero."""
        decoded = class2angle(
            torch.tensor([0]),
            torch.tensor([0.0]),
            lead_config.policy.transfuser,
        )
        assert np.isclose(float(decoded[0]), 0.0)

    def test_limit_period_wraps_large_angles(self, lead_config):
        """Test limit_period brings angles above π into the negative half."""
        num_yaw_bins = lead_config.policy.transfuser.num_yaw_bins
        last_class = num_yaw_bins - 1
        decoded = class2angle(
            torch.tensor([last_class]),
            torch.tensor([0.0]),
            lead_config.policy.transfuser,
            limit_period=True,
        )
        assert float(decoded[0]) <= np.pi

    def test_limit_period_disabled_leaves_angle_unwrapped(self, lead_config):
        """Test the raw decoding exceeds π when wrapping is off."""
        num_yaw_bins = lead_config.policy.transfuser.num_yaw_bins
        angle_per_class = 2 * np.pi / num_yaw_bins
        last_class = num_yaw_bins - 1
        decoded = class2angle(
            torch.tensor([last_class]),
            torch.tensor([0.0]),
            lead_config.policy.transfuser,
            limit_period=False,
        )
        assert np.isclose(float(decoded[0]), last_class * angle_per_class)

    def test_batched_input(self, lead_config):
        """Test a batch decodes elementwise."""
        classes = torch.tensor([0, 1, 2])
        residuals = torch.tensor([0.0, 0.0, 0.0])
        decoded = class2angle(
            classes,
            residuals,
            lead_config.policy.transfuser,
            limit_period=False,
        )
        angle_per_class = 2 * np.pi / lead_config.policy.transfuser.num_yaw_bins
        assert np.allclose(
            decoded.numpy(),
            [0.0, angle_per_class, 2 * angle_per_class],
        )

    def test_residual_shifts_the_bin_center(self, lead_config):
        """Test the residual is added on top of the class center."""
        decoded = class2angle(
            torch.tensor([1]),
            torch.tensor([0.05]),
            lead_config.policy.transfuser,
            limit_period=False,
        )
        angle_per_class = 2 * np.pi / lead_config.policy.transfuser.num_yaw_bins
        assert np.isclose(float(decoded[0]), angle_per_class + 0.05)


class TestUnitNormalizeBevPoints:
    """Tests for BEV point normalization to the unit square."""

    def test_corners_map_to_unit_square(self, lead_config, bev_area):
        """Test the BEV area corners map to exactly 0 and 1."""
        min_x, max_x, min_y, max_y = bev_area
        points = np.array([[min_x, min_y], [max_x, max_y]])
        normalized = unit_normalize_bev_points(points, lead_config)
        assert np.allclose(normalized, [[0.0, 0.0], [1.0, 1.0]])

    def test_center_maps_to_one_half(self, lead_config, bev_area):
        """Test the BEV area center maps to (0.5, 0.5)."""
        min_x, max_x, min_y, max_y = bev_area
        center = np.array([[(min_x + max_x) / 2, (min_y + max_y) / 2]])
        assert np.allclose(unit_normalize_bev_points(center, lead_config), [[0.5, 0.5]])

    def test_numpy_and_torch_paths_agree(self, lead_config, bev_area):
        """Test the two overloads produce identical values.

        They are separate code paths over the same arithmetic, so they can drift.
        """
        min_x, max_x, min_y, max_y = bev_area
        points = np.array(
            [[min_x, min_y], [max_x, max_y], [(min_x + max_x) / 2, min_y], [0.0, 0.0]],
        )
        from_numpy = unit_normalize_bev_points(points, lead_config)
        from_torch = unit_normalize_bev_points(
            torch.tensor(points, dtype=torch.float64),
            lead_config,
        )
        assert np.allclose(from_numpy, from_torch.numpy())

    def test_does_not_mutate_numpy_input(self, lead_config):
        """Test the caller's array is copied."""
        points = np.array([[1.0, 2.0]])
        original = points.copy()
        unit_normalize_bev_points(points, lead_config)
        assert np.allclose(points, original)

    def test_does_not_mutate_torch_input(self, lead_config):
        """Test the caller's tensor is cloned."""
        points = torch.tensor([[1.0, 2.0]])
        original = points.clone()
        unit_normalize_bev_points(points, lead_config)
        assert torch.equal(points, original)

    def test_is_monotone_in_each_axis(self, lead_config, bev_area):
        """Test normalization preserves ordering along both axes."""
        min_x, max_x, min_y, max_y = bev_area
        points = np.array(
            [
                [min_x, min_y],
                [(min_x + max_x) / 2, (min_y + max_y) / 2],
                [max_x, max_y],
            ],
        )
        normalized = unit_normalize_bev_points(points, lead_config)
        assert np.all(np.diff(normalized[:, 0]) > 0)
        assert np.all(np.diff(normalized[:, 1]) > 0)


class TestBevGridSample:
    """Tests for bilinear sampling of BEV features."""

    def test_constant_map_returns_the_constant(self, lead_config, bev_area):
        """Test sampling a uniform feature map is the identity on its value."""
        min_x, max_x, min_y, max_y = bev_area
        bev = torch.full((1, 3, 8, 8), 2.5)
        points = torch.tensor(
            [[[0.0, 0.0], [(min_x + max_x) / 2, (min_y + max_y) / 2]]],
        )
        sampled = bev_grid_sample(bev, points, lead_config)
        assert torch.allclose(sampled, torch.full_like(sampled, 2.5))

    def test_output_shape(self, lead_config):
        """Test the output is (B, N, D)."""
        bev = torch.zeros((2, 5, 16, 16))
        points = torch.zeros((2, 7, 2))
        assert bev_grid_sample(bev, points, lead_config).shape == (2, 7, 5)

    def test_samples_the_corner_value(self, lead_config, bev_area):
        """Test sampling at an area corner returns that corner's feature."""
        min_x, max_x, min_y, max_y = bev_area
        bev = torch.zeros((1, 1, 4, 4))
        # grid_sample with align_corners=True maps u=-1,v=-1 to element [0, 0].
        bev[0, 0, 0, 0] = 9.0
        points = torch.tensor([[[min_x, min_y]]], dtype=torch.float32)
        sampled = bev_grid_sample(bev, points, lead_config)
        assert torch.isclose(sampled[0, 0, 0], torch.tensor(9.0))

    def test_x_and_y_are_not_swapped(self, lead_config, bev_area):
        """Test the ego x/y to grid v/u mapping is the right way round.

        The feature map is built so that the value encodes the row index only;
        moving along ego x must change the sample, moving along ego y must not.
        """
        min_x, max_x, min_y, max_y = bev_area
        size = 8
        rows = torch.arange(size, dtype=torch.float32).view(1, 1, size, 1)
        bev = rows.expand(1, 1, size, size).contiguous()

        mid_y = (min_y + max_y) / 2
        near_x, far_x = min_x, max_x
        along_x = torch.tensor([[[near_x, mid_y], [far_x, mid_y]]], dtype=torch.float32)
        sampled_x = bev_grid_sample(bev, along_x, lead_config)
        assert not torch.isclose(sampled_x[0, 0, 0], sampled_x[0, 1, 0])

        mid_x = (min_x + max_x) / 2
        along_y = torch.tensor([[[mid_x, min_y], [mid_x, max_y]]], dtype=torch.float32)
        sampled_y = bev_grid_sample(bev, along_y, lead_config)
        assert torch.isclose(sampled_y[0, 0, 0], sampled_y[0, 1, 0])

    def test_interpolates_between_cells(self, lead_config, bev_area):
        """Test a midpoint between two cells returns their average."""
        min_x, max_x, min_y, max_y = bev_area
        bev = torch.zeros((1, 1, 2, 2))
        bev[0, 0, 0, :] = 0.0
        bev[0, 0, 1, :] = 10.0
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        points = torch.tensor([[[mid_x, mid_y]]], dtype=torch.float32)
        sampled = bev_grid_sample(bev, points, lead_config)
        assert torch.isclose(sampled[0, 0, 0], torch.tensor(5.0), atol=1e-5)

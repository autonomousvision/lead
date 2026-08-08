import numpy as np
import pytest
import torch

from lead.policy.transfuser.decoder.center_net_decoder import (
    gather_feat,
    gaussian2d,
    gaussian_focal_loss,
    gaussian_radius,
    gen_gaussian_target,
    get_local_maximum,
    get_topk_from_heatmap,
    transpose_and_gather_feat,
)


class TestGaussian2d:
    """Tests for the 2D gaussian kernel."""

    @pytest.mark.parametrize("radius", [1, 2, 5])
    def test_shape_is_diameter_squared(self, radius):
        """Test the kernel is (2r+1) x (2r+1)."""
        kernel = gaussian2d(radius)
        assert kernel.shape == (2 * radius + 1, 2 * radius + 1)

    def test_peak_is_at_the_center(self):
        """Test the maximum sits at the kernel center and equals 1."""
        radius = 4
        kernel = gaussian2d(radius, sigma=2.0)
        assert np.isclose(kernel[radius, radius], 1.0)
        assert kernel.argmax() == radius * (2 * radius + 1) + radius

    def test_matches_the_analytic_gaussian(self):
        """Test the values are exp(-(x^2 + y^2) / (2 sigma^2))."""
        radius, sigma = 3, 1.5
        kernel = gaussian2d(radius, sigma=sigma, dtype=np.float64)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        expected = np.exp(
            -(offsets.reshape(1, -1) ** 2 + offsets.reshape(-1, 1) ** 2)
            / (2 * sigma**2),
        )
        expected[expected < np.finfo(np.float64).eps * expected.max()] = 0
        assert np.allclose(kernel, expected)

    def test_is_symmetric(self):
        """Test the kernel is symmetric under both flips and transposition."""
        kernel = gaussian2d(4, sigma=1.5, dtype=np.float64)
        assert np.allclose(kernel, kernel[::-1, :])
        assert np.allclose(kernel, kernel[:, ::-1])
        assert np.allclose(kernel, kernel.T)

    def test_values_are_within_unit_range(self):
        """Test the kernel is bounded by 0 and 1."""
        kernel = gaussian2d(6, sigma=2.0)
        assert kernel.min() >= 0.0
        assert kernel.max() <= 1.0

    def test_decays_away_from_the_center(self):
        """Test values decrease monotonically along a center row."""
        radius = 5
        row = gaussian2d(radius, sigma=2.0, dtype=np.float64)[radius, radius:]
        assert np.all(np.diff(row) <= 0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_is_respected(self, dtype):
        """Test the requested dtype is used."""
        assert gaussian2d(3, dtype=dtype).dtype == dtype

    def test_tiny_values_are_flushed_to_zero(self):
        """Test the epsilon cutoff zeroes negligible tail values."""
        kernel = gaussian2d(10, sigma=0.5, dtype=np.float64)
        assert kernel[0, 0] == 0.0


class TestGenGaussianTarget:
    """Tests for splatting a gaussian into a heatmap."""

    def test_peak_lands_on_the_requested_center(self):
        """Test the splat peaks at the given (x, y)."""
        heatmap = np.zeros((20, 20), dtype=np.float32)
        gen_gaussian_target(heatmap, [7, 12], radius=3)
        assert np.isclose(heatmap[12, 7], 1.0)
        assert heatmap.argmax() == 12 * 20 + 7

    def test_writes_in_place_and_returns_the_same_array(self):
        """Test the input heatmap is mutated and handed back."""
        heatmap = np.zeros((10, 10), dtype=np.float32)
        result = gen_gaussian_target(heatmap, [5, 5], radius=2)
        assert result is heatmap
        assert heatmap.max() > 0.0

    def test_keeps_the_elementwise_maximum(self):
        """Test an existing larger value is not overwritten by the splat."""
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[5, 5] = 5.0
        gen_gaussian_target(heatmap, [5, 5], radius=2, k=1)
        assert np.isclose(heatmap[5, 5], 5.0)

    def test_k_scales_the_kernel(self):
        """Test the coefficient multiplies the splatted peak."""
        heatmap = np.zeros((10, 10), dtype=np.float32)
        gen_gaussian_target(heatmap, [5, 5], radius=2, k=3)
        assert np.isclose(heatmap[5, 5], 3.0)

    @pytest.mark.parametrize(
        "center",
        [[0, 0], [0, 9], [9, 0], [9, 9], [0, 5], [9, 5]],
    )
    def test_centers_on_the_border_are_clipped_safely(self, center):
        """Test splatting at an edge or corner stays in bounds and peaks there."""
        heatmap = np.zeros((10, 10), dtype=np.float32)
        gen_gaussian_target(heatmap, center, radius=4)
        x, y = center
        assert np.isclose(heatmap[y, x], 1.0)
        assert np.isfinite(heatmap).all()

    def test_two_splats_accumulate_by_maximum(self):
        """Test overlapping splats keep the larger value at each pixel."""
        first = np.zeros((20, 20), dtype=np.float32)
        gen_gaussian_target(first, [8, 8], radius=3)
        both = first.copy()
        gen_gaussian_target(both, [10, 8], radius=3)
        assert np.all(both >= first - 1e-6)
        assert np.isclose(both[8, 10], 1.0)

    def test_far_pixels_are_untouched(self):
        """Test the splat is local to its radius."""
        heatmap = np.zeros((30, 30), dtype=np.float32)
        gen_gaussian_target(heatmap, [5, 5], radius=2)
        assert np.isclose(heatmap[25, 25], 0.0)


class TestGaussianRadius:
    """Tests for the CornerNet gaussian radius heuristic."""

    def test_is_non_negative(self):
        """Test the radius is never negative for plausible boxes."""
        for height in (1.0, 4.0, 16.0, 64.0):
            for width in (1.0, 4.0, 16.0, 64.0):
                assert gaussian_radius([height, width], 0.7) >= 0

    def test_returns_an_int(self):
        """Test the result is truncated to an integer."""
        assert isinstance(gaussian_radius([12.0, 8.0], 0.7), int)

    def test_grows_with_box_size(self):
        """Test a larger box gets a larger or equal radius."""
        radii = [
            gaussian_radius([size, size], 0.7)
            for size in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
        ]
        assert all(
            later >= earlier for earlier, later in zip(radii, radii[1:], strict=False)
        )

    def test_shrinks_as_required_overlap_grows(self):
        """Test demanding more overlap permits a smaller radius."""
        radii = [gaussian_radius([32.0, 32.0], overlap) for overlap in (0.3, 0.5, 0.9)]
        assert all(
            later <= earlier for earlier, later in zip(radii, radii[1:], strict=False)
        )

    def test_is_symmetric_in_height_and_width(self):
        """Test swapping height and width does not change the radius."""
        assert gaussian_radius([12.0, 30.0], 0.7) == gaussian_radius([30.0, 12.0], 0.7)

    def test_degenerate_box_has_zero_radius(self):
        """Test a zero-sized box needs no spread."""
        assert gaussian_radius([0.0, 0.0], 0.7) == 0

    def test_full_overlap_requirement_gives_zero_radius(self):
        """Test requiring perfect overlap collapses the radius."""
        assert gaussian_radius([32.0, 32.0], 1.0) == 0

    @pytest.mark.parametrize("min_overlap", [0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize(
        "det_size",
        [(15.0, 23.0), (34.0, 17.0), (100.0, 3.0), (8.0, 8.0)],
    )
    def test_radius_keeps_the_required_overlap_and_is_maximal(
        self,
        det_size,
        min_overlap,
    ):
        """Test the radius satisfies the IoU bound while radius + 1 violates it."""
        height, width = det_size
        radius = gaussian_radius([height, width], min_overlap)
        assert _worst_case_iou(height, width, radius) >= min_overlap - 1e-9
        assert _worst_case_iou(height, width, radius + 1) < min_overlap


def _worst_case_iou(height: float, width: float, radius: float) -> float:
    """Worst IoU over the three corner displacements the radius must survive.

    Independent geometric oracle: a same-size box shifted by the radius along
    both axes, a box with both corners moved inward, and one moved outward.

    Args:
        height: Box height.
        width: Box width.
        radius: Corner displacement in pixels.

    Returns:
        The smallest IoU across the shifted, shrunk and grown cases.
    """
    area = height * width
    shifted_intersection = max(height - radius, 0.0) * max(width - radius, 0.0)
    shifted = shifted_intersection / (2.0 * area - shifted_intersection)
    shrunk = max(height - 2 * radius, 0.0) * max(width - 2 * radius, 0.0) / area
    grown = area / ((height + 2 * radius) * (width + 2 * radius))
    return min(shifted, shrunk, grown)


class TestGetLocalMaximum:
    """Tests for non-maximum suppression on a heatmap."""

    def test_isolated_peak_survives(self):
        """Test a lone peak keeps its value."""
        heat = torch.zeros((1, 1, 7, 7))
        heat[0, 0, 3, 3] = 0.9
        assert torch.isclose(get_local_maximum(heat)[0, 0, 3, 3], torch.tensor(0.9))

    def test_neighbours_of_a_peak_are_suppressed(self):
        """Test smaller values next to a peak are zeroed."""
        heat = torch.zeros((1, 1, 7, 7))
        heat[0, 0, 3, 3] = 0.9
        heat[0, 0, 3, 4] = 0.5
        assert torch.isclose(get_local_maximum(heat)[0, 0, 3, 4], torch.tensor(0.0))

    def test_separated_peaks_both_survive(self):
        """Test two peaks further apart than the kernel both remain."""
        heat = torch.zeros((1, 1, 9, 9))
        heat[0, 0, 1, 1] = 0.8
        heat[0, 0, 7, 7] = 0.6
        suppressed = get_local_maximum(heat)
        assert torch.isclose(suppressed[0, 0, 1, 1], torch.tensor(0.8))
        assert torch.isclose(suppressed[0, 0, 7, 7], torch.tensor(0.6))

    def test_preserves_shape(self):
        """Test the output keeps the input shape."""
        heat = torch.rand((2, 3, 12, 10))
        assert get_local_maximum(heat).shape == heat.shape

    def test_larger_kernel_suppresses_more(self):
        """Test a wider kernel keeps no more peaks than a narrow one."""
        torch.manual_seed(0)
        heat = torch.rand((1, 1, 16, 16))
        kept_small = (get_local_maximum(heat, kernel=3) > 0).sum()
        kept_large = (get_local_maximum(heat, kernel=7) > 0).sum()
        assert kept_large <= kept_small

    def test_never_increases_a_value(self):
        """Test suppression only ever zeroes entries."""
        torch.manual_seed(1)
        heat = torch.rand((1, 2, 10, 10))
        suppressed = get_local_maximum(heat)
        assert torch.all((suppressed == heat) | (suppressed == 0))

    def test_constant_map_is_entirely_kept(self):
        """Test a flat heatmap has every pixel equal to the pooled maximum."""
        heat = torch.full((1, 1, 5, 5), 0.3)
        assert torch.allclose(get_local_maximum(heat), heat)


class TestGetTopkFromHeatmap:
    """Tests for the top-k decoding of a heatmap."""

    def test_recovers_known_peak_locations(self):
        """Test the returned class and coordinates identify the planted peaks."""
        scores = torch.zeros((1, 3, 5, 7))
        scores[0, 0, 1, 2] = 0.9
        scores[0, 2, 4, 6] = 0.8
        scores[0, 1, 0, 0] = 0.7

        topk_scores, topk_inds, topk_clses, topk_ys, topk_xs = get_topk_from_heatmap(
            scores,
            k=3,
        )
        assert torch.allclose(topk_scores[0], torch.tensor([0.9, 0.8, 0.7]))
        assert torch.equal(topk_clses[0], torch.tensor([0, 2, 1]))
        assert torch.equal(topk_ys[0], torch.tensor([1, 4, 0]))
        assert torch.allclose(topk_xs[0], torch.tensor([2.0, 6.0, 0.0]))
        assert torch.equal(topk_inds[0], torch.tensor([1 * 7 + 2, 4 * 7 + 6, 0]))

    def test_scores_are_sorted_descending(self):
        """Test the k results come back in decreasing score order."""
        torch.manual_seed(0)
        scores = torch.rand((2, 4, 6, 6))
        topk_scores, _, _, _, _ = get_topk_from_heatmap(scores, k=10)
        assert torch.all(topk_scores[:, :-1] >= topk_scores[:, 1:])

    def test_index_decomposition_is_consistent(self):
        """Test ind, class, y and x describe the same flattened position."""
        torch.manual_seed(1)
        scores = torch.rand((2, 3, 5, 7))
        topk_scores, topk_inds, topk_clses, topk_ys, topk_xs = get_topk_from_heatmap(
            scores,
            k=8,
        )
        width = 7
        assert torch.equal(topk_inds, topk_ys.long() * width + topk_xs.long())
        for batch in range(scores.shape[0]):
            for slot in range(8):
                expected = scores[
                    batch,
                    topk_clses[batch, slot].long(),
                    topk_ys[batch, slot].long(),
                    topk_xs[batch, slot].long(),
                ]
                assert torch.isclose(topk_scores[batch, slot], expected)

    def test_shapes(self):
        """Test every output is (batch, k)."""
        scores = torch.rand((3, 2, 4, 4))
        for output in get_topk_from_heatmap(scores, k=5):
            assert output.shape == (3, 5)

    def test_index_is_within_the_spatial_plane(self):
        """Test the flattened index is taken modulo height * width."""
        scores = torch.rand((1, 5, 4, 6))
        _, topk_inds, _, _, _ = get_topk_from_heatmap(scores, k=12)
        assert torch.all(topk_inds < 4 * 6)

    def test_batches_are_decoded_independently(self):
        """Test one batch entry's peaks do not leak into another's."""
        scores = torch.zeros((2, 1, 4, 4))
        scores[0, 0, 0, 1] = 0.9
        scores[1, 0, 3, 2] = 0.9
        _, _, _, topk_ys, topk_xs = get_topk_from_heatmap(scores, k=1)
        assert topk_ys[0, 0] == 0 and topk_xs[0, 0] == 1
        assert topk_ys[1, 0] == 3 and topk_xs[1, 0] == 2


class TestGatherFeat:
    """Tests for index-based feature gathering."""

    def test_identity_index_returns_the_input(self):
        """Test gathering with arange is a no-op."""
        feat = torch.rand((2, 6, 4))
        ind = torch.arange(6).unsqueeze(0).repeat(2, 1)
        assert torch.allclose(gather_feat(feat, ind), feat)

    def test_selects_the_requested_rows(self):
        """Test the gathered rows match direct indexing."""
        feat = torch.rand((1, 8, 3))
        ind = torch.tensor([[5, 2, 7]])
        assert torch.allclose(gather_feat(feat, ind), feat[:, [5, 2, 7], :])

    def test_output_shape(self):
        """Test the output is (batch, num_indices, dim)."""
        assert gather_feat(
            torch.rand((3, 10, 5)),
            torch.zeros((3, 4)).long(),
        ).shape == (
            3,
            4,
            5,
        )

    def test_mask_flattens_selected_entries(self):
        """Test a boolean mask reduces the result to the kept rows."""
        feat = torch.rand((1, 5, 2))
        ind = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[True, False, True]])
        gathered = gather_feat(feat, ind, mask=mask)
        assert gathered.shape == (2, 2)
        assert torch.allclose(gathered, feat[0, [0, 2], :])

    def test_repeated_indices_are_allowed(self):
        """Test the same index can be gathered more than once."""
        feat = torch.rand((1, 4, 3))
        gathered = gather_feat(feat, torch.tensor([[2, 2, 2]]))
        assert torch.allclose(gathered[0, 0], gathered[0, 1])
        assert torch.allclose(gathered[0, 0], feat[0, 2])


class TestTransposeAndGatherFeat:
    """Tests for gathering from a channel-first feature map."""

    def test_output_shape(self):
        """Test the output is (batch, num_indices, channels)."""
        feat = torch.rand((2, 5, 4, 6))
        ind = torch.zeros((2, 7)).long()
        assert transpose_and_gather_feat(feat, ind).shape == (2, 7, 5)

    def test_matches_manual_channel_extraction(self):
        """Test gathering at a flattened position returns that pixel's channels."""
        feat = torch.rand((1, 3, 4, 6))
        row, col = 2, 5
        flat_index = row * 6 + col
        gathered = transpose_and_gather_feat(feat, torch.tensor([[flat_index]]))
        assert torch.allclose(gathered[0, 0], feat[0, :, row, col])

    def test_agrees_with_gather_feat_on_a_prepermuted_map(self):
        """Test it equals permuting and flattening by hand, then gathering."""
        feat = torch.rand((2, 3, 4, 5))
        ind = torch.tensor([[0, 7, 19], [3, 11, 4]])
        manual = feat.permute(0, 2, 3, 1).contiguous().view(2, -1, 3)
        assert torch.allclose(
            transpose_and_gather_feat(feat, ind),
            gather_feat(manual, ind),
        )

    def test_does_not_mutate_input(self):
        """Test the caller's feature map is left unchanged."""
        feat = torch.rand((1, 2, 3, 3))
        original = feat.clone()
        transpose_and_gather_feat(feat, torch.tensor([[0, 4]]))
        assert torch.equal(feat, original)


class TestGaussianFocalLoss:
    """Tests for the CornerNet gaussian focal loss."""

    def test_perfect_prediction_has_near_zero_loss(self):
        """Test a prediction matching the target costs almost nothing."""
        target = torch.zeros((1, 1, 5, 5))
        target[0, 0, 2, 2] = 1.0
        pred = target.clone().clamp(1e-6, 1 - 1e-6)
        assert gaussian_focal_loss(pred, target) < 1e-3

    def test_confident_and_wrong_costs_more_than_confident_and_right(self):
        """Test the loss ranks a wrong confident prediction worse."""
        target = torch.zeros((1, 1, 3, 3))
        target[0, 0, 1, 1] = 1.0
        right = torch.full((1, 1, 3, 3), 0.01)
        right[0, 0, 1, 1] = 0.99
        wrong = torch.full((1, 1, 3, 3), 0.01)
        wrong[0, 0, 1, 1] = 0.01
        assert gaussian_focal_loss(wrong, target) > gaussian_focal_loss(right, target)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reduction_shapes(self, reduction):
        """Test each reduction mode returns the documented shape."""
        pred = torch.full((2, 1, 3, 3), 0.5)
        target = torch.zeros((2, 1, 3, 3))
        loss = gaussian_focal_loss(pred, target, reduction=reduction)
        if reduction == "none":
            assert loss.shape == (2, 1, 3, 3)
        else:
            assert loss.shape == ()

    def test_sum_equals_mean_times_element_count(self):
        """Test the two reductions are consistent with each other."""
        torch.manual_seed(0)
        pred = torch.rand((2, 1, 4, 4)).clamp(1e-4, 1 - 1e-4)
        target = torch.zeros((2, 1, 4, 4))
        target[0, 0, 1, 1] = 1.0
        mean = gaussian_focal_loss(pred, target, reduction="mean")
        total = gaussian_focal_loss(pred, target, reduction="sum")
        assert torch.isclose(total, mean * pred.numel(), rtol=1e-5)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_is_finite_at_the_probability_bounds(self, value):
        """Test the eps guard keeps the log finite at 0 and 1."""
        pred = torch.full((1, 1, 2, 2), value)
        target = torch.zeros((1, 1, 2, 2))
        target[0, 0, 0, 0] = 1.0
        assert torch.isfinite(gaussian_focal_loss(pred, target))

    def test_only_exact_ones_count_as_positive(self):
        """Test the positive weighting keys off target == 1, not target > 0."""
        pred = torch.full((1, 1, 2, 2), 0.5)
        near_one = torch.full((1, 1, 2, 2), 0.999)
        exactly_one = torch.ones((1, 1, 2, 2))
        assert not torch.isclose(
            gaussian_focal_loss(pred, near_one),
            gaussian_focal_loss(pred, exactly_one),
        )

    def test_loss_is_non_negative(self):
        """Test the loss never goes below zero."""
        torch.manual_seed(2)
        pred = torch.rand((2, 3, 5, 5)).clamp(1e-4, 1 - 1e-4)
        target = torch.rand((2, 3, 5, 5))
        assert gaussian_focal_loss(pred, target, reduction="none").min() >= 0.0

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from math import sqrt

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp.autocast_mode import autocast

import lead.common.geometry as geometry
from lead.config import LeadConfig
from lead.policy.transfuser import ops, precision
from lead.policy.transfuser.dataloader import labels
from lead.policy.transfuser.labels import BoundingBoxIndex


class CenterNetDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lead_config: LeadConfig,
        device: torch.device,
    ) -> None:
        """Center Net Head implementation adapted from MM Detection
        Args:
            num_classes: Number of classes to predict.
            lead_config: Root config tree.
            device: Device to run the model on.
        """
        super().__init__()
        self.device = device
        self.lead_config = lead_config
        config = lead_config.policy.transfuser
        self.config = config
        self.num_classes = num_classes

        self.heatmap_head: nn.Sequential = self._build_head(
            config.bb_input_channel,
            num_classes,
        )
        self.wh_head: nn.Sequential = self._build_head(config.bb_input_channel, 2)
        self.offset_head: nn.Sequential = self._build_head(config.bb_input_channel, 2)
        self.yaw_class_head: nn.Sequential = self._build_head(
            config.bb_input_channel,
            config.num_dir_bins,
        )
        self.yaw_res_head: nn.Sequential = self._build_head(config.bb_input_channel, 1)
        if lead_config.training.data.training_used_lidar_steps > 1:
            self.velocity_head: nn.Sequential = self._build_head(
                config.bb_input_channel,
                1,
            )

    def _build_head(self, in_channel: int, out_channel: int) -> nn.Sequential:
        """Build head for each branch.
        Args:
            in_channel: Number of input channels.
            out_channel: Number of output channels.
        Returns:
            Head network.
        """
        layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )
        return layer

    def forward(
        self,
        data: dict,
        bev_feature_grid: torch.Tensor,
        log: dict,
    ) -> CenterNetBoundingBoxPrediction:
        """
        Forward feature of a single level.

        Args:
            data: Data dictionary containing valid labels mask.
            bev_feature_grid: Feature of static lidar BEV grid.
            log: Dictionary to store debug messages.
        Returns:
            Object containing all predictions with proper shapes.
        """
        center_heatmap_pred: torch.Tensor = self.heatmap_head(bev_feature_grid)
        wh_pred: torch.Tensor = self.wh_head(bev_feature_grid)
        offset_pred: torch.Tensor = self.offset_head(bev_feature_grid)
        yaw_class_pred: torch.Tensor = self.yaw_class_head(bev_feature_grid)
        yaw_res_pred: torch.Tensor = self.yaw_res_head(bev_feature_grid)
        velocity_pred: torch.Tensor | None = None
        if self.lead_config.training.data.training_used_lidar_steps > 1:
            velocity_pred = self.velocity_head(bev_feature_grid)

        return CenterNetBoundingBoxPrediction(
            center_heatmap_logit_pred=center_heatmap_pred,
            center_heatmap_pred=center_heatmap_pred.float().sigmoid(),
            wh_pred=wh_pred,
            offset_pred=offset_pred,
            yaw_class_pred=yaw_class_pred,
            yaw_res_pred=yaw_res_pred,
            velocity_pred=velocity_pred,
            lead_config=self.lead_config,
        )

    def compute_loss(
        self,
        data: dict,
        bounding_box_features: CenterNetBoundingBoxPrediction,
        losses: dict,
        log: dict,
    ):
        """
        Compute bounding box prediction losses and metrics.
        Args:
            data: Data dictionary containing ground truth labels and masks.
            bounding_box_features: Bounding box predictions.
            losses: Dictionary to store computed losses.
            log: Dictionary to store debug messages.
        """
        center_heatmap_target: torch.Tensor = data["center_net_heatmap"].to(
            self.device,
            dtype=self.lead_config.training.optimization.torch_float_type,
            non_blocking=True,
        )

        wh_target: torch.Tensor = data["center_net_wh"].to(
            self.device,
            dtype=self.lead_config.training.optimization.torch_float_type,
            non_blocking=True,
        )

        yaw_class_target: torch.Tensor = data["center_net_yaw_class"].to(
            self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        yaw_res_target: torch.Tensor = data["center_net_yaw_res"].to(
            self.device,
            dtype=self.lead_config.training.optimization.torch_float_type,
            non_blocking=True,
        )

        offset_target: torch.Tensor = data["center_net_offset"].to(
            self.device,
            dtype=self.lead_config.training.optimization.torch_float_type,
            non_blocking=True,
        )

        velocity_target: torch.Tensor = data["center_net_velocity"].to(
            self.device,
            dtype=self.lead_config.training.optimization.torch_float_type,
            non_blocking=True,
        )

        pixel_weight: torch.Tensor = data["center_net_pixel_weight"].to(
            self.device,
            dtype=self.lead_config.training.optimization.torch_float_type,
            non_blocking=True,
        )  # [bs, 2, h, w]

        # avg_factor is the number of valid bounding boxes in the batch: losses use
        # sum reduction divided by it so pixels without a box (zeroed by pixel_weight)
        # have no impact. A small epsilon keeps this stable when there are no boxes.
        avg_factor = data["center_net_avg_factor"].to(
            self.device,
            dtype=self.lead_config.training.optimization.torch_float_type,
            non_blocking=True,
        )  # (B,)

        with autocast(device_type="cuda", enabled=False):
            # Compute per-sample losses for heatmap
            loss_center_heatmap_per_sample = gaussian_focal_loss(
                pred=bounding_box_features.center_heatmap_pred,
                gaussian_target=center_heatmap_target,
                reduction="none",
            )  # (B, C, H, W)
            loss_center_heatmap_per_sample = loss_center_heatmap_per_sample.sum(
                dim=(1, 2, 3),
            )  # (B,)
            avg_factor_clamped = (
                avg_factor
                + torch.finfo(
                    self.lead_config.training.optimization.torch_float_type,
                ).eps
            )
            loss_center_heatmap = (
                loss_center_heatmap_per_sample / avg_factor_clamped
            ).mean()

            # Compute per-sample losses for wh
            loss_wh_per_sample = (
                F.l1_loss(
                    bounding_box_features.wh_pred.float(),
                    wh_target.float(),
                    reduction="none",
                )
                * pixel_weight.float()
            )  # (B, 2, H, W)
            loss_wh_per_sample = loss_wh_per_sample.sum(dim=(1, 2, 3))  # (B,)
            loss_wh = (
                loss_wh_per_sample
                / (avg_factor_clamped * bounding_box_features.wh_pred.shape[1])
            ).mean()

            # Compute per-sample losses for offset
            loss_offset_per_sample = (
                F.l1_loss(
                    bounding_box_features.offset_pred.float(),
                    offset_target.float(),
                    reduction="none",
                )
                * pixel_weight.float()
            )  # (B, 2, H, W)
            loss_offset_per_sample = loss_offset_per_sample.sum(dim=(1, 2, 3))  # (B,)
            loss_offset = (
                loss_offset_per_sample
                / (avg_factor_clamped * bounding_box_features.wh_pred.shape[1])
            ).mean()

            # Compute per-sample losses for yaw class
            loss_yaw_class_per_sample = (
                F.cross_entropy(
                    bounding_box_features.yaw_class_pred.float(),
                    yaw_class_target,
                    reduction="none",
                )
                * pixel_weight[:, 0].float()
            )  # (B, H, W)
            loss_yaw_class_per_sample = loss_yaw_class_per_sample.sum(
                dim=(1, 2),
            )  # (B,)
            loss_yaw_class = (loss_yaw_class_per_sample / avg_factor_clamped).mean()

            # Compute per-sample losses for yaw res
            loss_yaw_res_per_sample = (
                F.smooth_l1_loss(
                    bounding_box_features.yaw_res_pred.float(),
                    yaw_res_target.float(),
                    reduction="none",
                )
                * pixel_weight[:, 0:1].float()
            )  # (B, 1, H, W)
            loss_yaw_res_per_sample = loss_yaw_res_per_sample.sum(dim=(1, 2, 3))  # (B,)
            loss_yaw_res = (loss_yaw_res_per_sample / avg_factor_clamped).mean()

        loss_velocity = torch.zeros(
            1,
            dtype=self.lead_config.training.optimization.torch_float_type,
            device=self.device,
        )
        if self.lead_config.training.data.training_used_lidar_steps > 1:
            assert bounding_box_features.velocity_pred is not None
            loss_velocity_per_sample = (
                F.l1_loss(
                    bounding_box_features.velocity_pred,
                    velocity_target,
                    reduction="none",
                )
                * pixel_weight[:, 0:1]
            )  # (B, 1, H, W)
            loss_velocity_per_sample = loss_velocity_per_sample.sum(
                dim=(1, 2, 3),
            )  # (B,)
            loss_velocity = (loss_velocity_per_sample / avg_factor_clamped).mean()

        losses.update(
            {
                "loss_center_net_heatmap": loss_center_heatmap,
                "loss_center_net_wh": loss_wh,
                "loss_center_net_offset": loss_offset,
                "loss_center_net_yaw_class": loss_yaw_class,
                "loss_center_net_yaw_res": loss_yaw_res,
                "loss_center_net_velocity": loss_velocity,
            },
        )

        if (
            "iteration" in data
            and (
                (data["iteration"] + 1)
                % self.lead_config.training.experiment.log_scalars_frequency
            )
            == 0
        ):
            heatmap_pred = bounding_box_features.center_heatmap_pred
            wh_pred = bounding_box_features.wh_pred
            offset_pred = bounding_box_features.offset_pred
            yaw_class_pred = bounding_box_features.yaw_class_pred
            yaw_res_pred = bounding_box_features.yaw_res_pred
            log["center_net_output/heatmap_pred_min"] = heatmap_pred.min().item()
            log["center_net_output/heatmap_pred_max"] = heatmap_pred.max().item()
            log["center_net_output/wh_pred_min"] = wh_pred.min().item()
            log["center_net_output/wh_pred_max"] = wh_pred.max().item()
            log["center_net_output/offset_pred_min"] = offset_pred.min().item()
            log["center_net_output/offset_pred_max"] = offset_pred.max().item()
            log["center_net_output/yaw_class_pred_min"] = yaw_class_pred.min().item()
            log["center_net_output/yaw_class_pred_max"] = yaw_class_pred.max().item()
            log["center_net_output/yaw_res_pred_min"] = yaw_res_pred.min().item()
            log["center_net_output/yaw_res_pred_max"] = yaw_res_pred.max().item()
            if self.lead_config.training.data.training_used_lidar_steps > 1:
                velocity_pred = bounding_box_features.velocity_pred
                assert velocity_pred is not None
                log["center_net_output/velocity_pred_min"] = velocity_pred.min().item()
                log["center_net_output/velocity_pred_max"] = velocity_pred.max().item()


@dataclass
class CenterNetBoundingBoxPrediction:
    """Output features of the CenterNet head."""

    center_heatmap_logit_pred: torch.Tensor
    center_heatmap_pred: torch.Tensor
    wh_pred: torch.Tensor
    offset_pred: torch.Tensor
    yaw_class_pred: torch.Tensor
    yaw_res_pred: torch.Tensor
    velocity_pred: torch.Tensor | None
    lead_config: LeadConfig

    @cached_property
    def pred_bounding_box_image_system(self) -> jt.Float[npt.NDArray, "B K 9"]:
        """Numpy array of shape (bs, k, 9) with features (x, y, w, h, yaw, velocity, brake, class, score) in image system"""
        config = self.lead_config.policy.transfuser
        data_config = self.lead_config.expert.data_collection
        k = config.top_k_center_keypoints
        kernel = config.center_net_max_pooling_kernel

        center_heatmap_pred = get_local_maximum(self.center_heatmap_pred, kernel=kernel)

        batch_scores, batch_index, batch_topk_classes, topk_ys, topk_xs = (
            get_topk_from_heatmap(center_heatmap_pred, k=k)
        )

        wh = transpose_and_gather_feat(self.wh_pred, batch_index)
        offset = transpose_and_gather_feat(self.offset_pred, batch_index)
        yaw_class = transpose_and_gather_feat(self.yaw_class_pred, batch_index)
        yaw_res = transpose_and_gather_feat(self.yaw_res_pred, batch_index)

        # convert class + res to yaw
        yaw_class = torch.argmax(yaw_class, -1)
        yaw = ops.class2angle(yaw_class, yaw_res.squeeze(2), config)

        brake = torch.zeros_like(
            yaw,
        )  # We don't predict brake but keep it for now to avoid refactoring

        if self.lead_config.training.data.training_used_lidar_steps <= 1:
            velocity = torch.zeros_like(yaw)
        else:
            assert self.velocity_pred is not None
            velocity = transpose_and_gather_feat(self.velocity_pred, batch_index)
            velocity = velocity[..., 0]

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        batch_bboxes = torch.stack(
            [topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity, brake],
            dim=2,
        )
        batch_bboxes = torch.cat(
            (
                batch_bboxes,
                batch_topk_classes[..., np.newaxis],
                batch_scores[..., np.newaxis],
            ),
            dim=-1,
        )
        batch_bboxes[:, :, : BoundingBoxIndex.YAW] *= data_config.pixels_per_meter

        return batch_bboxes.detach().cpu().float().numpy()

    @cached_property
    def pred_bounding_box_vehicle_system(self) -> jt.Float[npt.NDArray, "B K 9"]:
        """Numpy array of shape (bs, k, 9) with features (x, y, w, h, yaw, velocity, brake, class, score) in vehicle system"""
        config = self.lead_config.policy.transfuser
        data_config = self.lead_config.expert.data_collection
        bboxes_image_system = self.pred_bounding_box_image_system
        # filter bbox based on the confidence of the prediction
        bboxes_image_system = bboxes_image_system[
            bboxes_image_system[:, :, BoundingBoxIndex.SCORE]
            > config.bb_confidence_threshold
        ]
        # convert to vehicle system
        bounding_box_vehicle_system = []
        for bis in bboxes_image_system:
            original_shape = bis.shape
            bis = bis.reshape(-1, 9)
            bounding_box_vehicle_system.append(
                labels.bb_image_to_vehicle_system(
                    bis,
                    data_config.pixels_per_meter,
                    data_config.min_x_meter,
                    data_config.min_y_meter,
                ).reshape(original_shape),
            )
        return np.array(bounding_box_vehicle_system).reshape(
            -1,
            len(bboxes_image_system),
            9,
        )


@dataclass(frozen=True)
class PredictedBoundingBox:
    """Bounding box object after post-processing and maximum-suppression."""

    x: float
    y: float
    w: float
    h: float
    yaw: float
    velocity: float
    brake: float
    clazz: int
    score: float

    @property
    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)

    def update(
        self,
        x: float,
        y: float,
        orientation: float,
        x_target: float,
        y_target: float,
        orientation_target: float,
    ) -> PredictedBoundingBox:
        pos_diff = np.array([x_target, y_target]) - np.array([x, y])
        rot_diff = geometry.normalize_angle(orientation_target - orientation)

        # Rotate difference vector from global to local coordinate system.
        rotation_matrix = np.array(
            [
                [np.cos(orientation_target), -np.sin(orientation_target)],
                [np.sin(orientation_target), np.cos(orientation_target)],
            ],
        )
        pos_diff = rotation_matrix.T @ pos_diff

        # Rotation matrix in local coordinate system
        local_rot_matrix = np.array(
            [
                [np.cos(rot_diff), -np.sin(rot_diff)],
                [np.sin(rot_diff), np.cos(rot_diff)],
            ],
        )

        # Calculate new coordinates
        local_coords = local_rot_matrix.T @ (np.array([self.x, self.y]) - pos_diff).T
        new_x, new_y = float(local_coords[0]), float(local_coords[1])
        new_yaw = float(geometry.normalize_angle(self.yaw - rot_diff))

        # Return a new bounding box with updated values
        return PredictedBoundingBox(
            x=new_x,
            y=new_y,
            w=self.w,
            h=self.h,
            yaw=new_yaw,
            velocity=self.velocity,
            brake=self.brake,
            clazz=self.clazz,
            score=self.score,
        )

    def scale(self, factor: float) -> PredictedBoundingBox:
        factor = float(factor)
        return PredictedBoundingBox(
            x=self.x * factor,
            y=self.y * factor,
            w=self.w * factor,
            h=self.h * factor,
            yaw=self.yaw,
            velocity=self.velocity,
            brake=self.brake,
            clazz=self.clazz,
            score=self.score,
        )

    def __getitem__(self, index):
        return [
            self.x,
            self.y,
            self.w,
            self.h,
            self.yaw,
            self.velocity,
            self.brake,
            self.clazz,
            self.score,
        ][index]


@precision.force_fp32(apply_to=("pred", "gaussian_target"))
def gaussian_focal_loss(
    pred: torch.Tensor,
    gaussian_target: torch.Tensor,
    alpha: float = 2.0,
    gamma: float = 4.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Adapted from mmdetection
    Args:
        pred: The prediction.
        gaussian_target: The learning target of the prediction in gaussian distribution.
        alpha: A balanced form for Focal Loss. Defaults to 2.0.
        gamma: The gamma for calculating the modulating factor. Defaults to 4.0.
        reduction: The reduction method to apply to the output: 'none' | 'mean' | 'sum'.
    Returns:
        The computed loss.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    loss = pos_loss + neg_loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    # All other reductions will be no reduction.
    return loss


def gaussian2d(
    radius: int,
    sigma: float = 1,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    """Generate 2D gaussian kernel.

    Args:
        radius: Radius of gaussian kernel.
        sigma: Sigma of gaussian function. Default: 1.
        dtype: Dtype of gaussian array. Default: np.float32.

    Returns:
        h: Gaussian kernel with a ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = np.arange(-radius, radius + 1, dtype=dtype).reshape(1, -1)
    y = np.arange(-radius, radius + 1, dtype=dtype).reshape(-1, 1)

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def gen_gaussian_target(
    heatmap: np.ndarray,
    center: list[int],
    radius: int,
    k: int = 1,
) -> np.ndarray:
    """Generate 2D gaussian heatmap.

    Args:
        heatmap: Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center: Coord of gaussian kernel's center.
        radius: Radius of gaussian kernel.
        k: Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap: Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2d(radius, sigma=diameter / 6, dtype=heatmap.dtype)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian_kernel[
        radius - top : radius + bottom,
        radius - left : radius + right,
    ]
    out_heatmap = heatmap
    np.maximum(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top : y + bottom, x - left : x + right],
    )

    return out_heatmap


def gaussian_radius(det_size: list[float], min_overlap: float) -> int:
    r"""Gaussian kernel radius for a box of shape ``det_size`` keeping IoU ``min_overlap``.

    The radius is the smallest root over three corner-placement cases; each case's
    quadratic coefficients (a, b, c) are Vieta's-formula terms for its IoU bound.
    From CornerNet-Lite: https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/utils.py#L65
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return int(min(r1, r2, r3))


def get_local_maximum(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Extract local maximum pixel with given kernel.

    Args:
        heat: Target heatmap.
        kernel: Kernel size of max pooling. Default: 3.

    Returns:
        heat: A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(
    scores: torch.Tensor,
    k: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get top k positions from heatmap.

    Args:
        scores: Target heatmap with shape [batch, num_classes, height, width].
        k: Target number. Default: 20.

    Returns:
        Scores, indexes, categories and coords of topk keypoint. Containing following Tensors:
        - topk_scores: Max scores of each topk keypoint.
        - topk_inds: Indexes of each topk keypoint.
        - topk_clses: Categories of each topk keypoint.
        - topk_ys: Y-coord of each topk keypoint.
        - topk_xs: X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, -1), k)
    topk_clses = torch.div(topk_inds, (height * width), rounding_mode="trunc")
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="trunc")
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(
    feat: torch.Tensor,
    ind: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather feature according to index.

    Args:
        feat: Target feature map.
        ind: Target coord index.
        mask: Mask of feature map. Default: None.

    Returns:
        Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Transpose and gather feature according to index.

    Args:
        feat: Target feature map.
        ind: Target coord index.

    Returns:
        Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

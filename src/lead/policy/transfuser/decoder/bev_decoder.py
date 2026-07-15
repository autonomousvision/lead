import torch
import torch.nn as nn
import torchmetrics
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F

from lead.config import LeadConfig


class BEVDecoder(nn.Module):
    def __init__(
        self,
        lead_config: LeadConfig,
        num_classes: int,
        device: torch.device,
    ) -> None:
        """Dense BEV decoder for BEV semantic segmentation.

        Args:
            lead_config: Root config tree
            num_classes: Number of semantic classes to predict
            device: Device to run the model on
        """
        super().__init__()
        self.lead_config = lead_config
        self.config = lead_config.policy.transfuser
        self.data_config = lead_config.expert.data_collection
        self.num_classes = num_classes
        self.device = device

        self.net = nn.Sequential(
            nn.Conv2d(
                self.config.bev_features_chanels,
                self.config.bev_features_chanels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.bev_features_chanels,
                num_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Upsample(
                size=(
                    self.data_config.lidar_height_pixel,
                    self.data_config.lidar_width_pixel,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

    def compute_loss(self, pred: torch.Tensor, data: dict, loss: dict, log: dict):
        """
        Compute BEV semantic segmentation loss.

        Args:
            pred: (B, C, H, W) BEV semantic prediction tensor
            data: dict containing the ground truth labels and masks
            loss: dict to store the computed loss
            log: dict to store computed metrics and logs
        Returns:
            None
        """
        if not self.config.use_bev_semantic:
            return

        label = data["bev_semantic"].to(
            pred.device,
            dtype=torch.long,
            non_blocking=True,
        )
        with autocast(device_type="cuda", enabled=False):
            loss_bev = F.cross_entropy(pred.float(), label)

        loss["loss_bev_semantic"] = loss_bev

        if (
            "iteration" in data
            and (
                (data["iteration"] + 1)
                % self.lead_config.training.experiment.log_scalars_frequency
            )
            == 0
        ):
            log["bev_semantic/output_min"] = pred.min().item()
            log["bev_semantic/output_max"] = pred.max().item()
            pred_classes = pred.argmax(dim=1)
            miou = torchmetrics.functional.jaccard_index(
                pred_classes,
                label,
                task="multiclass",
                num_classes=self.num_classes,
            )
            f1 = torchmetrics.functional.f1_score(
                pred_classes,
                label,
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
            )
            log["metric/bev_semantic_miou"] = miou.item()
            log["metric/bev_semantic_f1"] = f1.item()

    def forward(self, bev_feature_grid: torch.Tensor, log: dict):
        """Forward pass for the BEV decoder.

        Args:
            bev_feature_grid: (B, D, H, W) BEV feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) BEV feature grid after passing through the decoder
        """
        return self.net(bev_feature_grid)

import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F

from lead.common.constants import SOURCE_DATASET_NAME_MAP, SourceDataset
from lead.config import LeadConfig


class BEVDecoder(nn.Module):
    def __init__(
        self,
        lead_config: LeadConfig,
        num_classes: int,
        device: torch.device,
        source_data: int,
    ) -> None:
        """Dense BEV decoder for BEV semantic segmentation.

        Args:
            lead_config: Root config tree
            num_classes: Number of semantic classes to predict
            device: Device to run the model on
            source_data: Source dataset enum value for which this decoder is responsible
        """
        super().__init__()
        self.lead_config = lead_config
        self.config = lead_config.agent.transfuser
        self.data_config = lead_config.expert.data_collection
        self.num_classes = num_classes
        self.device = device
        self.source_data = source_data

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

        # Mask for samples from the correct source dataset
        source_dataset = data["source_dataset"].to(
            pred.device,
            dtype=torch.long,
            non_blocking=True,
        )  # (B,)
        source_mask = (source_dataset == self.source_data).float()  # (B,)
        if source_mask.sum() == 0:
            return  # No samples from this source dataset in the batch
        dataset_name = SOURCE_DATASET_NAME_MAP[self.source_data]
        prefix = f"{dataset_name}_"
        if self.source_data == SourceDataset.CARLA:
            prefix = ""

        label = data[f"{prefix}bev_semantic"].to(
            pred.device,
            dtype=torch.long,
            non_blocking=True,
        )
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Compute per-sample loss over every BEV pixel
            loss_bev_per_sample = F.cross_entropy(
                pred.float(),
                label,
                reduction="none",
            )  # (B, H, W)
            loss_bev_per_sample = loss_bev_per_sample.mean(dim=(1, 2))  # (B,)

            # Mask out losses from other data sources
            loss_bev = (
                loss_bev_per_sample * source_mask
            ).sum() / source_mask.sum().clamp(min=1)

        # Add dataset name prefix
        loss[f"{prefix}loss_bev_semantic"] = loss_bev

        if (
            "iteration" in data
            and (
                (data["iteration"] + 1)
                % self.lead_config.training.experiment.log_scalars_frequency
            )
            == 0
        ):
            subset = source_mask.bool()
            log[f"{prefix}bev_semantic/output_min"] = pred.min().item()
            log[f"{prefix}bev_semantic/output_max"] = pred.max().item()
            pred_classes = pred[subset].argmax(dim=1)
            subset_label = label[subset]
            miou = torchmetrics.functional.jaccard_index(
                pred_classes,
                subset_label,
                task="multiclass",
                num_classes=self.num_classes,
            )
            f1 = torchmetrics.functional.f1_score(
                pred_classes,
                subset_label,
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
            )
            log[f"{prefix}metric/bev_semantic_miou"] = miou.item()
            log[f"{prefix}metric/bev_semantic_f1"] = f1.item()

    def forward(self, bev_feature_grid: torch.Tensor, log: dict):
        """Forward pass for the BEV decoder.

        Args:
            bev_feature_grid: (B, D, H, W) BEV feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) BEV feature grid after passing through the decoder
        """
        return self.net(bev_feature_grid)

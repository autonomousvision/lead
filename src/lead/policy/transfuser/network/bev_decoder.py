import torch
import torchmetrics
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F

from lead.api.abstract_policy import AuxiliaryLog, TaskLosses
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch


class BEVDecoder(nn.Module):
    def __init__(
        self,
        lead_config: LeadConfig,
        num_classes: int,
    ) -> None:
        """Dense BEV decoder for BEV semantic segmentation.

        Args:
            lead_config: Root config tree
            num_classes: Number of semantic classes to predict
        """
        super().__init__()
        self.lead_config = lead_config
        self.config = lead_config.policy.transfuser
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Conv2d(
                self.config.bev_feature_channels,
                self.config.bev_feature_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.bev_feature_channels,
                num_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Upsample(
                size=(
                    self.config.lidar_height_pixel,
                    self.config.lidar_width_pixel,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

    def compute_loss(
        self,
        pred: torch.Tensor,
        data: TransfuserForwardBatch,
        loss: TaskLosses,
        log: AuxiliaryLog,
    ) -> None:
        """
        Compute BEV semantic segmentation loss.

        Args:
            pred: (B, C, H, W) BEV semantic prediction tensor
            data: TransfuserForwardBatch containing the ground truth labels and masks
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

        gradient_step = data.get("current_gradient_step")
        log_every = self.lead_config.training.experiment.log_scalars_every_n_steps
        if gradient_step is not None and ((gradient_step + 1) % log_every) == 0:
            log["outputs/bev_semantic_min"] = pred.min().item()
            log["outputs/bev_semantic_max"] = pred.max().item()
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

    def forward(self, bev_feature_grid: torch.Tensor, log: AuxiliaryLog):
        """Forward pass for the BEV decoder.

        Args:
            bev_feature_grid: (B, D, H, W) BEV feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) BEV feature grid after passing through the decoder
        """
        return self.net(bev_feature_grid)

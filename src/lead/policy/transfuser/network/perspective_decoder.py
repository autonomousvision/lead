import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.amp.autocast_mode import autocast

from lead.api.abstract_policy import AuxiliaryLog, TaskLosses
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.utils.ops import dequantize_depth


class PerspectiveDecoder(nn.Module):
    def __init__(
        self,
        lead_config: LeadConfig,
        in_channels: int,
        out_channels: int,
        perspective_upsample_factor: int,
        modality: str,
    ) -> None:
        """Decodes a low resolution perspective grid to a full resolution output. E.g. semantic segmentation, depth

        Args:
            lead_config: Root config tree
            in_channels: Feature channels of the input feature grid
            out_channels: Feature channels of the output feature grid
            perspective_upsample_factor: Upsampling factor from input feature grid to output
            modality: "semantic" or "depth"
        """
        super().__init__()
        self.modality = modality
        self.lead_config = lead_config
        self.config = lead_config.policy.transfuser
        self.scale_factor_0 = (
            perspective_upsample_factor // self.config.deconv_scale_factor_0
        )
        self.scale_factor_1 = (
            perspective_upsample_factor // self.config.deconv_scale_factor_1
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.config.deconv_channel_num_0, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_0,
                self.config.deconv_channel_num_1,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(
                self.config.deconv_channel_num_1,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                out_channels,
                3,
                1,
                1,
                bias=True,
            ),
        )

    def compute_loss(
        self,
        prediction: torch.Tensor,
        data: TransfuserForwardBatch,
        loss: TaskLosses,
        log: AuxiliaryLog,
    ) -> None:
        """Compute loss and metrics for the given modality.

        Args:
            prediction: (B, C, H, W) Prediction tensor
            data: TransfuserForwardBatch containing the ground truth labels and masks
            loss: dict to store the computed loss
            log: dict to store computed metrics and logs
        """
        if self.modality == "semantic":
            label = data["semantic"].to(
                prediction.device,
                dtype=torch.long,
                non_blocking=True,
            )
        else:
            label = dequantize_depth(
                data["depth"].to(prediction.device, non_blocking=True),
                self.lead_config.expert.storage.save_depth_max_meters,
            )

        with autocast(device_type="cuda", enabled=False):
            if self.modality == "semantic":
                loss_value = F.cross_entropy(prediction.float(), label)
            else:
                loss_value = F.l1_loss(prediction.float(), label)

        loss.update({f"loss_{self.modality}": loss_value})

        gradient_step = data.get("current_gradient_step")
        log_every = self.lead_config.training.experiment.log_scalars_every_n_steps
        if gradient_step is not None and ((gradient_step + 1) % log_every) == 0:
            log[f"outputs/{self.modality}_min"] = prediction.min().item()
            log[f"outputs/{self.modality}_max"] = prediction.max().item()
            if self.modality == "semantic":
                miou = torchmetrics.functional.jaccard_index(
                    prediction,
                    label,
                    task="multiclass",
                    num_classes=self.config.num_semantic_classes,
                )
                f1 = torchmetrics.functional.f1_score(
                    prediction,
                    label,
                    task="multiclass",
                    num_classes=self.config.num_semantic_classes,
                    average="macro",
                )
                log["metric/semantic_miou"] = miou.item()
                log["metric/semantic_f1"] = f1.item()
            else:
                mae = torchmetrics.functional.mean_absolute_error(
                    prediction.float(),
                    label.float(),
                )
                log["metric/depth_mae"] = mae.item()

    def forward(
        self,
        data: TransfuserForwardBatch,
        image_feature_grid: torch.Tensor,
        log: AuxiliaryLog,
    ):
        """Forward pass for the decoder.

        Args:
            data: TransfuserForwardBatch containing the ground truth labels and masks
            image_feature_grid: (B, D, H, W) Image feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) Prediction tensor
        """
        x = self.deconv1(image_feature_grid)
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor_0,
            mode=self.config.upsample_mode,
        )
        x = self.deconv2(x)
        if self.config.upsample_perspective_logits:
            x = self.deconv3(x)
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor_1,
            mode=self.config.upsample_mode,
        )
        if not self.config.upsample_perspective_logits:
            x = self.deconv3(x)

        # Ensure output size matches expected size
        expected_h = self.config.final_image_height
        expected_w = self.config.final_image_width
        if x.shape[2] != expected_h or x.shape[3] != expected_w:
            height_error = abs(x.shape[2] - expected_h) / expected_h * 100
            width_error = abs(x.shape[3] - expected_w) / expected_w * 100
            if max(height_error, width_error) > 10:
                raise ValueError(
                    f"Output size mismatch too large: got ({x.shape[2]}, {x.shape[3]}), expected ({expected_h}, {expected_w})",
                )
            x = F.interpolate(
                x,
                size=(expected_h, expected_w),
                mode=self.config.upsample_mode,
            )

        if self.modality == "depth":
            x = x.squeeze(1)
        return x

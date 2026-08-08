import jaxtyping as jt
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp.autocast_mode import autocast

from lead.api.abstract_policy import AuxiliaryLog, TaskLosses
from lead.common.constants import RadarDataIndex, RadarLabels
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.utils import ops


class RadarDetector(nn.Module):
    # Declared so the type checker resolves the registered buffer here rather
    # than through nn.Module.__getattr__, which types every attribute as a union.
    feature_scale: torch.Tensor

    def __init__(
        self,
        bev_input_dim: int,
        lead_config: LeadConfig,
    ) -> None:
        super().__init__()
        self.lead_config = lead_config
        config = lead_config.policy.transfuser
        self.config = config
        self.num_radar_sensors = (
            self.lead_config.expert.sensor_rig.num_radar_sensors
        )  # 4 radar sensors

        # Encoder
        self.bev_proj = nn.Conv2d(bev_input_dim, config.radar_token_dim, kernel_size=1)
        self.ego_vel_proj = nn.Linear(1, config.radar_token_dim)
        self.radar_point_tokenizer = nn.Sequential(
            nn.Linear(
                self.config.radar_token_dim + 1 + self.num_radar_sensors,
                self.config.radar_hidden_dim_tokenizer,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.config.radar_hidden_dim_tokenizer,
                self.config.radar_token_dim,
            ),
        )

        # Positional embeddings
        self.bev_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                config.lidar_bev_grid_cols * config.lidar_bev_grid_rows,
                config.radar_token_dim,
            ),
        )
        self.ego_vel_pos_embed = nn.Parameter(torch.zeros(1, 1, config.radar_token_dim))

        # Learned queries and transformer
        self.q = nn.Parameter(
            torch.zeros(1, config.num_radar_queries, config.radar_token_dim),
        )
        self.tf = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.radar_token_dim,
                nhead=config.radar_num_heads,
                dim_feedforward=config.radar_tf_dim_ff,
                dropout=config.radar_dropout,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=config.radar_num_layers,
            norm=nn.LayerNorm(config.radar_token_dim),
        )
        torch.nn.init.xavier_uniform_(self.q)

        # Decoders
        # Calculate output dimension: 3 (x, y, v)
        state_output_dim = len(RadarLabels) - 1  # x, y, v

        self.state_decoder = nn.Sequential(
            nn.Linear(config.radar_token_dim, config.radar_hidden_dim_decoder),
            nn.ReLU(inplace=True),
            nn.Linear(config.radar_hidden_dim_decoder, state_output_dim),
        )
        self.label_decoder = nn.Linear(
            config.radar_token_dim,
            1,
        )  # Invalid prediction = a vector with negative dot product with learned weights
        self.register_buffer(
            "feature_scale",
            torch.tensor(
                [
                    self.config.bev_max_x_meter - self.config.bev_min_x_meter,
                    self.config.bev_max_y_meter - self.config.bev_min_y_meter,
                    self.config.max_speed_mps,
                ],
            ),
            persistent=False,
        )

    def forward(
        self,
        bev_tokens: jt.Float[torch.Tensor, "B D H W"],
        data: TransfuserForwardBatch,
    ) -> tuple[
        jt.Float[torch.Tensor, "B Q C"],
        jt.Float[torch.Tensor, "B Q 4"],
    ]:
        """
        Args:
            bev_tokens: BEV tokens
            data: Input data dictionary
        Returns:
            Radar features.
            Radar predictions.
        """
        # Load data
        radars = data["radar"]  # (B, 300, 5)

        # Prepare context
        bev_tokens = self.bev_proj(bev_tokens)  # (B, D, H, W)
        ego_vel_token = self.ego_vel_proj(
            data["speed"].reshape(-1, 1) / self.config.max_speed_mps,
        ).unsqueeze(1)  # (B, 1, D)
        radar_tokens = self._tokenize_radar(bev_tokens, radars)  # (B, 300, D)

        # Add positional embeddings
        bev_tokens = bev_tokens.flatten(2).permute(0, 2, 1) + self.bev_pos_embed
        ego_vel_token = ego_vel_token + self.ego_vel_pos_embed
        kv = torch.cat(
            [bev_tokens, ego_vel_token, radar_tokens],
            dim=1,
        )  # (B, H*W+1+300, D)

        # Cross-attention
        radar_Features = self.tf(self.q.repeat(kv.shape[0], 1, 1), kv)  # (B, Q, D)

        decoder_output = self.state_decoder(radar_Features)  # (B, Q, 3)
        logits = self.label_decoder(radar_Features)  # (B, Q, 1)

        # Split decoder output into state
        unscaled_state = decoder_output[..., :3]  # (B, Q, 3) - x, y, v

        radar_predictions = unscaled_state.clone()

        # X coordinate: maps to [-1, 1], then scale to [min_x, max_x]
        x_center = (self.config.bev_max_x_meter + self.config.bev_min_x_meter) / 2
        x_range = (self.config.bev_max_x_meter - self.config.bev_min_x_meter) / 2
        radar_predictions[..., RadarLabels.X] = (
            torch.tanh(unscaled_state[..., RadarLabels.X]) * x_range + x_center
        )

        # Y coordinate: maps to [-1, 1], then scale to [min_y, max_y]
        y_center = (self.config.bev_max_y_meter + self.config.bev_min_y_meter) / 2
        y_range = (self.config.bev_max_y_meter - self.config.bev_min_y_meter) / 2
        radar_predictions[..., RadarLabels.Y] = (
            torch.tanh(unscaled_state[..., RadarLabels.Y]) * y_range + y_center
        )

        # Velocity: maps to [0, 1], then scale to [0, max_speed_mps]
        radar_predictions[..., RadarLabels.V] = (
            (torch.tanh(unscaled_state[..., RadarLabels.V]) + 1)
            / 2
            * self.config.max_speed_mps
        )

        radar_predictions = torch.cat([radar_predictions, logits], dim=-1)  # (B, Q, 4)
        return radar_Features, radar_predictions

    def _tokenize_radar(
        self,
        bev_tokens: jt.Float[torch.Tensor, "B D H W"],
        radars: jt.Float[torch.Tensor, "B 300 5"],
    ) -> jt.Float[torch.Tensor, "B 300 D"]:
        """Tokenize radar points by sampling BEV features at radar locations and combining with radar features."""
        pos, rel_vel, sensor_id = (
            radars[..., : RadarDataIndex.Y + 1],
            radars[..., RadarDataIndex.V : RadarDataIndex.V + 1],
            radars[..., RadarDataIndex.SENSOR_ID : RadarDataIndex.SENSOR_ID + 1],
        )  # (B, 300, 2), (B, 300, 1), (B, 300, 1)

        # Building features for each radar point
        sensor_features = torch.nn.functional.one_hot(
            sensor_id.to(torch.int64).squeeze(-1),
            num_classes=self.num_radar_sensors,
        ).to(dtype=bev_tokens.dtype)  # (B, 300, 4)
        radar_features = ops.bev_grid_sample(
            bev_tokens,
            pos,
            self.lead_config,
        )  # (B, 300, D)
        rel_vel_features = rel_vel / self.config.max_speed_mps
        features = torch.cat(
            [
                radar_features,
                rel_vel_features,
                sensor_features,
            ],
            dim=-1,
        )

        tokens = self.radar_point_tokenizer(features)
        pos = pos.reshape(-1, 2)
        return tokens + ops.gen_sineembed_for_position(
            ops.unit_normalize_bev_points(pos.reshape(-1, 2), self.lead_config),
            hidden_dim=self.config.radar_token_dim,
        ).reshape(tokens.shape)  # Positional embedding

    def compute_loss(
        self,
        pred: jt.Float[torch.Tensor, "B Q 4"],  # [x, y, v, valid_logit]
        data: TransfuserForwardBatch,
        loss: TaskLosses,
        log: AuxiliaryLog,
    ) -> None:
        gt_state = data["radar_detections"][
            ...,
            [RadarLabels.X, RadarLabels.Y, RadarLabels.V],
        ]  # (B, Q, 3)
        gt_label = data["radar_detections"][..., [RadarLabels.VALID]]  # (B, Q, 1)

        pred_state = pred[
            :,
            :,
            [RadarLabels.X, RadarLabels.Y, RadarLabels.V],
        ]  # (B, Q, 3)
        pred_label = pred[:, :, [RadarLabels.VALID]]  # (B, Q, 1)

        # Compute cost matrices for all batches at once
        state_cost = self._l1_cost_batch(
            pred_state,
            gt_state,
            gt_label.squeeze(-1),
        )  # (B, Q, Q)
        classification_cost = self._ce_cost_batch(pred_label, gt_label)  # (B, Q, Q)
        cost = (
            self.config.radar_regression_loss_weight * state_cost
            + self.config.radar_classification_loss_weight * classification_cost
        )  # (B, Q, Q)

        # Batch Hungarian matching
        pred_indices, gt_indices = self._batch_hungarian_matching(
            cost,
        )  # (B, Q), (B, Q)

        # Gather matched predictions and ground truth using advanced indexing
        batch_indices = torch.arange(pred_state.shape[0], device=pred_state.device)[
            :,
            None,
        ]  # (B, 1)

        matched_state_pred = pred_state[batch_indices, pred_indices]  # (B, Q, 3)
        matched_label_pred = pred_label[batch_indices, pred_indices]  # (B, Q, 1)
        matched_state_gt = gt_state[batch_indices, gt_indices]  # (B, Q, 3)
        matched_label_gt = gt_label[batch_indices, gt_indices]  # (B, Q, 1)

        # Compute losses in batch
        state_losses = self._l1_loss_batch(
            matched_state_pred,
            matched_state_gt,
            matched_label_gt.squeeze(-1),
        )  # (B,)
        classification_losses = self._ce_loss_batch(
            matched_label_pred,
            matched_label_gt,
        )  # (B,)

        # Final loss is batch mean
        loss["radar_loss"] = (
            self.config.radar_regression_loss_weight * state_losses
            + self.config.radar_classification_loss_weight * classification_losses
        ).mean()

        gradient_step = data.get("current_gradient_step")
        log_every = self.lead_config.training.experiment.log_scalars_every_n_steps
        if gradient_step is not None and ((gradient_step + 1) % log_every) == 0:
            gt_valid_mask = matched_label_gt.squeeze(-1).bool()  # (B, Q)

            # Distance error (L2 distance for x, y coordinates)
            xy_pred = matched_state_pred[
                ...,
                [RadarLabels.X, RadarLabels.Y],
            ]  # (B, Q, 2)
            xy_gt = matched_state_gt[..., [RadarLabels.X, RadarLabels.Y]]  # (B, Q, 2)
            distance_errors = torch.norm(xy_pred - xy_gt, dim=-1)  # (B, Q)
            valid_distance_errors = distance_errors[gt_valid_mask]
            log["metric/radar_distance_error"] = (
                valid_distance_errors.mean()
                if len(valid_distance_errors) > 0
                else torch.tensor(0.0)
            )

            # Velocity error (absolute difference)
            vel_pred = matched_state_pred[..., [RadarLabels.V]]  # (B, Q, 1)
            vel_gt = matched_state_gt[..., [RadarLabels.V]]  # (B, Q, 1)
            vel_errors = torch.abs(vel_pred - vel_gt)  # (B, Q, 1)
            valid_vel_errors = vel_errors[gt_valid_mask]
            log["metric/radar_vel_error"] = (
                valid_vel_errors.mean()
                if len(valid_vel_errors) > 0
                else torch.tensor(0.0)
            )

            # Valid Classification F1 score
            log["metric/radar_classification_f1"] = torchmetrics.functional.f1_score(
                preds=matched_label_pred.float(),
                target=matched_label_gt.long(),
                task="binary",
            )

    def _batch_hungarian_matching(
        self,
        cost: jt.Float[torch.Tensor, "B N N"],
    ) -> tuple[jt.Int[torch.Tensor, "B N"], jt.Int[torch.Tensor, "B N"]]:
        """Batch Hungarian matching using linear_sum_assignment"""
        B, N, _ = cost.shape
        # One device sync for the whole batch, solve on CPU, one upload back.
        cost_cpu = cost.detach().cpu().numpy()
        indices = np.empty((2, B, N), dtype=np.int64)
        for b in range(B):
            indices[0, b], indices[1, b] = linear_sum_assignment(cost_cpu[b])
        stacked = torch.from_numpy(indices).to(cost.device)
        return stacked[0], stacked[1]

    def _l1_loss_batch(
        self,
        pred: jt.Float[torch.Tensor, "B Q 3"],
        gt: jt.Float[torch.Tensor, "B Q 3"],
        mask: jt.Float[torch.Tensor, "B Q"],
        scale: bool = True,
    ) -> jt.Float[torch.Tensor, " B"]:
        diff = pred - gt  # (B, Q, 3)
        if scale:
            diff = diff / self.feature_scale
        losses = diff.abs().sum(dim=-1)  # (B, Q)
        losses = losses * mask  # (B, Q)
        # Normalize by number of valid matches per batch
        valid_counts = mask.sum(dim=-1).clamp(min=1)  # (B,)
        return losses.sum(dim=-1) / valid_counts

    def _ce_loss_batch(
        self,
        pred: jt.Float[torch.Tensor, "B Q 1"],
        gt: jt.Float[torch.Tensor, "B Q 1"],
    ) -> jt.Float[torch.Tensor, " B"]:
        with autocast(device_type="cuda", enabled=False):
            losses = F.binary_cross_entropy_with_logits(
                pred.float(),
                gt.float(),
                reduction="none",
            )  # (B, Q, 1)
        return losses.squeeze(-1).mean(dim=-1).to(pred.dtype)  # (B,)

    def _l1_cost_batch(
        self,
        pred: jt.Float[torch.Tensor, "B N 3"],
        gt: jt.Float[torch.Tensor, "B N 3"],
        mask: jt.Float[torch.Tensor, "B N"],
        scale: bool = True,
    ) -> jt.Float[torch.Tensor, "B N N"]:
        diff = pred[:, :, None] - gt[:, None]  # (B, N, N, 3)
        if scale:
            diff = diff / self.feature_scale
        diff = diff.abs().sum(dim=-1)  # (B, N, N)
        return mask[:, None] * diff  # Broadcast mask to (B, 1, N) then multiply

    def _ce_cost_batch(
        self,
        pred: jt.Float[torch.Tensor, "B N 1"],
        gt: jt.Float[torch.Tensor, "B N 1"],
    ) -> jt.Float[torch.Tensor, "B N N"]:
        B, N, _ = pred.shape
        pred_expanded = pred[:, :, None].expand(B, N, N, 1)  # (B, N, N, 1)
        gt_expanded = gt[:, None].expand(B, N, N, 1)  # (B, N, N, 1)
        with autocast(device_type="cuda", enabled=False):
            return (
                F.binary_cross_entropy_with_logits(
                    pred_expanded.float(),
                    gt_expanded.float(),
                    reduction="none",
                )
                .squeeze(-1)
                .to(pred.dtype)
            )  # (B, N, N)

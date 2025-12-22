import jaxtyping as jt
import timm
import torch
import torch.nn.functional as F
from beartype import beartype
from torch import nn

from lead.training.config_training import TrainingConfig
from lead.training.tfv6 import fn


class TransfuserBackbone(nn.Module):
    @beartype
    def __init__(self, device: torch.device, config: TrainingConfig):
        super().__init__()
        self.device = device
        self.config = config

        # Image branch
        self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        image_start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            image_start_index += 1
        self.num_image_features = self.image_encoder.feature_info.info[image_start_index + 3]["num_chs"]

        # LiDAR branch
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture, pretrained=False, in_chans=2 if config.LTF else 1, features_only=True
        )
        lidar_start_index = 0
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_start_index += 1
        self.num_lidar_features = self.lidar_encoder.feature_info.info[lidar_start_index + 3]["num_chs"]
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[lidar_start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[image_start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(0, 4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[image_start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[lidar_start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(0, 4)
            ]
        )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

        # Fusion transformers
        self.transformers = nn.ModuleList(
            [
                GPT(n_embd=self.image_encoder.feature_info.info[image_start_index + i]["num_chs"], config=config)
                for i in range(0, 4)
            ]
        )

        # Post-fusion convs
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[image_start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode="bilinear", align_corners=False)
        self.upsample2 = nn.Upsample(
            size=(
                self.config.lidar_height_pixel // self.config.bev_down_sample_factor,
                self.config.lidar_width_pixel // self.config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )
        self.up_conv5 = nn.Conv2d(self.config.bev_features_chanels, self.config.bev_features_chanels, (3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(self.config.bev_features_chanels, self.config.bev_features_chanels, (3, 3), padding=1)
        self.c5_conv = nn.Conv2d(self.num_lidar_features, self.config.bev_features_chanels, (1, 1))

    def top_down(self, x):
        p5 = F.relu(self.c5_conv(x), inplace=True)
        p4 = F.relu(self.up_conv5(self.upsample(p5)), inplace=True)
        p3 = F.relu(self.up_conv4(self.upsample2(p4)), inplace=True)
        return p3

    def forward(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = data["rgb"].to(self.device, dtype=self.config.torch_float_type, non_blocking=True)
        if self.config.LTF:
            x = torch.linspace(0, 1, self.config.lidar_width_pixel)
            y = torch.linspace(0, 1, self.config.lidar_height_pixel)
            y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

            lidar = torch.zeros(
                (rgb.shape[0], 2, self.config.lidar_height_pixel, self.config.lidar_width_pixel),
                device=rgb.device,
            )
            lidar[:, 0] = y_grid.unsqueeze(0)  # Top down positional encoding
            lidar[:, 1] = x_grid.unsqueeze(0)  # Left right positional encoding
        else:
            lidar = data["rasterized_lidar"].to(self.device, dtype=self.config.torch_float_type, non_blocking=True)
        return self._forward(rgb, lidar)

    @jt.jaxtyped(typechecker=beartype)
    def _forward(
        self,
        image: jt.Float[torch.Tensor, "B 3 img_h img_w"],
        lidar: jt.Float[torch.Tensor, "B 1 bev_h bev_w"] | jt.Float[torch.Tensor, "B 2 bev_h bev_w"] | None,
    ) -> tuple[jt.Float[torch.Tensor, "B D1 H1 W1"], jt.Float[torch.Tensor, "B D2 H2 W2"]]:
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image: RGB image.
            lidar: Pseudo-image LiDAR.
        Returns:
            lidar_features: BEV feature map for planning.
            image_features: Image feature map for perception.
        """
        image_features = fn.normalize_imagenet(image)
        lidar_features = lidar

        if self.config.channel_last:
            image = image.to(memory_format=torch.channels_last)
            if lidar is not None:
                lidar = lidar.to(memory_format=torch.channels_last)

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        # Loop through the 4 blocks of the network.
        for i in range(4):
            # Branch-specific forward pass
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
            image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)

        return lidar_features, image_features

    @beartype
    def forward_layer_block(self, layers, return_layers: dict[str, str], features: torch.Tensor) -> torch.Tensor:
        """Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block.

        Args:
            layers: Iterator starting at the current layer block of the target network.
            return_layers: TIMM dictionary describing at which intermediate layers features are returned.
            features: Input features.

        Return:
            torch.Tensor: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(
        self, image_features: torch.Tensor, lidar_features: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a TransFuser feature fusion block using a Transformer module.
        Args:
            image_features: Features from the image branch
            lidar_features: Features from the LiDAR branch
            layer_idx: Transformer layer index.
        Returns:
            image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](image_embd_layer, lidar_embd_layer)

        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)
        image_features_layer = F.interpolate(
            image_features_layer, size=(image_features.shape[2], image_features.shape[3]), mode="bilinear", align_corners=False
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer, size=(lidar_features.shape[2], lidar_features.shape[3]), mode="bilinear", align_corners=False
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    def __init__(self, n_embd, config):
        super().__init__()
        self.n_embd = n_embd
        self.config = config
        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.config.img_vert_anchors * self.config.img_horz_anchors
                + self.config.lidar_vert_anchors * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
                for layer in range(config.n_layer)
            ]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor (tensor): B, C, H, W
            lidar_tensor (tensor): B, C, H, W
        """
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[:, self.config.img_vert_anchors * self.config.img_horz_anchors :, :]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()
        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

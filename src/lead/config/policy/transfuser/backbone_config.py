"""TransFuser backbone and fusion-transformer architecture."""

from lead.config.node import ConfigNode


class TransfuserBackboneConfig(ConfigNode):
    """Image/LiDAR encoders and the GPT fusion layers."""

    # Backbone class as "module:ClassName"; the throughput variants under
    # lead.policy.transfuser.encoder swap in here.
    backbone_target: str = (
        "lead.policy.transfuser.encoder.transfuser_backbone:TransfuserBackbone"
    )
    # If true freeze the backbone weights during training.
    freeze_backbone: bool = False
    # If true run all normalization layers in fp32 under autocast; if false they
    # follow the autocast dtype.
    norm_layers_in_fp32: bool = True
    # Architecture name for image encoder backbone.
    image_architecture: str = "resnet34"
    # Architecture name for LiDAR encoder backbone.
    lidar_architecture: str = "resnet34"
    # Latent TF
    LTF: bool = False

    # GPT Encoder
    # Block expansion factor for GPT layers.
    block_exp: int = 4
    # Number of transformer layers used in the vision backbone.
    n_layer: int = 2
    # Number of attention heads in transformer.
    n_head: int = 4
    # Embedding dropout probability.
    embd_pdrop: float = 0.1
    # Residual connection dropout probability.
    resid_pdrop: float = 0.1
    # Attention dropout probability.
    attn_pdrop: float = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_mean: float = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_std: float = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight: float = 1.0

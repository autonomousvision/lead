"""Perspective auxiliary heads of the TransFuser model: semantics and depth."""

from lead.config.node import ConfigNode
from lead.config.policy.transfuser.label_classes import PerspectiveSemanticClass


class TransfuserSemanticConfig(ConfigNode):
    """Semantic-segmentation and depth auxiliary tasks."""

    # If true use semantic segmentation as auxiliary loss.
    use_semantic: bool = True
    # Total number of semantic segmentation classes.
    num_semantic_classes: int = len(PerspectiveSemanticClass)
    # Number of channels at the first deconvolution layer
    deconv_channel_num_0: int = 128
    # Number of channels at the second deconvolution layer
    deconv_channel_num_1: int = 64
    # Number of channels at the third deconvolution layer
    deconv_channel_num_2: int = 32
    # Fraction of the down-sampling factor that will be up-sampled in the first Up-sample
    deconv_scale_factor_0: int = 4
    # Fraction of the down-sampling factor that will be up-sampled in the second Up-sample.
    deconv_scale_factor_1: int = 8

    # --- Depth ---
    # If true use depth prediction as auxiliary task.
    use_depth: bool = True

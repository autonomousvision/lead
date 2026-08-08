"""Radar branch of the TransFuser model."""

from lead.config.node import ConfigNode


class TransfuserRadarConfig(ConfigNode):
    """Radar tokens, transformer, and detection losses."""

    # If true use radar points as additional input to the model.
    use_radar_detection: bool = True
    # Fixed number of radar points per sensor.
    num_radar_points_per_sensor: int = 75
    # Number of radar queries in the transformer.
    num_radar_queries: int = 20
    # Hidden dimension for radar tokenizer.
    radar_hidden_dim_tokenizer: int = 1024
    # Dimension of radar tokens.
    radar_token_dim: int = 256
    # Feed-forward dimension in radar transformer.
    radar_tf_dim_ff: int = 1024
    # Dropout rate for radar components.
    radar_dropout: float = 0.1
    # Number of attention heads in radar transformer.
    radar_num_heads: int = 8
    # Number of transformer layers for radar processing.
    radar_num_layers: int = 4
    # Hidden dimension for radar decoder.
    radar_hidden_dim_decoder: int = 1024
    # Loss weight for radar classification.
    radar_classification_loss_weight: float = 1.0
    # Loss weight for radar regression.
    radar_regression_loss_weight: float = 5.0

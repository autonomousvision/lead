"""Planning decoder of the TransFuser model."""

from lead.config.node import ConfigNode


class TransfuserPlanningConfig(ConfigNode):
    """Planning decoder, route, and waypoint prediction."""

    # If true model will use the planning decoder.
    use_planning_decoder: bool = False

    @property
    def is_pretraining(self) -> bool:
        """Perception-only phase: the planning head is not trained yet."""
        return not self.use_planning_decoder

    @property
    def needs_planning_targets(self) -> bool:
        """Whether the pipeline builds the planning labels: decoder training or dataset visualization."""
        return (
            self.use_planning_decoder
            or self._root.training.experiment.visualize_dataset
        )

    # Number of BEV cross-attention layers in TransFuser.
    transfuser_num_bev_cross_attention_layers: int = 6
    # Number of attention heads in BEV cross-attention.
    transfuser_num_bev_cross_attention_heads: int = 8
    # Dimension of tokens in the transformer.
    transfuser_token_dim: int = 256
    # If true predict target speed.
    predict_target_speed: bool = True
    # If true predict spatial path.
    predict_spatial_path: bool = True
    # If true predict temporal spatial waypoints.
    predict_temporal_spatial_waypoints: bool = True
    # Carla target speed prediction classes in m/s.
    target_speed_classes: list[float] = [
        0.0,
        4.0,
        8.0,
        10.0,
        13.88888888,
        16.0,
        17.77777777,
        20.0,
    ]
    # If true smooth the route points with a spline.
    smooth_route: bool = True
    # Number of route points we use for smoothing.
    num_route_points_smoothing: int = 20
    # Number of route checkpoints to predict. Needs to be smaller than num_route_points_smoothing!
    num_route_points_prediction: int = 10

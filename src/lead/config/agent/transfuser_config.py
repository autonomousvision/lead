"""TransFuser architecture configuration: backbone, decoders, and network inputs."""

from lead.common.constants import (
    TransfuserBEVSemanticClass,
    TransfuserBoundingBoxClass,
    TransfuserSemanticSegmentationClass,
)
from lead.config.node import ConfigNode


class TransfuserConfig(ConfigNode):
    """TransFuser architecture, auxiliary heads, and network-input knobs."""

    # --- Image config ---
    @property
    def final_image_width(self) -> int:
        """Final width of images across all cameras."""
        return self._root.expert.sensor_rig.image_width

    @property
    def final_image_height(self) -> int:
        """Final height of images."""
        return self._root.expert.sensor_rig.image_height

    @property
    def img_vert_anchors(self) -> int:
        """Number of vertical anchors for image feature maps."""
        return self.final_image_height // 32

    @property
    def img_horz_anchors(self) -> int:
        """Number of horizontal anchors for image feature maps."""
        return self.final_image_width // 32

    @property
    def lidar_vert_anchors(self) -> int:
        """Number of vertical anchors for LiDAR feature maps."""
        return self._root.expert.data_collection.lidar_height_pixel // 32

    @property
    def lidar_horz_anchors(self) -> int:
        """Number of horizontal anchors for LiDAR feature maps."""
        return self._root.expert.data_collection.lidar_width_pixel // 32

    # --- TransFuser backbone ---
    # If true freeze the backbone weights during training.
    freeze_backbone: bool = False
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

    # --- RaDAR ---
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

    # --- Semantic segmentation ---
    # If true use semantic segmentation as auxiliary loss.
    use_semantic: bool = True
    # Total number of semantic segmentation classes.
    num_semantic_classes: int = len(TransfuserSemanticSegmentationClass)
    # Resolution at which the perspective auxiliary tasks are predicted
    perspective_downsample_factor: int = 1
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

    # --- BEV Semantic ---
    # If true use bev semantic segmentation as auxiliary loss for training.
    use_bev_semantic: bool = True
    # Total number of BEV semantic segmentation classes.
    num_bev_semantic_classes: int = len(TransfuserBEVSemanticClass)
    # Scale factor for pedestrian BEV semantic size.
    scale_pedestrian_bev_semantic_size: float = 2.5
    # Minimum extent for pedestrian BEV representation.
    pedestrian_bev_min_extent: float = 0.4
    # Number of channels for the BEV feature pyramid.
    bev_features_chanels: int = 64
    # Resolution at which the BEV auxiliary tasks are predicted.
    bev_down_sample_factor: int = 4
    # Upsampling factor for BEV features.
    bev_upsample_factor: int = 2

    # --- Bounding boxes detection ---
    # If true use the bounding box auxiliary task.
    detect_boxes: bool = True
    # List of static object types to include in bounding box detection.
    data_bb_static_types_white_list: list[str] = [
        "static.prop.constructioncone",
        "static.prop.trafficwarning",
    ]
    # Confidence of a bounding box that is needed for the detection to be accepted.
    bb_confidence_threshold: float = 0.3
    # Maximum number of bounding boxes our system can detect.
    max_num_bbs: int = 90
    # Number of direction bins for object orientation.
    num_dir_bins: int = 12
    # Top K center keypoints to consider during detection.
    top_k_center_keypoints: int = 100
    # Kernel size for CenterNet max pooling operation.
    center_net_max_pooling_kernel: int = 3
    # Number of input channels for bounding box detection head.
    bb_input_channel: int = 64
    # Extra width to add when car doors are open for safety.
    car_open_door_extra_width: float = 1.2
    # Total number of bounding box classes to detect.
    num_bb_classes: int = len(TransfuserBoundingBoxClass)

    # --- Context and statuses (network inputs) ---
    # If true train model with noisy target points. The level of noise depends on use_kalman_filter_for_gps.
    use_noisy_tp: bool = False
    # If true, use the Kalman filter for less noisy ego state estimation.
    use_kalman_filter_for_gps: bool = True
    # If true use the velocity as input to the network.
    use_velocity: bool = True
    # Maximum speed limit for the vehicle in m/s.
    max_speed: float = 25.0
    # If true use the previous/visited target point as input to the network.
    use_previous_tp: bool = True
    # If true use the next/subsequent target point as input to the network.
    use_next_tp: bool = True
    # If true use the current target point as input to the network.
    use_tp: bool = True
    # Normalization constants for target points [x_norm, y_norm].
    target_points_normalization_constants: list[list[float]] = [[200.0, 50.0]]
    # Distance threshold for popping target points from route.
    tp_pop_distance: float = 3.25

    # --- Planning decoder configuration ---
    # If true model will use the planning decoder.
    use_planning_decoder: bool = False
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
    # Assume maximum distance between two future waypoints in meters.
    max_distance_future_waypoint: float = 10.0
    # Number of waypoints to predict (4Hz and 2 seconds).
    num_way_points_prediction: int = 8
    # Spacing between predicted waypoints. For example: spacing 5 = 4Hz prediction.
    waypoints_spacing: int = 5

    def detailed_loss_weights(self, _: int) -> dict[str, float]:
        """Computed loss weights for all auxiliary tasks with normalization."""

        weights = {
            "loss_semantic": 1.0,
            "loss_depth": 0.00001,
            "loss_bev_semantic": 1.0,
            "loss_center_net_heatmap": 1.0,
            "loss_center_net_wh": 1.0,
            "loss_center_net_offset": 1.0,
            "loss_center_net_yaw_class": 1.0,
            "loss_center_net_yaw_res": 1.0,
            "loss_center_net_velocity": 1.0,
            "radar_loss": 1.0,
        }

        if self.LTF:
            weights["loss_center_net_velocity"] = 0.0

        if not self.use_semantic:
            weights["loss_semantic"] = 0.0

        if not self.use_depth:
            weights["loss_depth"] = 0.0

        if not self.use_bev_semantic:
            weights["loss_bev_semantic"] = 0.0

        if not self.detect_boxes:
            weights["loss_center_net_heatmap"] = 0.0
            weights["loss_center_net_wh"] = 0.0
            weights["loss_center_net_offset"] = 0.0
            weights["loss_center_net_yaw_class"] = 0.0
            weights["loss_center_net_yaw_res"] = 0.0
            weights["loss_center_net_velocity"] = 0.0

        if self._root.training.data.training_used_lidar_steps <= 1:
            weights["loss_center_net_velocity"] = 0.0

        if not self.use_radar_detection:
            weights["radar_loss"] = 0.0

        # Unified planning loss
        weights.update(
            {
                "loss_spatio_temporal_waypoints": 1.0,
                "loss_target_speed": 1.0,
                "loss_spatial_route": 1.0,
            },
        )

        # Disable planning losses during pretraining
        if not self.use_planning_decoder:
            weights["loss_spatio_temporal_waypoints"] = 0.0
            weights["loss_spatial_route"] = 0.0
            weights["loss_target_speed"] = 0.0

        return weights

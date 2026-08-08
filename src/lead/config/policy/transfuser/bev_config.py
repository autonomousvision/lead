"""BEV-semantic auxiliary head of the TransFuser model."""

from lead.config.node import ConfigNode
from lead.config.policy.transfuser.label_classes import BEVSemanticClass


class TransfuserBevConfig(ConfigNode):
    """BEV semantic-segmentation auxiliary task."""

    # If true use bev semantic segmentation as auxiliary loss for training.
    use_bev_semantic: bool = True
    # Total number of BEV semantic segmentation classes.
    num_bev_semantic_classes: int = len(BEVSemanticClass)
    # Scale factor for pedestrian BEV semantic size.
    pedestrian_bev_extent_scale: float = 2.5
    # Minimum extent for pedestrian BEV representation.
    pedestrian_bev_min_half_extent_meter: float = 0.4
    # Width the map's lane markings are rasterized at, in meters. Declared in
    # meters rather than pixels so it holds its physical width at any raster
    # resolution.
    lane_marker_width_meter: float = 0.25

    @property
    def lane_marker_thickness_pixel(self) -> int:
        """Stroke width the lane markings are drawn with, at least one pixel."""
        return max(
            1,
            round(
                self.lane_marker_width_meter
                * self._root.policy.transfuser.bev_pixels_per_meter,
            ),
        )

    # Number of channels for the BEV feature pyramid.
    bev_feature_channels: int = 64
    # Resolution at which the BEV auxiliary tasks are predicted.
    bev_downsample_factor: int = 4
    # Upsampling factor for BEV features.
    bev_upsample_factor: int = 2

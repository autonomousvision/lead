"""Camera input geometry of the TransFuser model."""

from py123d.datatypes.sensors.base_camera import CameraID

from lead.config.node import ConfigNode, overridable_property


class TransfuserCameraConfig(ConfigNode):
    """Camera selection and the stitched image geometry."""

    @overridable_property
    def input_cameras(self) -> list[CameraID]:
        """Cameras the model ingests, in left-to-right stitch order."""
        return [CameraID.PCAM_L0, CameraID.PCAM_F0, CameraID.PCAM_R0]

    @property
    def final_image_width(self) -> int:
        """Final width of the stitched model input across the input cameras."""
        return len(self.input_cameras) * self._root.expert.sensor_rig.camera_width

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

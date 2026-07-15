"""Sensor rig configuration: LiDARs, cameras and radars."""

from typing import TypedDict

from lead.config.node import ConfigNode


class CameraSpec(TypedDict):
    """Calibration of one camera: mounting pose, resolution and field of view."""

    pos: list[float]
    rot: list[float]
    width: int
    height: int
    fov: int


class RadarSpec(TypedDict):
    """Calibration of one radar: mounting pose and field of view."""

    pos: list[float]
    rot: list[float]
    horz_fov: float
    vert_fov: float


class SensorRigConfig(ConfigNode):
    """Mounting positions and parameters of the LiDAR, camera and radar rig."""

    # --- LiDAR Configuration ---
    # The sensor rig always uses two LiDARs; a single-LiDAR setup is not supported.

    # x, y, z mounting position of the first LiDAR
    lidar_pos_1: list[float] = [1.0, 0.0, 2.5]
    # Roll, pitch, yaw rotation of first LiDAR (degrees)
    lidar_rot_1: list[float] = [0.0, 0.0, -90.0]
    # x, y, z mounting position of the second LiDAR
    lidar_pos_2: list[float] = [-1.0, 0.0, 2.5]
    # Roll, pitch, yaw rotation of second LiDAR (degrees)
    lidar_rot_2: list[float] = [0.0, 0.0, -270.0]

    # --- Camera Configuration ---
    # Calibration of the RGB/depth/semantic cameras; camera ``i`` (1-based) in
    # sensor specs corresponds to ``cameras[i - 1]``. Defaults to the six-camera
    # surround rig; config profiles override this list for other rigs.
    cameras: list[CameraSpec] = [
        {  # front-left
            "pos": [0.0, -0.3, 2.25],
            "rot": [0.0, 0.0, -57.5],
            "width": 384,
            "height": 384,
            "fov": 60,
        },
        {  # front
            "pos": [0.25, 0.0, 2.25],
            "rot": [0.0, 0.0, 0.0],
            "width": 384,
            "height": 384,
            "fov": 60,
        },
        {  # front-right
            "pos": [0.0, 0.3, 2.25],
            "rot": [0.0, 0.0, 57.5],
            "width": 384,
            "height": 384,
            "fov": 60,
        },
        {  # rear-right
            "pos": [-0.30, 0.3, 2.25],
            "rot": [0.0, 0.0, 122.5],
            "width": 384,
            "height": 384,
            "fov": 60,
        },
        {  # rear
            "pos": [-0.55, 0.0, 2.25],
            "rot": [0.0, 0.0, 180.0],
            "width": 384,
            "height": 384,
            "fov": 60,
        },
        {  # rear-left
            "pos": [-0.30, -0.3, 2.25],
            "rot": [0.0, 0.0, -122.5],
            "width": 384,
            "height": 384,
            "fov": 60,
        },
    ]

    @property
    def num_cameras(self) -> int:
        """Number of cameras in the rig."""
        return len(self.cameras)

    @property
    def camera_width(self) -> int:
        """Width of a single camera image; all cameras share one resolution."""
        return self.cameras[0]["width"]

    @property
    def camera_height(self) -> int:
        """Height of a single camera image; all cameras share one resolution."""
        return self.cameras[0]["height"]

    @property
    def image_width(self) -> int:
        """Width of all camera images stitched side by side."""
        return self.num_cameras * self.camera_width

    @property
    def image_height(self) -> int:
        """Height of the stitched camera image."""
        return self.camera_height

    # --- Radar Configuration ---
    # Calibration of the radar sensors. Radar ``i`` (1-based) in sensor specs
    # corresponds to ``radars[i - 1]``.
    radars: list[RadarSpec] = [
        {  # front-left
            "pos": [2.6, 0.0, 0.60],
            "rot": [0.0, 0.0, -45.0],
            "horz_fov": 90,
            "vert_fov": 0.1,
        },
        {  # front
            "pos": [2.6, 0.0, 0.60],
            "rot": [0.0, 0.0, 45.0],
            "horz_fov": 90,
            "vert_fov": 0.1,
        },
        {  # front-right
            "pos": [-2.6, 0.0, 0.60],
            "rot": [0.0, 0.0, 135.0],
            "horz_fov": 90,
            "vert_fov": 0.1,
        },
        {  # rear
            "pos": [-2.6, 0.0, 0.60],
            "rot": [0.0, 0.0, 225.0],
            "horz_fov": 90,
            "vert_fov": 0.1,
        },
    ]

    @property
    def num_radar_sensors(self) -> int:
        """Number of radar sensors in the rig."""
        return len(self.radars)

    # If true use radar sensors
    use_radars: bool = True

"""The frame: one tick of raw driving data that the generic layer hands to a
model-specific loader."""

from dataclasses import dataclass

import jaxtyping as jt
import numpy.typing as npt
from py123d.api.map.map_api import MapAPI
from py123d.datatypes import (
    BoxDetectionsSE3,
    EgoStateSE3,
    TrafficLightDetections,
)
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.base_camera import Camera
from py123d.datatypes.sensors.lidar import Lidar
from py123d.datatypes.sensors.radar import Radar


@dataclass
class RigPerturbation:
    """The rig offset the frame's sensors were captured with, in the CARLA
    conventions of collection time."""

    # Lateral translation in meters.
    translation: float
    # Yaw rotation in degrees.
    rotation: float


@dataclass
class Frame:
    """One tick of raw driving data in the ego view frame.

    The 123D modalities are in 123D's ISO 8855 conventions; everything else is
    in the chosen view's CARLA ego frame (identity for the normal view).
    """

    # --- Sensors: the log's own 123D modalities, in 123D's conventions ---
    # The anchor tick's cameras in LEAD camera order (1..num_cameras), each
    # carrying its image, its calibration and its pose (``Camera.image``,
    # ``.metadata``, ``.camera_to_global_se3``).
    cameras: list[Camera]
    # The lidar sweeps of the temporal window, keyed by frame age (0 = anchor
    # tick), each in the IMU frame of its own tick; None when not loaded.
    lidar_sweeps: dict[int, Lidar] | None = None
    # The merged radar returns of the same window, keyed by frame age. The
    # per-point features carry the sensor id and the radial velocity.
    radar_sweeps: dict[int, Radar] | None = None
    # The semantic cameras, same order as ``cameras``: their image holds the
    # raw CARLA class of every pixel, of the taxonomy their
    # ``metadata.segmentation_label_class`` names. Consumers reduce it to their
    # own label space.
    semantics: list[Camera] | None = None
    # The instance cameras, same order: their image holds the CARLA actor id of
    # every pixel, which the box detections' track tokens index into.
    instances: list[Camera] | None = None
    # The depth cameras, same order as ``cameras``. Their image is the stored
    # encoding; ``metadata.decode_depth(image)`` turns it into metric depth.
    depths: list[Camera] | None = None

    # --- Native 123D modalities: the log's own data, in 123D's conventions ---
    # The ego's SE(3) pose and dynamic state in the global frame.
    ego_state: EgoStateSE3 | None = None
    # The tick's box detections in the global frame, with 123D labels and the
    # track tokens that identify an object across frames. The per-box fields
    # the stream has no slot for are in ``meta["box_attributes"]``.
    box_detections: BoxDetectionsSE3 | None = None
    # The tick's per-lane traffic-light states.
    traffic_lights: TrafficLightDetections | None = None
    # Map of the town, for BEV rasterization.
    map_api: MapAPI | None = None

    # --- Provenance: the log's own 123D metadata; unset online ---
    log_metadata: LogMetadata | None = None
    scene_metadata: SceneMetadata | None = None

    # --- Navigation and ego history: available both offline and online ---
    # The route planner's target points in the view frame.
    target_point_previous: jt.Float[npt.NDArray, " 2"] | None = None
    target_point: jt.Float[npt.NDArray, " 2"] | None = None
    target_point_next: jt.Float[npt.NDArray, " 2"] | None = None
    # Filtered ego pose history in the anchor frame, index i = i ticks ago.
    past_positions: jt.Float[npt.NDArray, "t 2"] | None = None
    past_yaws: jt.Float[npt.NDArray, " t"] | None = None

    # --- View: the rig perturbation the sensors were captured with ---
    # None for the normal view; consumers re-express ego-frame quantities in
    # the view frame themselves (``lead.dataloader.view_geometry``).
    perturbation: RigPerturbation | None = None

    # --- Privileged: read from the log, absent at inference time ---
    # Raw driving-meta dict of the anchor tick (ego-frame contents untouched).
    meta: dict | None = None

    @property
    def is_privileged(self) -> bool:
        """Whether the frame carries the log-only fields a policy needs for labels."""
        return self.meta is not None

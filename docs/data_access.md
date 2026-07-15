# Data access

## Reading raw py123d modalities

The logs are ordinary py123d logs — point py123d directly at them to read any
modality, at any iteration (0 = anchor). No LEAD import needed:

```python
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.datatypes import LidarID

# "normal_view" is the nominal sensor rig; the perturbated rig is a separate,
# opt-in split "perturbated_view", specific for CARLA Leaderboard.
scenes = ArrowSceneBuilder(
    logs_root="/path/to/lead-data/logs",
    maps_root="/path/to/lead-data/maps",
).get_scenes(
    SceneFilter(split_names=["normal_view"], future_num_iterations=8),
    ThreadPoolExecutor(),
)

scene = scenes[0]

ego = scene.get_ego_state_se3_at_iteration(0)  # EgoStateSE3
boxes = scene.get_box_detections_se3_at_iteration(0)  # BoxDetectionsSE3
lights = scene.get_traffic_light_detections_at_iteration(0)  # TrafficLightDetections
lidar = scene.get_lidar_at_iteration(0, LidarID.LIDAR_TOP)  # Lidar
camera_ids = scene.get_camera_metadatas()  # {CameraID: metadata}
camera = scene.get_camera_at_iteration(0, next(iter(camera_ids)))  # Camera
map_api = scene.get_map_api()  # MapAPI

# LEAD's expert state is just a py123d custom modality — read the raw dict:
meta = scene.get_custom_modality_at_iteration(0, "driving_meta").data
```

## Reading a CARLA-ready `Frame` with LEAD's loader

py123d doesn't know about perturbated sensor rigs, lidar sweep-window
accumulation, or navigation target points — a CARLA policy needs all three.
`Py123DDataLoader` assembles them per tick into a `Frame`:

```python
from py123d.api.scene.scene_filter import SceneFilter

from lead.dataloader import Py123DDataLoader

loader = Py123DDataLoader(
    "/path/to/lead-data",
    SceneFilter(future_num_iterations=8),
    perturbation_prob=0.0,  # chance of the perturbated rig instead of the nominal one
)

frame = loader[0]  # len(loader) frames, indexed like a sequence

# Sensors, in LEAD order (left to right, 1..n); every field is py123d-native.
frame.cameras  # list[Camera]
frame.depths  # list[Camera]
frame.semantics  # list[Camera]
frame.instances  # list[Camera]
frame.lidar_sweeps  # dict[int, Lidar], keyed by frame age (0 = now)
frame.radar_sweeps  # dict[int, Radar], keyed by frame age (0 = now)

# Perception and the world; also py123d-native.
frame.ego_state  # EgoStateSE3
frame.box_detections  # BoxDetectionsSE3
frame.traffic_lights  # TrafficLightDetections
frame.map_api  # MapAPI
frame.log_metadata  # LogMetadata
frame.scene_metadata  # SceneMetadata

# LEAD-specific fields with no py123d slot, so they live on the frame directly.
frame.meta  # dict
frame.target_point  # NDArray (2,), next navigation goal in the ego frame
frame.perturbation  # RigPerturbation | None
```

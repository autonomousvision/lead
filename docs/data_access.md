# Data access

## How the data is organized

Each log is one expert run of one route, grouped by the
[CARLA Leaderboard 2.0 scenario type](https://leaderboard.carla.org/scenarios/)
it exercises:

```
<lead-data>/
├── logs/
│   ├── normal_view/                 # nominal sensor rig
│   │   └── <ScenarioType>/          # e.g. Accident
│   │       └── <log_name>/          # e.g. Town13_Rep-1_1073_1_route0_07_20_13_10_39
│   │           └── *.arrow          # one file per modality stream, see table below
│   └── perturbated_view/            # same tree, sensors re-rendered from a perturbated rig
├── maps/
│   └── carla/carla_<town>.arrow     # converted OpenDRIVE map, one per town
└── results/
    └── <ScenarioType>/<route>.json  # leaderboard route record of the run
```

## Storage frequencies

The simulator runs in [synchronous mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)
at 20 fps (fixed 0.05 s step). State and range sensors are stored every tick;
the render-heavy camera streams and the driving meta are stored on every fifth
tick (4 Hz), the *save ticks*. Scene iterations count save ticks, so
`SceneFilter(future_num_iterations=8)` spans 2 s of future.

| Stream                     | Content                                                                                                                                                                     | Rate  |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---- |
| `ego_state_se3`            | Ground-truth ego pose and dynamics (ISO 8855)                                                                                                                               | 20 Hz |
| `box_detections_se3`       | Ground-truth boxes of all actors                                                                                                                                            | 20 Hz |
| `lidar.lidar_top`          | Merged sweep of the two roof [lidars](https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor)                                                                     | 20 Hz |
| `radar.radar_merged`       | Merged points of the four [radars](https://carla.readthedocs.io/en/latest/ref_sensors/#radar-sensor), with radial velocity                                                  | 20 Hz |
| `camera.pcam_*`            | Six [RGB cameras](https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera), JPEG-encoded                                                                             | 4 Hz  |
| `camera_depth.pcam_*`      | [Depth cameras](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera), 8-bit linear quantization saturating at 50 m                                             | 4 Hz  |
| `camera_semantic.pcam_*`   | Semantic class channel of the [instance segmentation cameras](https://carla.readthedocs.io/en/latest/ref_sensors/#instance-segmentation-camera) (raw 29-class CARLA labels) | 4 Hz  |
| `camera_instance.pcam_*`   | Instance id channel of the same cameras                                                                                                                                     | 4 Hz  |
| `traffic_light_detections` | Traffic-light states, linked to map lanes                                                                                                                                   | 4 Hz  |
| `custom.driving_meta`      | CARLA-specific meta information, see below                                                                                                                                  | 4 Hz  |
| `sync`                     | One row per save tick; defines the scene iterations                                                                                                                         | 4 Hz  |

There are two ego poses. `ego_state_se3` is the exact pose from the simulator.
For end-to-end driving, policies only have access to the noisy
[GNSS](https://carla.readthedocs.io/en/latest/ref_sensors/#gnss-sensor) and
[IMU](https://carla.readthedocs.io/en/latest/ref_sensors/#imu-sensor) signals;
the pose estimated from them is stored in `custom.driving_meta`.

Traffic lights appear in two forms: `traffic_light_detections` carries the
light *state* per affected map lane, while `box_detections_se3` carries two box
classes per light, the stop-line trigger box (`TRAFFIC_LIGHT`, with state and
`affects_ego` in its box attributes) and the visible housing
(`TRAFFIC_LIGHT_PHYSICAL`).

## The perturbated view

Each save tick is also rendered from a perturbated rig (cameras shifted
0.1–1.0 m, rotated 5–12.5° yaw) and written to the `perturbated_view` split
under the same log name. It holds only the view-dependent streams (RGB, depth,
segmentation, radar) plus ego states; everything else lives in `normal_view`.

`normal_view` alone is a self-contained py123d dataset. The perturbated view
is read through LEAD's `Py123DDataLoader`: it pairs both views per scene,
picks the perturbated sensors with `perturbation_prob`, re-projects ego-frame
outputs such as the target points, and reports the rig offset as
`frame.perturbation`.

## The `driving_meta` stream

`custom.driving_meta` is a py123d custom modality holding the CARLA-specific
state as a plain dict per save tick. Each tick carries:

- **Localization**: `localized_pos_global` and `theta`, the GNSS+IMU pose the
  ego drives by, plus `past_positions`/`past_yaws` in the current ego frame.
- **Route**: `target_point_indices` (current target point per pop distance),
  the dense future `route` the expert follows, and route progress fields.
- **Driving decisions**: `target_speed`, `speed_limit`, the executed `steer`/
  `throttle`/`brake`, and what reduced the speed (`speed_reduced_by_obj_*`).
- **Hazards**: `vehicle_hazard`, `light_hazard`, `walker_hazard`,
  `stop_sign_hazard` with the affecting actor ids.
- **Scenario and map context**: active scenario type and its actor ids,
  distances to scenario obstacles, road/lane/junction ids, lane markings,
  weather, and traffic-light style flags (`europe_traffic_light`,
  `over_head_traffic_light`).
- **`box_attributes`**: per-box fields the box stream has no native slot for
  (occlusion pixel counts, `affects_ego`, traffic-light state), keyed by the
  box's track token.

## Reading raw py123d modalities

The logs are ordinary py123d logs: point py123d directly at them to read any
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

# LEAD's expert state is just a py123d custom modality, read as a raw dict:
meta = scene.get_custom_modality_at_iteration(0, "driving_meta").data
```

## Reading a CARLA-ready `Frame` with LEAD's loader

py123d doesn't know about perturbated sensor rigs, lidar sweep-window
accumulation, or navigation target points, but a CARLA policy needs all three.
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

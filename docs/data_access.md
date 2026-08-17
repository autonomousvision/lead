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
└── maps/
    └── carla/carla_<town>.arrow     # converted OpenDRIVE map, one per town
```

## Reading raw py123d modalities

Point py123d directly at the logs to read any
modality, at any iteration (0 = anchor), no LEAD import needed:

```python
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.datatypes import LidarID

scenes = ArrowSceneBuilder(
    logs_root="/path/to/lead-data/logs",
    maps_root="/path/to/lead-data/maps",
).get_scenes(
    SceneFilter(
        split_names=["normal_view"],
        future_num_iterations=40,
        required_scene_modalities=["camera:all@initial"],
    ),
    ThreadPoolExecutor(),
)

scene = scenes[0]

ego = scene.get_ego_state_se3_at_iteration(0)
boxes = scene.get_box_detections_se3_at_iteration(0)
lights = scene.get_traffic_light_detections_at_iteration(0)
lidar = scene.get_lidar_at_iteration(0, LidarID.LIDAR_TOP)
camera_ids = scene.get_camera_metadatas()
camera = scene.get_camera_at_iteration(0, next(iter(camera_ids)))
map_api = scene.get_map_api()

# LEAD's expert state is a py123d custom modality:
meta = scene.get_custom_modality_at_iteration(0, "driving_meta").data
```

## Data-loader for CARLA Leaderboard

For E2E driving policies, we provide `SceneLoader`, which assembles temporal and
novel view features into a `SceneData`:

```python
from py123d.api.scene.scene_filter import SceneFilter

from lead.log_reader import SceneLoader

loader = SceneLoader(
    "/path/to/lead-data",
    SceneFilter(future_num_iterations=40),
    perturbation_probability=0.0,
)

scene_data = loader[0]  # len(loader) scenes, indexed like a sequence
```

The sensor lists are in LEAD camera order (left to right, 1..n); the 123D
modalities are py123d-native and in its ISO 8855 conventions, everything else
is in the chosen view's CARLA ego frame:

| Attribute                                                    | Type                                      | Description                                                                                                                                                                                                                                      |
| :----------------------------------------------------------- | :---------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cameras`                                                    | `list[Camera]`                            | The anchor tick's RGB cameras; each carries its image, calibration and pose                                                                                                                                                                      |
| `depth_cameras`                                              | `list[Camera] \| None`                    | Depth cameras, same order; `metadata.decode_depth(image)` turns the stored encoding into metric depth                                                                                                                                            |
| `semantic_cameras`                                           | `list[Camera] \| None`                    | Semantic cameras, same order; each pixel holds the raw CARLA class                                                                                                                                                                               |
| `instance_cameras`                                           | `list[Camera] \| None`                    | Instance cameras, same order; each pixel holds the CARLA actor id the box track tokens index into                                                                                                                                                |
| `lidar_sweeps`                                               | `dict[int, Lidar] \| None`                | Lidar history; key `i` is the sweep captured `i` ticks (50 ms each) before the anchor, in the IMU frame of its own tick; ages without a stored sweep are absent                                                                                  |
| `radar_sweeps`                                               | `dict[int, Radar] \| None`                | Merged radar returns with the same age keying; per-point features carry the sensor id and the radial velocity                                                                                                                                    |
| `ego_state`                                                  | `EgoStateSE3 \| None`                     | Ground-truth ego pose and dynamic state in the global frame                                                                                                                                                                                      |
| `box_detections`                                             | `BoxDetectionsSE3 \| None`                | Boxes in the global frame with 123D labels and track tokens; the extra per-box fields live in `driving_meta["box_attributes"]`                                                                                                                   |
| `traffic_lights`                                             | `TrafficLightDetections \| None`          | The tick's per-lane traffic-light states                                                                                                                                                                                                         |
| `map_api`                                                    | `MapAPI \| None`                          | Map of the town, for BEV rasterization                                                                                                                                                                                                           |
| `log_metadata`, `scene_metadata`                             | `LogMetadata`, `SceneMetadata`            | Which log the scene comes from (dataset, split, log name, town) and where its scene sits in that log; only filled when reading from logs, not at inference                                                                                       |
| `previous_target_point`, `target_point`, `next_target_point` | `numpy.typing.NDArray (2,)`               | The route planner's target points in the view frame                                                                                                                                                                                              |
| `past_ego_positions`                                         | `numpy.typing.NDArray (t, 2)`             | Localized ego position history in the anchor frame, one entry per tick (20 Hz); index `i` is `i` ticks ago                                                                                                                                       |
| `past_ego_yaws`                                              | `numpy.typing.NDArray (t,)`               | Localized ego yaw history, one entry per tick (20 Hz), same indexing                                                                                                                                                                             |
| `rig_perturbation`                                           | `lead.log_reader.RigPerturbation \| None` | Rig offset the sensors were captured with; None for the normal view                                                                                                                                                                              |
| `driving_meta`                                               | `dict \| None`                            | The anchor tick's `driving_meta` dict, documented below. Only scenes read from logs have it; scene data built live in the simulator carries None, since the expert's state does not exist there — `scene_data.is_privileged` tells the two apart |
| `future_ego_states`                                          | `dict[int, EgoStateSE3 \| None] \| None`  | Ego states at the iterations the loading spec asked for, keyed by iteration; None at inference, where there is no future                                                                                                                         |
| `future_driving_metas`                                       | `dict[int, dict \| None] \| None`         | The `driving_meta` dicts at those same iterations, same keying                                                                                                                                                                                   |

The attributes above are the inputs, and the same ones inference builds live
from the simulator. The future exists only when reading from logs, and only
when asked for: a `SceneLoadingSpec(future_iterations=...)` passed to
`loader.read()` fills `future_ego_states` and `future_driving_metas`, keyed by
iteration past the anchor (up to the filter's `future_num_iterations`). The
same reads are also available directly on the loader, by sample index:

```python
states = loader.read_future_ego_states(0, iterations=[5, 10, 15, 20, 25, 30, 35, 40])
states[40]  # EgoStateSE3 2 s after the anchor of sample 0

metas = loader.read_future_driving_metas(0, iterations=[5, 10])
```

## Storage frequencies

The simulator runs in [synchronous mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)
at 20 fps (fixed 0.05 s step). Every stream is stored on every tick, except
the render-heavy camera streams, which are stored on every fifth tick (4 Hz),
the *save ticks*. Scene iterations count simulator ticks, so
`SceneFilter(future_num_iterations=40)` spans 2 s of future. Camera reads
return None between save ticks; requiring `camera:all@initial` anchors scenes
on the save ticks (LEAD's loader always does).

| Stream                     | Content                                                                                                                                         | Rate  |
| :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- | :---- |
| `ego_state_se3`            | Ground-truth ego pose and dynamics (ISO 8855)                                                                                                   | 20 Hz |
| `box_detections_se3`       | Ground-truth boxes of all actors                                                                                                                | 20 Hz |
| `lidar.lidar_top`          | Merged sweep of the two roof [lidars](https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor)                                         | 20 Hz |
| `radar.radar_merged`       | Merged points of the four [radars](https://carla.readthedocs.io/en/latest/ref_sensors/#radar-sensor), with radial velocity                      | 20 Hz |
| `camera.pcam_*`            | Six [RGB cameras](https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera), JPEG-encoded                                                 | 4 Hz  |
| `camera_depth.pcam_*`      | [Depth cameras](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera), 8-bit linear quantization saturating at 50 m                 | 4 Hz  |
| `camera_semantic.pcam_*`   | Semantic class channel of the [instance segmentation cameras](https://carla.readthedocs.io/en/latest/ref_sensors/#instance-segmentation-camera) | 4 Hz  |
| `camera_instance.pcam_*`   | Instance id channel of the same cameras                                                                                                         | 4 Hz  |
| `traffic_light_detections` | Traffic-light states, linked to map lanes                                                                                                       | 20 Hz |
| `custom.driving_meta`      | CARLA-specific meta information, see below                                                                                                      | 20 Hz |
| `sync`                     | One row per tick; defines the scene iterations                                                                                                  | 20 Hz |

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
is read through LEAD's `SceneLoader`: it pairs both views per scene,
picks the perturbated sensors with `perturbation_probability`, re-projects ego-frame
outputs such as the target points, and reports the rig offset as
`scene_data.rig_perturbation`.

## The `driving_meta` stream

`custom.driving_meta` is a py123d custom modality holding the CARLA-specific
state as a plain dict per tick. The core fields are what LEAD's loader and the
label pipelines consume; the diagnostic fields are the expert's internal state,
kept for analysis and visualization.

### Core fields

#### Localization

| Property                  | Type                          | Description                                                                                                         |
| :------------------------ | :---------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| `localized_ego_state_se3` | `numpy.typing.NDArray (4, 4)` | SE(3) pose from GNSS+IMU fusion in the global frame — the noisy pose the policy observes (yaw-only rotation, z = 0) |

#### Route & Navigation

| Property               | Type                          | Description                                                                                 |
| :--------------------- | :---------------------------- | :------------------------------------------------------------------------------------------ |
| `target_point_indices` | `dict[str, int]`              | Current target waypoint index per distance; keys are distance strings (e.g., "5.0", "10.0") |
| `route`                | `numpy.typing.NDArray (M, 2)` | Dense future route waypoints in global coordinates (M waypoints)                            |
| `route_original`       | `numpy.typing.NDArray (M, 2)` | Original output of A\* before any modification (static obstacles)                           |
| `changed_route`        | `bool`                        | True if route was modified from the original                                                |

#### Speed & Control

| Property                            | Type    | Description                                        |
| :---------------------------------- | :------ | :------------------------------------------------- |
| `target_speed`                      | `float` | Desired speed (m/s)                                |
| `speed_limit`                       | `float` | Road speed limit (m/s)                             |
| `target_speed_limit`                | `float` | Target speed limit for upcoming road segment (m/s) |
| `last_encountered_speed_limit_sign` | `float` | Speed limit from the most recent sign (m/s)        |
| `steer`                             | `float` | Steering angle executed this tick (radians)        |
| `throttle`                          | `float` | Throttle pedal value (0–1)                         |
| `brake`                             | `bool`  | True if brake was applied                          |

#### Scenario

| Property                        | Type  | Description                                                              |
| :------------------------------ | :---- | :----------------------------------------------------------------------- |
| `scenario`                      | `str` | CARLA Leaderboard scenario type (e.g., "Accident", "PedestrianCrossing") |
| `current_active_scenario_type`  | `str` | Currently active scenario type                                           |
| `previous_active_scenario_type` | `str` | Previously active scenario type                                          |

#### Metadata

| Property                 | Type              | Description                                                                                            |
| :----------------------- | :---------------- | :----------------------------------------------------------------------------------------------------- |
| `jpeg_storage_quality`   | `int`             | JPEG compression quality (0–100)                                                                       |
| `box_attributes`         | `dict[str, dict]` | Per-box metadata keyed by py123d track token (the stringified CARLA actor id)                          |
| ├─ (user-defined fields) | Various           | Any non-native fields from box detections (occlusion counts, `affects_ego`, traffic-light state, etc.) |

### Diagnostic fields

#### Hazards & Safety

| Property                               | Type            | Description                                                        |
| :------------------------------------- | :-------------- | :----------------------------------------------------------------- |
| `vehicle_hazard`                       | `bool`          | True if a vehicle poses an immediate hazard                        |
| `vehicle_affecting_id`                 | `int`           | Actor ID of the affecting vehicle (or -1 if none)                  |
| `light_hazard`                         | `bool`          | True if a traffic light requires action                            |
| `walker_hazard`                        | `bool`          | True if a pedestrian poses an immediate hazard                     |
| `walker_affecting_id`                  | `int`           | Actor ID of the affecting pedestrian                               |
| `walker_close`                         | `bool`          | True if a pedestrian is nearby (not necessarily hazard)            |
| `walker_close_id`                      | `int`           | Actor ID of the nearby pedestrian                                  |
| `stop_sign_hazard`                     | `bool`          | True if a stop sign requires action                                |
| `stop_sign_close`                      | `bool`          | True if a stop sign is within proximity                            |
| `emergency_brake_for_special_vehicle`  | `bool`          | True if emergency brake triggered for special vehicles             |
| `does_emergency_brake_for_pedestrians` | `bool`          | True if emergency brake was applied for pedestrians                |
| `rear_danger_8`                        | `bool`          | True if rear collision risk within 8 m                             |
| `rear_danger_16`                       | `bool`          | True if rear collision risk within 16 m                            |
| `rear_adversarial_id`                  | `int`           | Actor ID of rear-following vehicle (or -1 if none)                 |
| `brake_cutin`                          | `bool`          | True if braking for a cut-in vehicle                               |
| `speed_reduced_by_obj_type`            | `str \| None`   | Type of object causing speed reduction ("vehicle", "walker", etc.) |
| `speed_reduced_by_obj_id`              | `int \| None`   | Actor ID of object causing speed reduction                         |
| `speed_reduced_by_obj_distance`        | `float \| None` | Distance to object causing speed reduction (m)                     |

#### Route Progress

| Property                | Type    | Description                                        |
| :---------------------- | :------ | :------------------------------------------------- |
| `route_left_length`     | `float` | Distance remaining along the route (m)             |
| `distance_ego_to_route` | `float` | Lateral distance from ego to the planned route (m) |

#### Road & Lane Context

| Property                             | Type            | Description                                                                                                          |
| :----------------------------------- | :-------------- | :------------------------------------------------------------------------------------------------------------------- |
| `road_id`                            | `int`           | Current road ID from the map                                                                                         |
| `lane_id`                            | `int`           | Current lane ID from the map                                                                                         |
| `ego_lane_id`                        | `int`           | Ego's lane ID                                                                                                        |
| `is_junction`                        | `bool`          | True if ego is in a junction                                                                                         |
| `junction_id`                        | `int`           | Junction ID if in a junction                                                                                         |
| `next_road_ids`                      | `list[int]`     | Road IDs of the next lanes                                                                                           |
| `next_next_road_ids_ego`             | `list[int]`     | Road IDs of lanes after next                                                                                         |
| `lane_change_str`                    | `str`           | Lane change availability (e.g., "BOTH", "LEFT", "RIGHT", "NONE")                                                     |
| `lane_type_str`                      | `str`           | Lane type (e.g., "Driving", "Shoulder", "Sidewalk")                                                                  |
| `left_lane_marking_type_str`         | `str`           | Left lane marking type (e.g., "SOLID", "DASHED", "NONE")                                                             |
| `left_lane_marking_color_str`        | `str`           | Left lane marking color (e.g., "White", "Yellow")                                                                    |
| `right_lane_marking_type_str`        | `str`           | Right lane marking type                                                                                              |
| `right_lane_marking_color_str`       | `str`           | Right lane marking color                                                                                             |
| `ego_lane_width`                     | `float`         | Width of the current lane (m)                                                                                        |
| `target_lane_width`                  | `float`         | Width of the target lane (m)                                                                                         |
| `dist_to_junction`                   | `float \| None` | Distance to the junction at the end of the current lane (m); 0 inside a junction, None when the next lane is not one |
| `distance_to_next_junction`          | `float \| None` | Distance to the next junction along the lane the ego drives (m)                                                      |
| `distance_to_intersection_index_ego` | `float`         | Distance to the intersection point of the active junction scenario (m); `inf` outside such a scenario                |

#### Scenario Internals

| Property                         | Type        | Description                               |
| :------------------------------- | :---------- | :---------------------------------------- |
| `scenario_actors_ids`            | `list[int]` | Actor IDs involved in the active scenario |
| `scenario_obstacles_ids`         | `list[int]` | Obstacle actor IDs from the scenario      |
| `scenario_obstacles_convex_hull` | `list`      | Convex hull of scenario obstacles         |

#### Obstacle Distances

| Property                     | Type        | Description                                |
| :--------------------------- | :---------- | :----------------------------------------- |
| `dist_to_construction_site`  | `float`     | Distance to construction site obstacle (m) |
| `dist_to_accident_site`      | `float`     | Distance to accident site obstacle (m)     |
| `dist_to_parked_obstacle`    | `float`     | Distance to parked vehicle (m)             |
| `dist_to_vehicle_opens_door` | `float`     | Distance to vehicle with open door (m)     |
| `dist_to_cutin_vehicle`      | `float`     | Distance to cut-in vehicle (m)             |
| `dist_to_pedestrian`         | `float`     | Distance to pedestrian (m)                 |
| `dist_to_biker`              | `float`     | Distance to cyclist (m)                    |
| `cut_in_actors_ids`          | `list[int]` | Actor IDs of vehicles that cut in          |

#### Stuck-State Flags

| Property                               | Type   | Description                            |
| :------------------------------------- | :----- | :------------------------------------- |
| `construction_obstacle_two_ways_stuck` | `bool` | True if stuck by construction obstacle |
| `accident_two_ways_stuck`              | `bool` | True if stuck by accident obstacle     |
| `parked_obstacle_two_ways_stuck`       | `bool` | True if stuck by parked obstacle       |
| `vehicle_opens_door_two_ways_stuck`    | `bool` | True if stuck by door-opening vehicle  |

#### Visibility & Adversarial

| Property                    | Type   | Description                                                                              |
| :-------------------------- | :----- | :--------------------------------------------------------------------------------------- |
| `visual_visibility`         | `int`  | Weather visibility class (`WeatherVisibility`: 0 clear, 1 ok, 2 limited, 3 very limited) |
| `slower_bad_visibility`     | `bool` | True if ego slowed down due to low visibility                                            |
| `slower_clutterness`        | `bool` | True if ego slowed down due to scene clutter                                             |
| `num_dangerous_adversarial` | `int`  | Count of dangerous adversarial actors                                                    |
| `num_safe_adversarial`      | `int`  | Count of safe adversarial actors                                                         |
| `num_ignored_adversarial`   | `int`  | Count of ignored adversarial actors                                                      |

#### Traffic Lights & Environment

| Property                  | Type   | Description                                                      |
| :------------------------ | :----- | :--------------------------------------------------------------- |
| `europe_traffic_light`    | `bool` | True if traffic lights follow European convention                |
| `over_head_traffic_light` | `bool` | True if traffic lights are overhead-mounted                      |
| `weather_setting`         | `str`  | Name of the CARLA weather preset (e.g. "ClearNoon")              |
| `weather_parameters`      | `dict` | Detailed weather values (precipitation, clouds, wind, fog, etc.) |

#### Vehicle Door

| Property                            | Type   | Description                                               |
| :---------------------------------- | :----- | :-------------------------------------------------------- |
| `vehicle_opened_door`               | `bool` | True if a vehicle opened a door                           |
| `vehicle_door_side`                 | `str`  | Which side door opened ("left", "right", "both", or None) |
| `num_parking_vehicles_in_proximity` | `int`  | Count of parked vehicles nearby                           |

#### Lane Change & Ego State

| Property                     | Type    | Description                                   |
| :--------------------------- | :------ | :-------------------------------------------- |
| `signed_dist_to_lane_change` | `float` | Signed distance to next lane change point (m) |

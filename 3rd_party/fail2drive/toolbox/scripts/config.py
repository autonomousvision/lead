SCENARIO_TYPES = {
    # Junction scenarios
    "SignalizedJunctionLeftTurn": [
        ["flow_speed", "value", 20],
        ["source_dist_interval", "interval", [25, 50]],
    ],
    "SignalizedJunctionRightTurn": [
        ["flow_speed", "value", 20],
        ["source_dist_interval", "interval", [25, 50]],
    ],
    "OppositeVehicleRunningRedLight": [
        ["direction", "choice"],
    ],
    "NonSignalizedJunctionLeftTurn": [
        ["flow_speed", "value", 20],
        ["source_dist_interval", "interval", [25, 50]],
    ],
    "NonSignalizedJunctionRightTurn": [
        ["flow_speed", "value", 20],
        ["source_dist_interval", "interval", [25, 50]],
    ],
    "OppositeVehicleTakingPriority": [
        ["direction", "choice"],
    ],
    # Crossing actors
    "DynamicObjectCrossing": [
        ["distance", "value", 12],
        [
            "direction",
            "choice",
        ],  # intially this was of type value, but the class implementation only accepts right or false
        ["blocker_model", "value", "static.prop.vendingmachine"],
        ["crossing_angle", "value", 0],
        ["walker", "value", "walker.pedestrian.*"],
    ],
    "ParkingCrossingPedestrian": [
        ["distance", "value", 12],
        ["direction", "choice"],
        ["crossing_angle", "value", 0],
    ],
    "PedestrianCrossing": [
        ["walker", "value", "walker.pedestrian.*"],
    ],
    "VehicleTurningRoute": [],
    "VehicleTurningRoutePedestrian": [
        ["walker", "value", "walker.pedestrian.*"],
    ],
    "BlockedIntersection": [],
    # Actor flows
    "EnterActorFlow": [
        ["start_actor_flow", "location driving"],
        ["end_actor_flow", "location driving"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    "EnterActorFlowV2": [
        ["start_actor_flow", "location driving"],
        ["end_actor_flow", "location driving"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    "InterurbanActorFlow": [
        ["start_actor_flow", "location driving"],
        ["end_actor_flow", "location driving"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    "InterurbanAdvancedActorFlow": [
        ["start_actor_flow", "location driving"],
        ["end_actor_flow", "location driving"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    "HighwayExit": [
        ["start_actor_flow", "location driving"],
        ["end_actor_flow", "location driving"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    "MergerIntoSlowTraffic": [
        ["start_actor_flow", "location driving"],
        ["end_actor_flow", "location driving"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    "MergerIntoSlowTrafficV2": [
        ["start_actor_flow", "location driving"],
        ["end_actor_flow", "location driving"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    "CrossingBicycleFlow": [
        ["start_actor_flow", "location bicycle"],
        ["flow_speed", "value", 10],
        ["source_dist_interval", "interval", [20, 50]],
    ],
    # Route obstacles
    "ConstructionObstacle": [
        ["distance", "value", 100],
        [
            "direction",
            "choice",
        ],  # intially this was of type value, but the class implementation only accepts right or false
        ["speed", "value", 60],
    ],
    "PermutedConstructionObstacle": [
        ["distance", "value", 100],
        ["direction", "choice"],
        ["speed", "value", 60],
        ["warning_sign", "value", "static.prop.trafficwarning"],
        ["debris", "value", "static.prop.dirtdebris02"],
        ["cones", "value", "1111111"],
    ],
    "ConstructionObstacleTwoWays": [
        ["distance", "value", 100],
        ["frequency", "interval", [20, 100]],
    ],
    "PermutedConstructionObstacleTwoWays": [
        ["distance", "value", 100],
        ["frequency", "interval", [20, 100]],
        ["warning_sign", "value", "static.prop.trafficwarning"],
        ["debris", "value", "static.prop.dirtdebris02"],
        ["cones", "value", "1111111"],
    ],
    "Accident": [
        ["distance", "value", 120],
        [
            "direction",
            "choice",
        ],  # intially this was of type value, but the class implementation only accepts right or false
        ["speed", "value", 60],
    ],
    "AccidentTwoWays": [
        ["distance", "value", 120],
        ["frequency", "interval", [20, 100]],
    ],
    "ParkedObstacle": [
        ["distance", "value", 120],
        [
            "direction",
            "choice",
        ],  # intially this was of type value, but the class implementation only accepts right or false
        ["speed", "value", 60],
    ],
    "BadParkingObstacle": [
        ["distance", "value", 120],
        ["direction", "choice"],
        ["speed", "value", 60],
        ["vehicle", "value", "vehicle.*"],
        ["x", "value", 0],
        ["y", "value", 1.5],
        ["yaw", "value", 0],
    ],
    "ParkedObstacleTwoWays": [
        ["distance", "value", 120],
        ["frequency", "interval", [20, 100]],
    ],
    "BadParkingObstacleTwoWays": [
        ["distance", "value", 120],
        ["frequency", "interval", [20, 100]],
        ["vehicle", "value", "vehicle.*"],
        ["x", "value", 0],
        ["y", "value", 1.5],
        ["yaw", "value", 0],
    ],
    "VehicleOpensDoorTwoWays": [
        ["distance", "value", 50],
        ["direction", "choice"],
        ["speed", "value", 60],
        ["frequency", "interval", [20, 100]],
    ],
    "HazardAtSideLane": [
        ["distance", "value", 100],
        ["speed", "value", 60],
        ["bicycle_drive_distance", "value", 50],
        ["bicycle_speed", "value", 10],
    ],
    "HazardAtSideLaneTwoWays": [
        ["distance", "value", 100],
        ["frequency", "value", 100],
        ["bicycle_drive_distance", "value", 50],
        ["bicycle_speed", "value", 10],
    ],
    "InvadingTurn": [
        ["distance", "value", 100],
        ["offset", "value", 0.25],
    ],
    "ConstructionObstaclePedestrian": [
        ["distance", "value", 100],
        ["direction", "choice"],
        ["speed", "value", 60],
        ["pedestrian", "value", "walker.pedestrian.0030"],
    ],
    "ConstructionObstacleRightLane": [
        ["distance", "value", 100],
        ["y_offset", "value", 0.0],
    ],
    "ConstructionObstacleOppositeLane": [
        ["distance", "value", 100],
    ],
    "CustomObstacle": [
        ["distance", "value", 100],
    ],
    "CustomObstacleTwoWays": [
        ["distance", "value", 100],
        ["frequency", "interval", [20, 100]],
    ],
    "RoadBlocked": [
        ["distance", "value", 100],
        ["wait", "value", 60],
    ],
    "ImageOnObject": [
        ["distance", "value", 10],
        ["prop", "value", "static.prop.advertisement"],
        ["image", "value", "static.prop.barrel"],
        ["offset_x", "value", 0.0],
        ["offset_y", "value", 0.0],
        ["offset_z", "value", 0.0],
    ],
    "ObscuredStopSign": [
        ["prop", "value", "static.prop.barrel"],
        ["offset_y", "value", 0.0],
        ["offset_z", "value", 0.0],
    ],
    "PedestrianCrowd": [
        ["pedestrians", "value", 20],
        ["pedestrian_center", "value", 40],
        ["pedestrian_radius", "value", 20],
        ["side", "choice"],
    ],
    "PedestriansOnRoad": [
        ["distance", "value", 20],
        ["pedestrians", "value", 3],
        ["walker_duration", "value", 20],
        ["walker_speed", "value", 2],
    ],
    "HardBrakeNoLights": [],
    "NormalVehicleRunningRedLight": [
        ["direction", "choice"],
        ["vehicle", "value", ""],
    ],
    "NormalVehicleTakingPriority": [
        ["direction", "choice"],
        ["vehicle", "value", ""],
    ],
    # Cut ins
    "HighwayCutIn": [
        ["other_actor_location", "location driving"],
    ],
    "ParkingCutIn": [
        ["direction", "choice"],
    ],
    "StaticCutIn": [
        ["distance", "value", 100],
        ["direction", "choice"],
    ],
    # Others
    "ControlLoss": [],
    "HardBreakRoute": [],
    "ParkingExit": [
        ["direction", "choice"],
        ["front_vehicle_distance", "value", 20],
        ["behind_vehicle_distance", "value", 10],
        ["flow_distance", "value", 25],
        ["speed", "value", 60],
    ],
    "YieldToEmergencyVehicle": [
        ["distance", "value", 140],
    ],
    # Special ones
    "BackgroundActivityParametrizer": [
        ["num_front_vehicles", "value"],  # there are no default parameters for this scenario
        ["num_back_vehicles", "value"],
        ["road_spawn_dist", "value"],
        ["opposite_source_dist", "value"],
        ["opposite_max_actors", "value"],
        ["opposite_spawn_dist", "value"],
        ["opposite_active", "bool"],
        ["junction_source_dist", "value"],
        ["junction_max_actors", "value"],
        ["junction_spawn_dist", "value"],
        ["junction_source_perc", "value"],
    ],
    "PriorityAtJunction": [],
}

SCENARIO_TOOLTIPS = {
    "SignalizedJunctionLeftTurn": "Ego turns left at a signalized junction with crossing flow.",
    "SignalizedJunctionRightTurn": "Ego turns right at a signalized junction with crossing flow.",
    "OppositeVehicleRunningRedLight": "An oncoming vehicle runs a red light at the junction.",
    "NonSignalizedJunctionLeftTurn": "Ego turns left at an unsignalized junction with crossing flow.",
    "NonSignalizedJunctionRightTurn": "Ego turns right at an unsignalized junction with crossing flow.",
    "OppositeVehicleTakingPriority": "An oncoming vehicle takes priority at an unsignalized junction.",
    "DynamicObjectCrossing": "A pedestrian crosses near a blocker placed at the roadside.",
    "ParkingCrossingPedestrian": "A parked vehicle occludes a pedestrian crossing ahead.",
    "PedestrianCrossing": "Multiple pedestrians cross near a junction entry.",
    "VehicleTurningRoute": "Ego turns along route and encounters a crossing cyclist.",
    "VehicleTurningRoutePedestrian": "Ego turns along route and encounters a crossing pedestrian.",
    "BlockedIntersection": "A blocker vehicle appears after a turn and blocks the lane.",
    "EnterActorFlow": "Ego must enter/merge while a continuous same-direction vehicle flow passes through the conflict area.",
    "EnterActorFlowV2": "Ego merges from a dedicated entry lane while a continuous vehicle flow passes through the merge corridor and route-exit area.",
    "InterurbanActorFlow": "Ego turns left off an interurban road and crosses an oncoming actor flow at an intersection.",
    "InterurbanAdvancedActorFlow": "Interurban left-turn plus merge: ego first crosses one flow, then merges into a second adjacent flow.",
    "HighwayExit": "Ego exits a highway while background flow in a neighboring lane creates pressure to merge/extract correctly.",
    "MergerIntoSlowTraffic": "Ego merges into an existing slow same-direction traffic stream near a confluence area.",
    "MergerIntoSlowTrafficV2": "MergerIntoSlowTraffic variant with stricter exit-lane handling after the merge region.",
    "CrossingBicycleFlow": "Ego crosses a junction where bicycles flow through; placement must match a valid bike-lane path across and out of the junction.",
    "ConstructionObstacle": "Construction setup blocks lane and forces a lane change.",
    "PermutedConstructionObstacle": "Customizable construction setup with modified obstacle layout.",
    "ConstructionObstacleTwoWays": "Construction obstacle requiring temporary opposite-lane invasion.",
    "PermutedConstructionObstacleTwoWays": "Customizable two-way construction obstacle variant.",
    "Accident": "Accident scene blocks lane and forces lane change.",
    "AccidentTwoWays": "Accident scene requiring temporary opposite-lane invasion.",
    "ParkedObstacle": "Incorrectly parked vehicle blocks lane and forces lane change.",
    "BadParkingObstacle": "Custom badly parked vehicle setup that blocks route.",
    "ParkedObstacleTwoWays": "Parked obstacle requiring temporary opposite-lane invasion.",
    "BadParkingObstacleTwoWays": "Custom bad-parking two-way variant.",
    "VehicleOpensDoorTwoWays": "Parked vehicle opens door, forcing evasive maneuver into opposite lane.",
    "HazardAtSideLane": "Side-lane bicycles create a moving roadside hazard.",
    "HazardAtSideLaneTwoWays": "Two-way variant of side-lane bicycle hazard.",
    "InvadingTurn": "Temporary invading-turn behavior with route offset.",
    "ConstructionObstaclePedestrian": "Construction obstacle combined with a crossing pedestrian.",
    "ConstructionObstacleRightLane": "Construction-like objects placed on right lane side.",
    "ConstructionObstacleOppositeLane": "Construction-like objects placed in opposite lane area.",
    "CustomObstacle": "Fully custom object placement anchored to scenario trigger.",
    "CustomObstacleTwoWays": "Two-way custom object placement scenario.",
    "RoadBlocked": "Road is blocked by custom objects and ego waits then proceeds.",
    "ImageOnObject": "Spawn an image-like prop attached to a roadside prop.",
    "ObscuredStopSign": "Place prop to occlude an upcoming stop sign.",
    "PedestrianCrowd": "Spawn a crowd of pedestrians near roadside sidewalk area.",
    "PedestriansOnRoad": "Spawn pedestrians walking on road ahead of ego.",
    "HardBrakeNoLights": "Background front vehicles brake hard without brake lights.",
    "NormalVehicleRunningRedLight": "Normal (non-emergency) vehicle runs red light at junction.",
    "NormalVehicleTakingPriority": "Normal (non-emergency) vehicle takes priority at junction.",
    "HighwayCutIn": "A highway vehicle cuts in from another lane in front of ego.",
    "ParkingCutIn": "A parked vehicle cuts into ego lane at close range.",
    "StaticCutIn": "Static blocker + cut-in interaction scenario.",
    "ControlLoss": "Ego experiences temporary control disturbances over debris.",
    "HardBreakRoute": "Background front vehicles perform a hard brake event.",
    "ParkingExit": "Ego starts from parking position and merges into traffic.",
    "YieldToEmergencyVehicle": "Ego must yield lane to a faster emergency vehicle.",
    "BackgroundActivityParametrizer": "Adjust background traffic behavior parameters on route.",
    "PriorityAtJunction": "Force ego traffic light priority at the next junction.",
}

PARAM_TOOLTIPS_DEFAULT = {
    "flow_speed": "Target flow speed in km/h.",
    "source_dist_interval": "Spawn distance interval [from, to] in meters for flow actors.",
    "direction": "Side/direction choice used by the scenario logic.",
    "distance": "Distance ahead of trigger point in meters.",
    "blocker_model": "CARLA blueprint id for blocker object/prop.",
    "crossing_angle": "Crossing angle in degrees (typically between -90 and 90).",
    "start_actor_flow": "Start location for actor flow spawn lane.",
    "end_actor_flow": "End/sink location for actor flow.",
    "frequency": "Two-way flow spacing/frequency parameter in meters.",
    "speed": "Scenario-specific speed limit in km/h.",
    "bicycle_drive_distance": "Distance bicycles should travel in meters.",
    "bicycle_speed": "Bicycle speed in km/h.",
    "offset": "Lateral/route offset amount used by scenario behavior.",
    "wait": "Wait duration in seconds before scenario continues.",
    "pedestrians": "Number of pedestrians to spawn.",
    "other_actor_location": "Spawn location for the other actor vehicle.",
    "front_vehicle_distance": "Distance to front parked blocker in meters.",
    "behind_vehicle_distance": "Distance to rear parked blocker in meters.",
    "num_front_vehicles": "Number of background vehicles in front of ego.",
    "num_back_vehicles": "Number of background vehicles behind ego.",
    "road_spawn_dist": "Background spawn distance on ego road.",
    "opposite_source_dist": "Opposite-lane source distance for background traffic.",
    "opposite_max_actors": "Maximum opposite-lane background actors.",
    "opposite_spawn_dist": "Opposite-lane spawn distance.",
    "opposite_active": "Enable or disable opposite-lane background flow.",
    "junction_source_dist": "Junction source distance for background traffic.",
    "junction_max_actors": "Maximum junction background actors.",
    "junction_spawn_dist": "Junction spawn distance.",
    "junction_source_perc": "Percentage of active junction sources.",
    "vehicle": "CARLA vehicle blueprint id (empty/random where supported).",
    "x": "Longitudinal offset in meters in scenario-local frame.",
    "y": "Lateral offset in meters in scenario-local frame.",
    "yaw": "Yaw offset in degrees.",
    "walker": "CARLA walker blueprint id.",
    "pedestrian": "CARLA pedestrian blueprint id.",
    "y_offset": "Additional lateral offset in meters.",
    "warning_sign": "Blueprint id for warning sign prop, or 'none'.",
    "debris": "Blueprint id for debris prop, or 'none'.",
    "cones": "Bitmask string for cone placement (e.g. 1111111).",
    "prop": "CARLA prop blueprint id.",
    "image": "CARLA prop blueprint id used as the image object.",
    "offset_x": "Forward/backward offset in meters.",
    "offset_y": "Right/left offset in meters.",
    "offset_z": "Vertical offset in meters.",
    "pedestrian_center": "Center distance ahead for crowd placement in meters.",
    "pedestrian_radius": "Crowd radius in meters.",
    "side": "Road side used by the scenario.",
    "walker_duration": "Pedestrian movement duration in seconds.",
    "walker_speed": "Pedestrian speed in m/s.",
    "flow_distance": "Background road spawn distance in meters.",
}

SCENARIO_PARAM_TOOLTIPS = {
    "BadParkingObstacle": {
        "distance": "Distance to anchor bad parking setup in meters.",
        "direction": "Relevant side for lane/context interpretation.",
        "speed": "Speed cap during scenario handling in km/h.",
    },
    "BadParkingObstacleTwoWays": {
        "distance": "Distance to anchor bad parking setup in meters.",
        "frequency": "Opposite flow spacing interval [from, to] in meters.",
    },
    "ConstructionObstacleRightLane": {
        "distance": "Distance to construction setup in meters.",
    },
    "ConstructionObstacleOppositeLane": {
        "distance": "Distance to opposite-lane setup in meters.",
    },
    "CustomObstacle": {
        "distance": "Distance to custom object cluster anchor in meters.",
    },
    "CustomObstacleTwoWays": {
        "distance": "Distance to custom object cluster anchor in meters.",
        "frequency": "Opposite flow spacing interval [from, to] in meters.",
    },
    "RoadBlocked": {
        "distance": "Distance to blocked segment in meters.",
        "wait": "How long ego waits before scenario ends, in seconds.",
    },
    "ImageOnObject": {
        "distance": "Distance to spawned prop/image setup in meters.",
    },
    "PedestrianCrowd": {
        "pedestrians": "Number of pedestrians in the crowd.",
    },
    "PedestriansOnRoad": {
        "distance": "Distance to first pedestrian spawn in meters.",
        "pedestrians": "Number of pedestrians to place on road.",
    },
    "ParkingCutIn": {
        "direction": "Side from which parked actors are selected.",
    },
    "ParkingExit": {
        "direction": "Parking side relative to ego lane.",
        "front_vehicle_distance": "Distance to front parked blocker in meters.",
        "behind_vehicle_distance": "Distance to rear parked blocker in meters.",
    },
    "YieldToEmergencyVehicle": {
        "distance": "Distance behind ego where emergency vehicle spawns, in meters.",
    },
    "HighwayCutIn": {
        "other_actor_location": "Spawn location of the cut-in vehicle.",
    },
    "CrossingBicycleFlow": {
        "start_actor_flow": "Bike-lane source point for the bicycle flow.",
    },
    "EnterActorFlow": {
        "start_actor_flow": "Source lane waypoint for the vehicle flow.",
        "end_actor_flow": "Downstream sink waypoint for the same flow.",
    },
    "EnterActorFlowV2": {
        "start_actor_flow": "Source lane waypoint for the flow in the dedicated merge setup.",
        "end_actor_flow": "Downstream sink waypoint on the same flow corridor.",
    },
    "HighwayExit": {
        "start_actor_flow": "Highway flow source waypoint.",
        "end_actor_flow": "Highway flow sink waypoint.",
    },
    "InterurbanActorFlow": {
        "start_actor_flow": "Source waypoint of the crossing interurban flow.",
        "end_actor_flow": "Sink waypoint of that interurban flow.",
    },
    "InterurbanAdvancedActorFlow": {
        "start_actor_flow": "Primary flow source waypoint.",
        "end_actor_flow": "Primary flow sink waypoint.",
    },
    "MergerIntoSlowTraffic": {
        "start_actor_flow": "Source waypoint of the slow traffic stream.",
        "end_actor_flow": "Sink waypoint of the same stream.",
    },
    "MergerIntoSlowTrafficV2": {
        "start_actor_flow": "Source waypoint of the slow traffic stream (V2).",
        "end_actor_flow": "Sink waypoint of the same stream (V2).",
    },
}

SCENARIO_PARAM_PLACEMENT_HINTS = {
    "HighwayCutIn": {
        "other_actor_location": "Place this on a valid driving lane with enough forward lane to compute the cut-in target (~10 m ahead). Meaning: this is where the other vehicle starts before cutting left into ego lane.",
    },
    "CrossingBicycleFlow": {
        "start_actor_flow": "Must be on a bike lane (close to the selected point) and allow a continuous route that enters the junction, exits it, and continues afterward. Meaning: this defines the bicycle stream ego must cross.",
    },
    "EnterActorFlow": {
        "start_actor_flow": "Place on the source lane where flow vehicles should spawn before reaching ego's merge/conflict zone.",
        "end_actor_flow": "Place downstream on the same traffic direction so ActorFlow can continuously run from source to sink. Together, these points define the pressure corridor ego must merge through.",
    },
    "EnterActorFlowV2": {
        "start_actor_flow": "Place on the through-traffic lane before the dedicated merge area. This is where challenge traffic spawns.",
        "end_actor_flow": "Place downstream on the same drivable corridor after the merge/conflict zone. Start and end must define one continuous same-direction flow path.",
    },
    "HighwayExit": {
        "start_actor_flow": "Place on the highway lane that should create pressure before ego exits.",
        "end_actor_flow": "Place further downstream on that same corridor so flow keeps moving past the exit region.",
    },
    "InterurbanActorFlow": {
        "start_actor_flow": "Place on the source lane of the crossing interurban flow ego must pass through while turning left.",
        "end_actor_flow": "Place on the sink lane of that same crossing stream after the intersection.",
    },
    "InterurbanAdvancedActorFlow": {
        "start_actor_flow": "Primary flow source. Special rule: its left adjacent driving lane is reused as sink of flow 2.",
        "end_actor_flow": "Primary flow sink. Special rule: its left adjacent driving lane is reused as source of flow 2.",
    },
    "MergerIntoSlowTraffic": {
        "start_actor_flow": "Source of the slow traffic stream ego merges into.",
        "end_actor_flow": "Downstream sink on the same stream; choose a stable same-direction corridor through the merge region.",
    },
    "MergerIntoSlowTrafficV2": {
        "start_actor_flow": "Source of the slow traffic stream ego merges into (V2 has stricter post-merge route-exit handling).",
        "end_actor_flow": "Downstream sink on the same stream; keep geometry consistent for a clean post-merge continuation.",
    },
}


def get_scenario_tooltip(scenario_type):
    return SCENARIO_TOOLTIPS.get(scenario_type, "No description available.")


def get_scenario_param_tooltip(scenario_type, parameter_name):
    scenario_map = SCENARIO_PARAM_TOOLTIPS.get(scenario_type, {})
    return scenario_map.get(parameter_name, PARAM_TOOLTIPS_DEFAULT.get(parameter_name, "Scenario parameter."))


def get_scenario_param_placement_hint(scenario_type, parameter_name):
    scenario_map = SCENARIO_PARAM_PLACEMENT_HINTS.get(scenario_type, {})
    return scenario_map.get(parameter_name, get_scenario_param_tooltip(scenario_type, parameter_name))


# Generalization type -> baseline type used for paired export.
# Use `None` for "baseline is route without this scenario".
FAIL2DRIVE_EXPORT_BASELINE_BY_TYPE = {
    "PermutedConstructionObstacle": "ConstructionObstacle",
    "PermutedConstructionObstacleTwoWays": "ConstructionObstacleTwoWays",
    "BadParkingObstacle": "ParkedObstacle",
    "BadParkingObstacleTwoWays": "ParkedObstacleTwoWays",
    "ConstructionObstaclePedestrian": "ConstructionObstacle",
    "HardBrakeNoLights": "HardBreakRoute",
    "NormalVehicleRunningRedLight": "OppositeVehicleRunningRedLight",
    "NormalVehicleTakingPriority": "OppositeVehicleTakingPriority",
    "ConstructionObstacleRightLane": None,
    "ConstructionObstacleOppositeLane": None,
    "ImageOnObject": None,
    "ObscuredStopSign": None,
    "PedestriansOnRoad": None,
    "PedestrianCrowd": None,
    "RoadBlocked": None,
    # NOTE: CustomObstacle variants appear in multiple benchmark categories.
    # Defaulting to construction as baseline; adjust if your fixed mapping differs.
    "CustomObstacle": "ConstructionObstacle",
    "CustomObstacleTwoWays": "ConstructionObstacleTwoWays",
}


# Pedestrian scenario types that have a "walker" parameter.
# When the walker is set to an animal blueprint (walker.animal.*),
# the baseline replaces it with a generic pedestrian (walker.pedestrian.*).
PEDESTRIAN_WALKER_SCENARIO_TYPES = {
    "DynamicObjectCrossing",
    "PedestrianCrossing",
    "VehicleTurningRoutePedestrian",
}


# Scenarios that should open the shared graphical editor.
# Presets are shared via the graphical editor preset store.
GRAPHICAL_EDITOR_SCENARIO_TYPES = {
    "CustomObstacle",
    "CustomObstacleTwoWays",
    "RoadBlocked",
    "BadParkingObstacle",
    "PermutedConstructionObstacle",
}

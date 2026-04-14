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
    ],
    "ParkingCrossingPedestrian": [
        ["distance", "value", 12],
        ["direction", "choice"],
        ["crossing_angle", "value", 0],
    ],
    "PedestrianCrossing": [],
    "VehicleTurningRoute": [],
    "VehicleTurningRoutePedestrian": [],
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
    "ConstructionObstacleTwoWays": [
        ["distance", "value", 100],
        ["frequency", "interval", [20, 100]],
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
    "ParkedObstacleTwoWays": [
        ["distance", "value", 120],
        ["frequency", "interval", [20, 100]],
    ],
    "VehicleOpensDoorTwoWays": [
        ["distance", "value", 50],
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

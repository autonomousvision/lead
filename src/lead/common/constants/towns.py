"""CARLA town names, map paths and per-area speed limits."""

URBAN_MAX_SPEED_LIMIT = 15
SUBURBAN_MAX_SPEED_LIMIT = 25
HIGHWAY_MAX_SPEED_LIMIT = 35

CARLA_MAP_PATHS = {
    **{
        location.lower(): f"CarlaUE4/Content/Carla/Maps/OpenDrive/{location}.xodr"
        for location in [
            "Town01",
            "Town02",
            "Town03",
            "Town04",
            "Town05",
            "Town06",
            "Town07",
            "Town10HD",
        ]
    },
    **{
        location.lower(): f"CarlaUE4/Content/Carla/Maps/{location}/OpenDrive/{location}.xodr"
        for location in ["Town11", "Town12", "Town13", "Town15"]
    },
}

OLD_TOWNS = {
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
}

# List of all CARLA towns for logging purposes
ALL_TOWNS = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
]

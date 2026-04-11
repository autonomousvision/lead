from dataclasses import dataclass

import config


@dataclass(frozen=True)
class ScenarioEditorProfile:
    kind: str  # "graphical" | "text" | "none"
    object_mode: str = "any"  # for graphical: "any" | "vehicle_only" | "props_only" | "construction"
    overlay_types: tuple[str, ...] = ("Construction", "Accident", "ParkedVehicle")


_EXPLICIT_PROFILES = {
    # Graphical scenarios
    "CustomObstacle": ScenarioEditorProfile(kind="graphical", object_mode="any"),
    "CustomObstacleTwoWays": ScenarioEditorProfile(kind="graphical", object_mode="any"),
    "RoadBlocked": ScenarioEditorProfile(kind="graphical", object_mode="any"),
    "BadParkingObstacle": ScenarioEditorProfile(kind="graphical", object_mode="vehicle_only"),
    "BadParkingObstacleTwoWays": ScenarioEditorProfile(kind="graphical", object_mode="vehicle_only"),
    "PermutedConstructionObstacle": ScenarioEditorProfile(kind="graphical", object_mode="construction"),
    "PermutedConstructionObstacleTwoWays": ScenarioEditorProfile(kind="graphical", object_mode="construction"),
    # Text-only generalized scenarios
    "ImageOnObject": ScenarioEditorProfile(kind="text"),
    "ObscuredStopSign": ScenarioEditorProfile(kind="text"),
    "NormalVehicleRunningRedLight": ScenarioEditorProfile(kind="text"),
    "NormalVehicleTakingPriority": ScenarioEditorProfile(kind="text"),
    "ConstructionObstaclePedestrian": ScenarioEditorProfile(kind="text"),
    "PedestriansOnRoad": ScenarioEditorProfile(kind="text"),
    "PedestrianCrowd": ScenarioEditorProfile(kind="text"),
    # No customization
    "HardBrakeNoLights": ScenarioEditorProfile(kind="none"),
}


def get_scenario_editor_profile(scenario_type: str) -> ScenarioEditorProfile:
    if scenario_type in _EXPLICIT_PROFILES:
        return _EXPLICIT_PROFILES[scenario_type]

    fields = config.SCENARIO_TYPES.get(scenario_type, [])
    has_editable_field = any(field[1] in ("value", "bool", "interval", "choice", "transform") or "location" in field[1] for field in fields)
    if has_editable_field:
        return ScenarioEditorProfile(kind="text")
    return ScenarioEditorProfile(kind="none")

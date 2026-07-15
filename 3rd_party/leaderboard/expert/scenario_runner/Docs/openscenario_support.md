## OpenSCENARIO Support

The scenario_runner provides support for the [OpenSCENARIO](http://www.openscenario.org/) 1.0 standard.
The current implementation covers initial support for maneuver Actions, Conditions, Stories and the Storyboard.
If you would like to use evaluation criteria for a scenario to evaluate pass/fail results, these can be implemented
as StopTriggers (see below). However, not all features for these elements are yet available. If in doubt, please see the
module documentation in srunner/tools/openscenario_parser.py

An example for a supported scenario based on OpenSCENARIO is available [here](https://github.com/carla-simulator/scenario_runner/blob/master/srunner/examples/FollowLeadingVehicle.xosc)

In addition, it is recommended to take a look into the official documentation available [here](https://releases.asam.net/OpenSCENARIO/1.0.0/Model-Documentation/index.html) and [here](https://releases.asam.net/OpenSCENARIO/1.0.0/ASAM_OpenSCENARIO_BS-1-2_User-Guide_V1-0-0.html#_foreword).

### Migrating OpenSCENARIO 0.9.x to 1.0

The easiest way to convert old OpenSCENARIO samples to the official standard 1.0 is to use _xsltproc_ and the migration scheme located in the openscenario folder.
Example:

```bash
xsltproc -o newScenario.xosc migration0_9_1to1_0.xslt oldScenario.xosc
```

### Level of support

In the following the OpenSCENARIO attributes are listed with their current support status.

#### General OpenSCENARIO setup

This covers all part that are defined outside the OpenSCENARIO Storyboard

| Attribute                                         | Support | Notes                                                                                                                                                                                                                                              |
| ------------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileHeader`                                      | ✅      | Use "CARLA:" at the beginning of the description to use the CARLA coordinate system.                                                                                                                                                               |
| `CatalogLocations`<br>`ControllerCatalog`         | ✅      | While the catalog is supported, the reference/usage may not work.                                                                                                                                                                                  |
| `CatalogLocations`<br>`EnvironmentCatalog`        | ✅      |                                                                                                                                                                                                                                                    |
| `CatalogLocations`<br>`ManeuverCatalog`           | ✅      |                                                                                                                                                                                                                                                    |
| `CatalogLocations`<br>`MiscObjectCatalog`         | ✅      |                                                                                                                                                                                                                                                    |
| `CatalogLocations`<br>`PedestrianCatalog`         | ✅      |                                                                                                                                                                                                                                                    |
| `CatalogLocations`<br>`RouteCatalog`              | ✅      | While the catalog is supported, the reference/usage may not work.                                                                                                                                                                                  |
| `CatalogLocations`<br>`TrajectoryCatalog`         | ✅      | While the catalog is supported, the reference/usage may not work.                                                                                                                                                                                  |
| `CatalogLocations`<br>`VehicleCatalog`            | ✅      |                                                                                                                                                                                                                                                    |
| `ParameterDeclarations`                           | ✅      |                                                                                                                                                                                                                                                    |
| `RoadNetwork`<br>`LogicFile`                      | ✅      | The CARLA level can be used directly (e.g. LogicFile=Town01). Also any OpenDRIVE path can be provided.                                                                                                                                             |
| `RoadNetwork`<br>`SceneGraphFile`                 | ❌      | The provided information is not used.                                                                                                                                                                                                              |
| `RoadNetwork`<br>`TafficSignals`                  | ❌      | The provided information is not used.                                                                                                                                                                                                              |
| `Entities`<br>`EntitySelection`                   | ❌      | The provided information is not used.                                                                                                                                                                                                              |
| `Entities` `ScenarioObject`<br>`CatalogReference` | ✅      | The provided information is not used.                                                                                                                                                                                                              |
| `Entities` `ScenarioObject`<br>`MiscObject`       | ✅      | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. BoundingBox entries are ignored.                                                                                                    |
| `Entities` `ScenarioObject`<br>`ObjectController` | ❌      | The provided information is not used.                                                                                                                                                                                                              |
| `Entities` `ScenarioObject`<br>`Pedestrian`       | ✅      | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. BoundingBox entries are ignored.                                                                                                    |
| `Entities` `ScenarioObject`<br>`Vehicle`          | ✅      | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. The color can be set via properties ('Property name="color" value="0,0,255"'). Axles, Performance, BoundingBox entries are ignored. |

<br>

#### OpenSCENARIO Storyboard

##### OpenSCENARIO Actions

The OpenSCENARIO Actions can be used for two different purposes. First, Actions can be used to
define the initial behavior of something, e.g. a traffic participant. Therefore, Actions can be
used within the OpenSCENARIO Init. In addition, Actions are also used within the OpenSCENARIO
story. In the following, the support status for both application areas is listed. If an action
contains of submodules, which are not listed, the support status applies to all submodules.

###### GlobalAction

| GlobalAction                                                                    | Init  support | Story support | Notes                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------- | ------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EntityAction`                                                                  | ❌            | ❌            |                                                                                                                                                                                                                                                                                  |
| `EnvironmentAction`                                                             | ✅            | ❌            |                                                                                                                                                                                                                                                                                  |
| `ParameterAction`                                                               | ✅            | ✅            |                                                                                                                                                                                                                                                                                  |
| `InfrastructureAction` `TrafficSignalAction`<br>`TrafficAction`                 | ❌            | ❌            |                                                                                                                                                                                                                                                                                  |
| `InfrastructureAction` `TrafficSignalAction`<br>`TrafficSignalControllerAction` | ❌            | ❌            |                                                                                                                                                                                                                                                                                  |
| `InfrastructureAction` `TrafficSignalAction`<br>`TrafficSignalStateAction`      | ❌            | ✅            | As traffic signals in CARLA towns have no unique ID, they have to be set by providing their position (Example: TrafficSignalStateAction name="pos=x,y" state="green"). The id can also be used for none CARLA town (Example: TrafficSignalStateAction name="id=n" state="green") |

<br>

###### UserDefinedAction

| UserDefinedAction     | Init  support | Story support | Notes                                                                                                                        |
| --------------------- | ------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `CustomCommandAction` | ❌            | ✅            | This action is currently used to trigger the execution of an additional script. Example: type="python /path/to/script args". |

<br>

###### PrivateAction

| PrivateAction                                         | Init  support | Story support | Notes                                                                                                                                                                               |
| ----------------------------------------------------- | ------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ActivateControllerAction`                            | ❌            | ✅            | Can be used to activate/deactive the CARLA autopilot.                                                                                                                               |
| `ControllerAction`                                    | ✅            | ✅            | AssignControllerAction is supported, but a Python module has to be provided for the controller implementation, and in OverrideControllerValueAction all values need to be `False`.  |
| `LateralAction`<br> `LaneChangeAction`                | ❌            | ✅            | Currently all lane changes have a linear dynamicShape, the dynamicDimension is defined as the distance and are relative to the actor itself (RelativeTargetLane).                   |
| `LateralAction`<br>`LaneOffsetAction`                 | ❌            | ✅            | Currently all type of dynamicShapes are ignored and depend on the controller. This action might not work as intended if the offset is high enough to make the vehicle exit its lane |
| `LateralAction`<br> `LateralDistanceAction`           | ❌            | ❌            |                                                                                                                                                                                     |
| `LongitudinalAction`<br> `LongitudinalDistanceAction` | ❌            | ✅            | `timeGap` attribute is not supported                                                                                                                                                |
| `LongitudinalAction`<br> `SpeedAction`                | ✅            | ✅            |                                                                                                                                                                                     |
| `SynchronizeAction`                                   | ❌            | ❌            |                                                                                                                                                                                     |
| `TeleportAction`                                      | ✅            | ✅            |                                                                                                                                                                                     |
| `VisibilityAction`                                    | ❌            | ❌            |                                                                                                                                                                                     |
| `RoutingAction`<br> `AcquirePositionAction`           | ❌            | ✅            |                                                                                                                                                                                     |
| `RoutingAction`<br> `AssignRouteAction`               | ❌            | ✅            | Route Options (shortest/fastest/etc) are supported. Shortests means direct path between A and B, all other will use the shortest path along the road network between A and B        |
| `RoutingAction`<br> `FollowTrajectoryAction`          | ❌            | ❌            |                                                                                                                                                                                     |

<br>

##### OpenSCENARIO Conditions

Conditions in OpenSCENARIO can be defined either as ByEntityCondition or as ByValueCondition.
Both can be used for StartTrigger and StopTrigger conditions.
The following two tables list the support status for each.

###### ByEntityCondition

| EntityCondition             | Support | Notes                                          |
| --------------------------- | ------- | ---------------------------------------------- |
| `AccelerationCondition`     | ✅      |                                                |
| `CollisionCondition`        | ✅      |                                                |
| `DistanceCondition`         | ✅      | \*freespace\* attribute is still not supported |
| `EndOfRoadCondition`        | ✅      |                                                |
| `OffroadCondition`          | ✅      |                                                |
| `ReachPositionCondition`    | ✅      |                                                |
| `RelativeDistanceCondition` | ✅      |                                                |
| `RelativeSpeedCondition`    | ✅      |                                                |
| `SpeedCondition`            | ✅      |                                                |
| `StandStillCondition`       | ✅      |                                                |
| `TimeHeadwayCondition`      | ✅      |                                                |
| `TimeToCollisionCondition`  | ✅      |                                                |
| `TraveledDistanceCondition` | ✅      |                                                |

<br>

###### ByValueCondition

| ValueCondition                     | Support | Notes                                                                                                                                                                                                                                                                        |
| ---------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ParameterCondition`               | ✅      | The level of support depends on the parameter. It is recommended to use other conditions if possible. Please also consider the note below.                                                                                                                                   |
| `SimulationTimeCondition`          | ✅      |                                                                                                                                                                                                                                                                              |
| `StoryboardElementStateCondition`  | ✅      | startTransition, stopTransition, endTransition and completeState are currently supported.                                                                                                                                                                                    |
| `TimeOfDayCondition`               | ✅      |                                                                                                                                                                                                                                                                              |
| `TrafficSignalCondition`           | ✅      | As traffic signals in CARLA towns have no unique ID, they have to be set by providing their position (Example: TrafficSignalCondition name="pos=x,y" state="green"). The id can also be used for none CARLA town (Example: TrafficSignalCondition name="id=n" state="green") |
| `TrafficSignalControllerCondition` | ❌      |                                                                                                                                                                                                                                                                              |
| `UserDefinedValueCondition`        | ❌      |                                                                                                                                                                                                                                                                              |

<br>

!!! Note
In the OpenSCENARIO 1.0 standard, a definition of test / evaluation criteria is not
defined. For this purpose, you can re-use StopTrigger conditions with CARLA. The following
StopTrigger conditions for evaluation criteria are supported through ParameterConditions by
providing the criteria name for the condition:

```
 * criteria_RunningStopTest
 * criteria_RunningRedLightTest
 * criteria_WrongLaneTest
 * criteria_OnSideWalkTest
 * criteria_KeepLaneTest
 * criteria_CollisionTest
 * criteria_DrivenDistanceTest
```

##### OpenSCENARIO Positions

There are several ways of defining positions in OpenSCENARIO. In the following we list the
current support status for each definition format.

| Position                 | Support | Notes |
| ------------------------ | ------- | ----- |
| `LanePosition`           | ✅      |       |
| `RelativeLanePosition`   | ✅      |       |
| `RelativeObjectPosition` | ✅      |       |
| `RelativeRoadPosition`   | ✅      |       |
| `RelativeWorldPosition`  | ✅      |       |
| `RoadPosition`           | ✅      |       |
| `RoutePosition`          | ❌      |       |
| `WorldPosition`          | ✅      |       |

<br>

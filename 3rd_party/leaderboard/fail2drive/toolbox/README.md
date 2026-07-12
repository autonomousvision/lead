# Fail2Drive Toolbox

The Fail2Drive Toolbox includes a graphical tool for creating and editing routes and scenarios for the CARLA driving simulator. It provides an interactive map view where you can define driving paths and attach challenging scenarios to them, producing route files that can be used for autonomous driving evaluation.

> When adding Fail2Drive Generalization Scenarios to the route, the Toolbox will automatically generate the corresponding in-distribution scenarios

## Starting a Session

When you open the tool, you can either:

- **New Session** — select a CARLA town (Town01–Town07, Town10–13, Town15) and start from scratch
- **Load Route File** — open an existing single XML route file to continue editing
- **Load Route Folder** — load all route files from a folder at once

After loading, the map renders in the main canvas with roads, parking areas, bike lanes, traffic lights, and stop signs all visible for reference.

---

## Controls

| Action | Result |
|---|---|
| Scroll wheel | Zoom in/out |
| Middle mouse drag | Pan |
| Left click on road | Add waypoint |
| Right click on route | Add scenario |
| Left drag waypoint/scenario | Move waypoint/scenario |
| Ctrl + Left click waypoint/scenario | Remove waypoint/scenario |
| Left click scenario marker | Open scenario parameter editor |

The road layout shows road waypoints in black, parking areas in blue, and bike lanes in red. As you move the cursor, a green preview trace shows the planned path from your last waypoint to the cursor position.

---

## Managing Routes

You can work on multiple routes at the same time. Use the sidebar buttons to:

- **Add Route** — create a new empty route
- **Remove Route** — delete the currently selected route
- **Show other routes** (checkbox) — overlay all other routes on the canvas in light green for spatial reference

All routes appear in the tree panel on the left. Click a route entry to select it and start editing it.

---

## Building a Route

With a route selected, click anywhere on the map to add waypoints. The tool automatically plans a path between consecutive waypoints using CARLA's road network — you only place sparse anchor points and the dense trajectory is filled in automatically. The green preview trace that appears as you hover shows what the next segment will look like before you click.

---

## Adding Scenarios

Scenarios trigger when the ego vehicle passes the specified point on the route. Right click near a point on the route to open the scenario picker, select a type, fill in the parameters, and the scenario marker appears on the map.

Scenarios also appear as children under their route in the left-side tree. Clicking a scenario in the tree selects and opens it for editing too.

---

## Scenario Types

The tool includes scenarios from the base CARLA leaderboard benchmark as well as scenarios introduced by Fail2Drive.

**Fail2Drive Scenarios** — `BadParkingObstacle`, `BadParkingObstacleTwoWays`, `ConstructionObstacleOppositeLane`, `ConstructionObstaclePedestrian`, `ConstructionObstacleRightLane`, `CustomObstacle`, `CustomObstacleTwoWays`, `HardBrakeNoLights`, `ImageOnObject`, `NormalVehicleRunningRedLight`, `NormalVehicleTakingPriority`, `ObscuredStopSign`, `PedestrianCrowd`, `PedestriansOnRoad`, `PermutedConstructionObstacle`, `PermutedConstructionObstacleTwoWays`, `RoadBlocked`

**Junctions** — `SignalizedJunctionLeftTurn`, `SignalizedJunctionRightTurn`, `NonSignalizedJunctionLeftTurn`, `NonSignalizedJunctionRightTurn`, `OppositeVehicleRunningRedLight`, `OppositeVehicleTakingPriority`, `BlockedIntersection`, `PriorityAtJunction`

**Crossing Actors** — `DynamicObjectCrossing`, `ParkingCrossingPedestrian`, `PedestrianCrossing`, `VehicleTurningRoute`, `VehicleTurningRoutePedestrian`, `CrossingBicycleFlow`

**Actor Flows & Merging** — `EnterActorFlow`, `EnterActorFlowV2`, `InterurbanActorFlow`, `InterurbanAdvancedActorFlow`, `HighwayExit`, `MergerIntoSlowTraffic`, `MergerIntoSlowTrafficV2`, `HighwayCutIn`, `ParkingCutIn`, `StaticCutIn`

**Route Obstacles** — `ConstructionObstacle`, `ConstructionObstacleTwoWays`, `Accident`, `AccidentTwoWays`, `ParkedObstacle`, `ParkedObstacleTwoWays`, `VehicleOpensDoorTwoWays`, `HazardAtSideLane`, `HazardAtSideLaneTwoWays`, `InvadingTurn`

**Other** — `ControlLoss`, `HardBreakRoute`, `ParkingExit`, `YieldToEmergencyVehicle`, `BackgroundActivityParametrizer`

Each scenario type has its own set of parameters (speeds, distances, directions, actor models, etc.) that you configure through the parameter dialog when placing or editing the scenario.

---

## Scenarios with Graphical Layout Editors

Several obstacle scenarios have a dedicated visual layout editor in addition to the standard parameter form:

- **PermutedConstructionObstacle / PermutedConstructionObstacleTwoWays** — replace traffic cones, debris, and warning signs
- **BadParkingObstacle / BadParkingObstacleTwoWays** — position and orient a badly-parked vehicle relative to the lane
- **RoadBlocked** — arrange blocking objects across the road
- **CustomObstacle / CustomObstacleTwoWays** — fully custom object placement (see Custom Obstacle Designer below)

In the parameter dialog for these types, a **Customize Layout** button opens the graphical editor. After saving in the editor, you return to the parameter dialog to confirm.

---

## Custom Obstacle Designer

The Custom Obstacle Designer provides a canvas for building complex obstacle layouts from individual CARLA props and vehicles. It can be opened from within the Route Builder when configuring a `CustomObstacle` or `CustomObstacleTwoWays` scenario, or run independently.

**What you can do:**

- Add objects from a searchable list of CARLA blueprints (props, vehicles)
- Drag objects freely on the canvas to position them
- Rotate objects using CTRL + scroll
- Select multiple objects and move/rotate them as a group
- Visualize the in-distribution scenario reference (Construction, Accident, ParkedVehicle)

---

## Saving Routes

- **Save Route** — overwrite the current route's source file (disabled if not loaded from a file)
- **Save Route As** — save the current route to a new file
- **Save All Routes To Folder** — write every loaded route to a chosen folder, one file per route

Unsaved changes are indicated by an asterisk (`*`) next to the route name in the tree. The tool also auto-saves all modified routes every minute to a recovery folder.

---

## Tips

- When working with many routes, use **Load Route Folder** to load them all at once, then use **Show other routes** to see how new routes relate spatially to existing ones.
- Waypoints only snap to valid road waypoints — the cursor snaps to the nearest drivable point automatically.
- For actor-flow scenarios that require a start and end location (like `EnterActorFlow`), some attributes can be picked directly on the map — a **Pick on map** button appears in the parameter dialog for those fields.
- If the tool crashes unexpectedly, auto-saved files are available for recovery.

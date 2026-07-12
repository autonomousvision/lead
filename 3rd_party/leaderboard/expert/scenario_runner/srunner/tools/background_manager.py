#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Several atomic behaviors to help with the communication with the background activity,
removing its interference with other scenarios
"""
import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import \
    AtomicBehavior
from srunner.scenariomanager.timer import GameTime


class MakeRedLightLonger(AtomicBehavior):
    def __init__(self, world: carla.World, name="MakeRedLightLonger"):
        self.world = world
        self.original_durations = {}  # Store original red light durations
        self.modified_lights = set()  # Track which lights we've modified
        self.extension_time = 5.0  # Additional seconds to add to red lights
        super().__init__(name)

    def update(self):
        """Extends the duration of all red traffic lights by 5 seconds"""
        try:
            # Get all traffic lights in the world
            traffic_lights = self.world.get_actors().filter("traffic.traffic_light")

            for light in traffic_lights:
                # Only modify lights that are currently red and haven't been modified yet
                if light.get_state() == carla.TrafficLightState.Red and light.id not in self.modified_lights:
                    # Store original duration if not already stored
                    if light.id not in self.original_durations:
                        self.original_durations[light.id] = light.get_red_time()

                    # Extend the red light duration
                    new_duration = self.original_durations[light.id] + self.extension_time
                    light.set_red_time(new_duration)

                    # Mark this light as modified
                    self.modified_lights.add(light.id)

                    print(f"Extended red light {light.id} duration from {self.original_durations[light.id]:.1f}s to {new_duration:.1f}s")

                # Reset tracking when light cycles back to green
                elif light.get_state() != carla.TrafficLightState.Red and light.id in self.modified_lights:
                    self.modified_lights.remove(light.id)

        except Exception as e:
            print(f"Error in MakeRedLightLonger: {e}")

        return py_trees.common.Status.SUCCESS

    def __del__(self):
        """Restore original red light durations when the behavior is destroyed"""
        try:
            traffic_lights = self.world.get_actors().filter("traffic.traffic_light")

            for light in traffic_lights:
                if light.id in self.original_durations:
                    # Restore original duration
                    light.set_red_time(self.original_durations[light.id])
                    print(f"Restored red light {light.id} to original duration: {self.original_durations[light.id]:.1f}s")

        except Exception as e:
            print(f"Error restoring red light durations: {e}")

class MakeTrafficLightRedOnce(AtomicBehavior):
    def __init__(self, world: carla.World, ego_vehicle: carla.Actor, proximity_distance=48.0, name="MakeTrafficLightRedOnce"):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.proximity_distance = proximity_distance  # Distance in meters to trigger the behavior
        self.triggered_lights = set()  # Track which lights have been triggered to avoid repeating
        self.current_traffic_light_elapsed_time = 0.0
        self.current_traffic_light_id = None
        super().__init__(name)

    def update(self):
        """Makes the nearest traffic light red once when ego vehicle is nearby"""
        try:
            # Get ego vehicle location
            ego_location = self.ego_vehicle.get_location()

            # Find the nearest traffic light
            traffic_light = CarlaDataProvider._global_memory["next_traffic_light"]

            # Check if ego is close enough to the nearest light and we haven't triggered it yet
            if traffic_light is not None:
                # Calculate distance to the traffic light
                light_location = traffic_light.get_location()
                min_distance = ego_location.distance(light_location)

                if (min_distance <= self.proximity_distance and
                    traffic_light.id not in self.triggered_lights):

                    # Before setting to red:
                    original_red_time = traffic_light.get_red_time()
                    original_green_time = traffic_light.get_green_time()
                    original_yellow_time = traffic_light.get_yellow_time()

                    # Set to red
                    traffic_light.set_state(carla.TrafficLightState.Red)

                    # Later, restore normal cycling:
                    traffic_light.set_red_time(original_red_time)
                    traffic_light.set_green_time(original_green_time)
                    traffic_light.set_yellow_time(original_yellow_time)

                    print(f"Set nearest traffic light {traffic_light.id} to RED (distance: {min_distance:.1f}m). Is traffic light frozen?: {traffic_light.is_frozen()}")

                    # Mark this light as triggered so we don't repeat the action
                    self.triggered_lights.add(traffic_light.id)
                    self.current_traffic_light_elapsed_time = traffic_light.get_elapsed_time()
                    self.current_traffic_light_id = traffic_light.id

                if traffic_light.id == self.current_traffic_light_id:
                    if traffic_light.get_elapsed_time() < self.current_traffic_light_elapsed_time:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        self.current_traffic_light_id = None
                    else:
                        self.current_traffic_light_elapsed_time = traffic_light.get_elapsed_time()

        except Exception as e:
            print(f"Error in MakeTrafficLightRedOnce: {e}")

        return py_trees.common.Status.RUNNING

class KeepFrontClear(AtomicBehavior):
    def __init__(self, ego_vehicles: list[carla.Vehicle], world: carla.World, name="KeepFrontClear"):
        self.ego_vehicles = ego_vehicles  # List[carla.Vehicle]
        self.world = world
        self.map = world.get_map()
        super().__init__(name)

    def update(self):
        """Removes vehicles ahead in the same lane as ego"""
        vehicles = self.world.get_actors().filter("vehicle.*")

        for ego in self.ego_vehicles:
            ego_wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)

            for v in vehicles:
                if v.get_velocity().length() > 3.0:
                    continue # No need to remove moving vehicles
                if v.id == ego.id:
                    continue
                v_wp = self.map.get_waypoint(v.get_location(), project_to_road=True)

                # check same lane & road
                if v_wp.road_id == ego_wp.road_id and v_wp.lane_id == ego_wp.lane_id:
                    # longitudinal distance along lane
                    dist = ego.get_location().distance(v.get_location())
                    if dist < 24.0:
                        # ensure it's actually in front
                        forward = ego.get_transform().get_forward_vector()
                        rel = v.get_location() - ego.get_location()
                        if rel.x * forward.x + rel.y * forward.y + rel.z * forward.z > 0:
                            print(f"Destroying vehicle {v.id} ahead of ego {ego.id}")
                            v.destroy()

        return py_trees.common.Status.RUNNING

class DisallowActorGeneration(AtomicBehavior):
    def __init__(self):
        super().__init__("clear_scenario_from_carla_data_provider")

    def update(self):
        if CarlaDataProvider.current_active_scenario_type() in [
            "SignalizedJunctionRightTurn",
            "NonSignalizedJunctionRightTurn"
        ]:
            CarlaDataProvider._global_memory["allow_new_actors"] = False
        return py_trees.common.Status.SUCCESS

class AllowActorGeneration(AtomicBehavior):
    def __init__(self):
        super().__init__("clear_scenario_from_carla_data_provider")

    def update(self):
        print("[AllowActorGeneration] Allowing new actors to be spawned again")
        CarlaDataProvider._global_memory["allow_new_actors"] = True
        return py_trees.common.Status.SUCCESS


class ClearScenarioType(AtomicBehavior):
    def __init__(self, scenario_instance_id: int):
        self.scenario_instance_id = scenario_instance_id
        super().__init__("clear_scenario_from_carla_data_provider")

    def update(self):
        # Find and remove the scenario by ID (it may not be at position 0 due to distance-based sorting)
        for scenario in CarlaDataProvider.active_scenarios:
            if scenario.scenario_id == self.scenario_instance_id:
                print("[ClearScenarioType] Removing scenario with ID {} automatically after ending.".format(self.scenario_instance_id))
                CarlaDataProvider.remove_scenario(scenario)
                break
        return py_trees.common.Status.SUCCESS

class ChangeRoadBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the road behavior.
    None values imply that these values won't be changed.

    Args:
        num_front_vehicles (int): Amount of vehicles in front of the ego. Can't be negative
        num_back_vehicles (int): Amount of vehicles behind it. Can't be negative
        switch_source (bool): (De)activatea the road sources.
    """

    def __init__(self, num_front_vehicles=None, num_back_vehicles=None, spawn_dist=None, extra_space=None, name="ChangeRoadBehavior"):
        self._num_front = num_front_vehicles
        self._num_back = num_back_vehicles
        self._spawn_dist = spawn_dist
        self._extra_space = extra_space
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeRoadBehavior", [self._num_front, self._num_back, self._spawn_dist, self._extra_space], overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ChangeOppositeBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the opposite road behavior.
    None values imply that these values won't be changed

    Args:
        source_dist (float): Distance between the opposite sources and the ego vehicle. Must be positive
        max_actors (int): Max amount of concurrent alive actors spawned by the same source. Can't be negative
    """

    def __init__(self, source_dist=None, spawn_dist=None, active=None, name="ChangeOppositeBehavior"):
        self._source_dist = source_dist
        self._spawn_dist = spawn_dist
        self._active = active
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeOppositeBehavior", [self._source_dist, self._spawn_dist, self._active], overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ChangeJunctionBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the junction behavior.
    None values imply that these values won't be changed

    Args:
        source_dist (float): Distance between the junctiob sources and the junction entry. Must be positive
        max_actors (int): Max amount of concurrent alive actors spawned by the same source. Can't be negative
    """

    def __init__(self, source_dist=None, spawn_dist=None, max_actors=None, source_perc=None, name="ChangeJunctionBehavior"):
        self._source_dist = source_dist
        self._spawn_dist = spawn_dist
        self._max_actors = max_actors
        self._perc = source_perc
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeJunctionBehavior", [self._source_dist, self._spawn_dist, self._max_actors, self._perc], overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class SetMaxSpeed(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that its behavior is restriced to a maximum speed
    """

    def __init__(self, max_speed, name="SetMaxSpeed"):
        self._max_speed = max_speed
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_SetMaxSpeed", self._max_speed, overwrite=True)
        return py_trees.common.Status.SUCCESS


class StopFrontVehicles(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a HardBreak scenario has to be triggered.
    'stop_duration' is the amount of time, in seconds, the vehicles will be stopped
    """

    def __init__(self, name="StopFrontVehicles"):
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_StopFrontVehicles", True, overwrite=True)
        return py_trees.common.Status.SUCCESS


class StartFrontVehicles(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a HardBreak scenario has to be triggered.
    'stop_duration' is the amount of time, in seconds, the vehicles will be stopped
    """

    def __init__(self, name="StartFrontVehicles"):
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_StartFrontVehicles", True, overwrite=True)
        return py_trees.common.Status.SUCCESS


class StopBackVehicles(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to stop the vehicles behind the ego as to
    not interfere with the scenarios. This only works at roads, not junctions.
    """
    def __init__(self, name="StopBackVehicles"):
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_StopBackVehicles", True, overwrite=True)
        return py_trees.common.Status.SUCCESS


class StartBackVehicles(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to restart the vehicles behind the ego.
    """
    def __init__(self, name="StartBackVehicles"):
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_StartBackVehicles", True, overwrite=True)
        return py_trees.common.Status.SUCCESS


class LeaveSpaceInFront(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the ego needs more space in front.
    This only works at roads, not junctions.
    """
    def __init__(self, space, name="LeaveSpaceInFront"):
        self._space = space
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", [self._space], overwrite=True)
        return py_trees.common.Status.SUCCESS


class SwitchRouteSources(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to (de)activate all route sources
    """
    def __init__(self, enabled=True, name="SwitchRouteSources"):
        self._enabled = enabled
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_SwitchRouteSources", self._enabled, overwrite=True)
        return py_trees.common.Status.SUCCESS


class RemoveRoadLane(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to remove its actors from the given lane
    and stop generating new ones on this lane, or recover from stopping.

    Args:
        lane_wp (carla.Waypoint): A carla.Waypoint
        active (bool)
    """
    def __init__(self, lane_wp, name="RemoveRoadLane"):
        self._lane_wp = lane_wp
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_RemoveRoadLane", self._lane_wp, overwrite=True)
        return py_trees.common.Status.SUCCESS


class ReAddRoadLane(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to readd the ego road lane.

    Args:
        offset: 0 to readd the ego lane, 1 for the right side lane, -1 for the left...
        active (bool)
    """
    def __init__(self, offset, name="BA_ReAddRoadLane"):
        self._offset = offset
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_ReAddRoadLane", self._offset, overwrite=True)
        return py_trees.common.Status.SUCCESS


class LeaveSpaceInFront(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the ego needs more space in front.
    This only works at roads, not junctions.
    """
    def __init__(self, space, name="LeaveSpaceInFront"):
        self._space = space
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", self._space, overwrite=True)
        return py_trees.common.Status.SUCCESS


class LeaveCrossingSpace(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the ego needs more space in front.
    This only works at roads, not junctions.
    """
    def __init__(self, collision_wp, name="LeaveCrossingSpace"):
        self._collision_wp = collision_wp
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_LeaveCrossingSpace", self._collision_wp, overwrite=True)
        return py_trees.common.Status.SUCCESS

class HandleJunctionScenario(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to adapt to a junction scenario

    Args:
        clear_junction (bool): Remove all actors inside the junction, and all that enter it afterwards
        clear_ego_entry (bool): Remove all actors part of the ego road to ensure a smooth entry of the ego to the junction.
        remove_entries (list): list of waypoint representing a junction entry that needs to be removed
        remove_exits (list): list of waypoint representing a junction exit that needs to be removed
        stop_entries (bool): Stops all the junction entries
        extend_road_exit (float): Moves the road junction actors forward to leave more space for the scenario.
            It also deactivates the road sources.
        active (bool)
    """
    def __init__(self, clear_junction=True, clear_ego_entry=True, remove_entries=[],
                 remove_exits=[], stop_entries=True, extend_road_exit=0,
                 name="HandleJunctionScenario"):
        self._clear_junction = clear_junction
        self._clear_ego_entry = clear_ego_entry
        self._remove_entries = remove_entries
        self._remove_exits = remove_exits
        self._stop_entries = stop_entries
        self._extend_road_exit = extend_road_exit
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set(
            "BA_HandleJunctionScenario",
            [self._clear_junction, self._clear_ego_entry, self._remove_entries,
             self._remove_exits, self._stop_entries, self._extend_road_exit],
            overwrite=True)
        return py_trees.common.Status.SUCCESS

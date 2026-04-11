"""
This file defines a Route class that represents a route in the CARLA simulator. It handles the creation,
modification, and management of waypoints, scenarios, and other route-related data. The class provides
methods for adding or removing waypoints, adding or removing scenarios, interpolating dense waypoints
between route waypoints, and generating scenario XML elements.
"""

import carla
import numpy as np
from lxml import etree


class Route:
    def __init__(
        self,
        carla_client,
        route_id,
        map_name,
        weather_element,
        waypoints,
        scenarios,
        scenario_types,
        scenario_trigger_points,
        max_distance_when_removing=10,
    ):
        """
        Initialize a Route object.

        Args:
            carla_client (CarlaClient): The CARLA client instance.
            route_id (int): The ID of the route.
            map_name (str): The name of the map for this route.
            weather_element (lxml.etree.Element): The XML element representing the weather conditions.
            waypoints (list): A list of waypoints, each represented as [x, y, z].
            scenarios (list): A list of lxml.etree.Element objects representing scenarios.
            scenario_types (list): A list of strings representing the types of scenarios.
            scenario_trigger_points (list): A list of trigger points for scenarios, each represented as [x, y, z].
            max_distance_when_removing (float): The maximum distance threshold for removing waypoints or scenarios (default=10).
        """
        self.carla_client = carla_client
        self.route_id = route_id
        self.map_name = map_name
        self.weather_element = weather_element
        self.waypoints = waypoints
        self.scenarios = scenarios
        self.scenario_types = scenario_types
        self.scenario_trigger_points = scenario_trigger_points
        self.max_distance_when_removing = max_distance_when_removing

        self.route_length = 0  # in meters
        self.dense_waypoints = []  # [[x, y, z], ...]: list
        self.update_dense_route()

    def generate_scenario_elem(self, loc, scenario_type, scenario_attributes):
        """
        Generate an XML element representing a scenario.

        Args:
            loc (list): The location of the scenario trigger point as [x, y, z].
            scenario_type (str): The type of the scenario.
            scenario_attributes (list): A list of tuples (attribute_name, attribute_type, attribute_value).

        Returns:
            lxml.etree.Element: The generated scenario XML element.
        """
        scenario_elem = etree.Element("scenario")
        scenario_idx = self.scenario_types.count(scenario_type)

        scenario_elem.set("name", f"{scenario_type}_{scenario_idx}")
        scenario_elem.set("type", scenario_type)

        for attr_name, attr_type, attr_value in scenario_attributes:
            data = etree.SubElement(scenario_elem, attr_name)
            if attr_type == "transform":
                data.set("x", str(attr_value[0]))
                data.set("y", str(attr_value[1]))
                data.set("z", str(attr_value[2]))
                data.set("yaw", str(attr_value[3]))
            elif "location" in attr_type:
                data.set("x", str(attr_value[0]))
                data.set("y", str(attr_value[1]))
                data.set("z", str(attr_value[2]))
                if "probability" in attr_type:
                    data.set("p", str(attr_value[3]))
            elif attr_type in ("value", "choice", "bool"):
                data.set("value", str(attr_value))
            elif attr_type == "interval":
                data.set("from", str(attr_value[0]))
                data.set("to", str(attr_value[1]))
            elif attr_type == "objects":
                if isinstance(attr_value, dict):
                    for key, value in sorted(attr_value.items()):
                        data.set(str(key), str(value))

        return scenario_elem

    def _get_snapped_driving_or_parking_wp_loc(self, loc):
        carla_loc = carla.Location(loc[0], loc[1])
        wp = self.carla_client.carla_map.get_waypoint(
            carla_loc, lane_type=carla.LaneType.Driving | carla.LaneType.Parking
        )
        return wp, [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]

    def add_scenario(self, loc, scenario_type, scenario_attributes):
        """
        Add a scenario to the route.

        Args:
            loc (list): The location of the scenario trigger point as [x, y, z].
            scenario_type (str): The type of the scenario.
            scenario_attributes (list): A list of tuples (attribute_name, attribute_type, attribute_value).

        If there is an existing scenario closer than the `max_distance_when_removing` threshold, it is removed.
        """
        wp, wp_loc = self._get_snapped_driving_or_parking_wp_loc(loc)

        scenario_attributes.append(
            [
                "trigger_point",
                "transform",
                [round(wp_loc[0], 1), round(wp_loc[1], 1), round(wp_loc[2], 1), round(wp.transform.rotation.yaw, 1)],
            ]
        )
        scenario_elem = self.generate_scenario_elem(wp_loc, scenario_type, scenario_attributes)
        self.scenarios.append(scenario_elem)
        self.scenario_trigger_points.append(wp_loc)
        self.scenario_types.append(scenario_type)

    def get_scenario_index_near(self, loc):
        if not self.scenario_trigger_points:
            return None

        _, wp_loc = self._get_snapped_driving_or_parking_wp_loc(loc)
        diff = np.linalg.norm(np.array(self.scenario_trigger_points) - np.array(wp_loc)[None, :], axis=1)
        min_idx = int(diff.argmin())
        if diff[min_idx] < self.max_distance_when_removing:
            return min_idx
        return None

    def remove_scenario_by_index(self, idx):
        if idx is None or idx < 0 or idx >= len(self.scenarios):
            return
        self.scenarios.pop(idx)
        self.scenario_trigger_points.pop(idx)
        self.scenario_types.pop(idx)

    def move_scenario_by_index(self, idx, loc):
        if idx is None or idx < 0 or idx >= len(self.scenarios):
            return

        wp, wp_loc = self._get_snapped_driving_or_parking_wp_loc(loc)
        wp_loc_rounded = [round(wp_loc[0], 1), round(wp_loc[1], 1), round(wp_loc[2], 1)]
        yaw_rounded = round(wp.transform.rotation.yaw, 1)

        scenario_elem = self.scenarios[idx]
        trigger_point_elem = scenario_elem.find("trigger_point")
        if trigger_point_elem is None:
            trigger_point_elem = etree.SubElement(scenario_elem, "trigger_point")

        trigger_point_elem.set("x", str(wp_loc_rounded[0]))
        trigger_point_elem.set("y", str(wp_loc_rounded[1]))
        trigger_point_elem.set("z", str(wp_loc_rounded[2]))
        trigger_point_elem.set("yaw", str(yaw_rounded))
        self.scenario_trigger_points[idx] = wp_loc_rounded

    def update_scenario_by_index(self, idx, scenario_attributes):
        if idx is None or idx < 0 or idx >= len(self.scenarios):
            return

        scenario_type = self.scenario_types[idx]
        trigger_loc = self.scenario_trigger_points[idx]

        # Keep existing location/transform attributes unless user explicitly redefines them.
        existing_elem = self.scenarios[idx]
        existing_by_tag = {elem.tag: elem for elem in existing_elem if elem.tag != "trigger_point"}
        existing_attribute_types = {}
        for attr_def in scenario_attributes:
            existing_attribute_types[attr_def[0]] = attr_def[1]

        merged_attributes = [list(x) for x in scenario_attributes]
        for child in existing_elem:
            if child.tag == "trigger_point":
                continue
            if (
                scenario_type in ("PermutedConstructionObstacle", "PermutedConstructionObstacleTwoWays", "RoadBlocked")
                and child.tag in ("objects", "baseline_overlay", "overlay_direction")
            ):
                continue
            if child.tag in existing_attribute_types:
                continue
            if "yaw" in child.attrib:
                merged_attributes.append(
                    [
                        child.tag,
                        "transform",
                        [
                            float(child.get("x")),
                            float(child.get("y")),
                            float(child.get("z")),
                            float(child.get("yaw")),
                        ],
                    ]
                )
            elif "x" in child.attrib and "y" in child.attrib and "z" in child.attrib:
                attr_value = [float(child.get("x")), float(child.get("y")), float(child.get("z"))]
                if "p" in child.attrib:
                    attr_value.append(float(child.get("p")))
                merged_attributes.append([child.tag, "location", attr_value])
            elif child.tag == "objects":
                merged_attributes.append([child.tag, "objects", dict(child.attrib)])
            else:
                # Preserve untyped value-only tags to avoid silently dropping scenario-specific data.
                if "value" in child.attrib:
                    merged_attributes.append([child.tag, "value", child.get("value")])

        wp, wp_loc = self._get_snapped_driving_or_parking_wp_loc(trigger_loc)
        merged_attributes.append(
            [
                "trigger_point",
                "transform",
                [round(wp_loc[0], 1), round(wp_loc[1], 1), round(wp_loc[2], 1), round(wp.transform.rotation.yaw, 1)],
            ]
        )

        scenario_elem = self.generate_scenario_elem(wp_loc, scenario_type, merged_attributes)
        scenario_elem.set("name", existing_elem.get("name", scenario_elem.get("name")))
        scenario_elem.set("type", scenario_type)
        self.scenarios[idx] = scenario_elem
        self.scenario_trigger_points[idx] = wp_loc

    def remove_scenario(self, loc):
        """
        Remove a scenario from the route based on the provided location.

        Args:
            loc (list): The location near the scenario trigger point as [x, y, z].

        The scenario closest to the provided location is removed.
        """
        idx = self.get_scenario_index_near(loc)
        self.remove_scenario_by_index(idx)

    def should_remove_scenario(self, loc):
        """
        Check if a scenario should be removed based on the provided location.

        Args:
            loc (list): The location near the scenario trigger point as [x, y, z].

        Returns:
            bool: True if a scenario should be removed, False otherwise.
        """
        return self.get_scenario_index_near(loc) is not None

    def check_if_scenario_can_be_added(self, loc):
        """
        Check if a scenario can be added at the provided location.

        Args:
            loc (list): The location for the potential scenario trigger point as [x, y, z].

        Returns:
            bool: True if a scenario can be added, False otherwise.
        """
        if not self.dense_waypoints:
            return False

        carla_loc = carla.Location(loc[0], loc[1])
        wp = self.carla_client.carla_map.get_waypoint(
            carla_loc, lane_type=carla.LaneType.Driving | carla.LaneType.Parking
        )
        wp_loc = [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]

        diff = np.linalg.norm(np.array(self.dense_waypoints) - np.array(wp_loc)[None, :], axis=1)
        return diff.min() < self.max_distance_when_removing

    def add_or_remove_waypoint(self, loc):
        """
        Add or remove a waypoint from the route based on the provided location.

        Args:
            loc (list): The location for the potential waypoint as [x, y, z].

        If there is an existing waypoint closer than the `max_distance_when_removing` threshold, it is removed.
        Otherwise, a new waypoint is added.
        """
        add_point = True

        # Only the first waypoint can be on a parking lot in case the scenario starts with ParkingExit
        lane_type = (
            carla.LaneType.Driving | carla.LaneType.Parking if len(self.waypoints) == 0 else carla.LaneType.Driving
        )

        wp = self.carla_client.carla_map.get_waypoint(carla.Location(loc[0], loc[1]), lane_type=lane_type)
        wp_loc = [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]

        if self.waypoints:
            diff = np.linalg.norm(np.array(self.waypoints) - np.array(wp_loc)[None, :], axis=1)
            min_idx = diff.argmin()
            if diff[min_idx] < self.max_distance_when_removing:
                add_point = False

        if add_point:
            wp_loc = [round(wp_loc[0], 1), round(wp_loc[1], 1), round(wp_loc[2], 1)]
            self.waypoints.append(wp_loc)
        else:
            self.waypoints.pop(min_idx)

        self.update_dense_route()

    def add_waypoint(self, loc):
        lane_type = (
            carla.LaneType.Driving | carla.LaneType.Parking if len(self.waypoints) == 0 else carla.LaneType.Driving
        )
        wp = self.carla_client.carla_map.get_waypoint(carla.Location(loc[0], loc[1]), lane_type=lane_type)
        wp_loc = [round(wp.transform.location.x, 1), round(wp.transform.location.y, 1), round(wp.transform.location.z, 1)]
        self.waypoints.append(wp_loc)
        self.update_dense_route()

    def get_waypoint_index_near(self, loc):
        if not self.waypoints:
            return None
        wp = self.carla_client.carla_map.get_waypoint(carla.Location(loc[0], loc[1]), lane_type=carla.LaneType.Driving)
        wp_loc = [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
        diff = np.linalg.norm(np.array(self.waypoints) - np.array(wp_loc)[None, :], axis=1)
        idx = int(diff.argmin())
        if diff[idx] < self.max_distance_when_removing:
            return idx
        return None

    def remove_waypoint_by_index(self, idx):
        if idx is None or idx < 0 or idx >= len(self.waypoints):
            return
        self.waypoints.pop(idx)
        self.update_dense_route()

    def move_waypoint_by_index(self, idx, loc):
        if idx is None or idx < 0 or idx >= len(self.waypoints):
            return
        lane_type = (
            carla.LaneType.Driving | carla.LaneType.Parking if idx == 0 else carla.LaneType.Driving
        )
        wp = self.carla_client.carla_map.get_waypoint(carla.Location(loc[0], loc[1]), lane_type=lane_type)
        wp_loc = [round(wp.transform.location.x, 1), round(wp.transform.location.y, 1), round(wp.transform.location.z, 1)]
        self.waypoints[idx] = wp_loc
        self.update_dense_route()

    def update_dense_route(self):
        """
        Update the dense waypoints list by interpolating between the route waypoints.
        Also updates the route length.
        """
        self.dense_waypoints.clear()
        self.route_length = 0

        if self.waypoints:
            self.dense_waypoints.append(self.waypoints[0])

        for i in range(len(self.waypoints) - 1):
            from_loc, to_loc = self.waypoints[i], self.waypoints[i + 1]
            from_loc = carla.Location(from_loc[0], from_loc[1], from_loc[2])
            to_loc = carla.Location(to_loc[0], to_loc[1], to_loc[2])

            self.dense_waypoints += self.interpolate_trace(from_loc, to_loc)

        if len(self.dense_waypoints) > 1:
            self.route_length = np.linalg.norm(np.diff(np.array(self.dense_waypoints), axis=0), axis=1).sum()

    def interpolate_trace(self, from_loc, to_loc):
        """
        Interpolate dense waypoints between two route waypoints using the global route planner.

        Args:
            from_loc (carla.Location): The starting location for the interpolation.
            to_loc (carla.Location): The ending location for the interpolation.

        Returns:
            list: A list of interpolated waypoints, each represented as [x, y, z].
        """
        from_wp = self.carla_client.carla_map.get_waypoint(from_loc)
        from_loc = from_wp.transform.location

        to_wp = self.carla_client.carla_map.get_waypoint(to_loc)
        to_loc = to_wp.transform.location

        interpolated_trace = self.carla_client.global_route_planner.trace_route(from_loc, to_loc)
        interpolated_trace = [x[0].transform.location for x in interpolated_trace]
        interpolated_trace = [[x.x, x.y, x.z] for x in interpolated_trace]

        return interpolated_trace

    def interpolate_from_last_wp(self, to_loc):
        """
        Interpolate dense waypoints between the last route waypoint and the provided location.

        Args:
            to_loc (carla.Location): The ending location for the interpolation.

        Returns:
            list: A list of interpolated waypoints, each represented as [x, y, z].
        """
        if self.waypoints:
            from_loc = carla.Location(self.waypoints[-1][0], self.waypoints[-1][1], self.waypoints[-1][2])
            return self.interpolate_trace(from_loc, to_loc)

        return []

    def add_location_transform_attributes_to_last_scenario(self, location_transform_attributes):
        """
        Add location and transform attributes to the last scenario element.

        Args:
            location_transform_attributes (list): A list of tuples (attribute_name, attribute_type, attribute_value).

        Raises:
            NotImplementedError: If an unsupported attribute type is encountered.
        """
        scenario_elem = self.scenarios[-1]
        for attr_name, attr_type, attr_value in location_transform_attributes:
            data = etree.SubElement(scenario_elem, attr_name)
            if attr_type == "transform":
                data.set("x", str(attr_value[0]))
                data.set("y", str(attr_value[1]))
                data.set("z", str(attr_value[2]))
                data.set("yaw", str(attr_value[3]))
            elif "location" in attr_type:
                data.set("x", str(attr_value[0]))
                data.set("y", str(attr_value[1]))
                data.set("z", str(attr_value[2]))
                if "probability" in attr_type:
                    data.set("p", str(attr_value[3]))
            else:
                raise NotImplementedError("Unsupported attribute type encountered!")

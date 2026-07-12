"""
This module provides a RouteManager class for managing routes in the CARLA simulator.
It handles loading, saving, and manipulating routes, including adding and removing waypoints,
scenarios, and weather conditions. The routes are stored as Route objects, which encapsulate
the route data and related functionality.

Proposed file name: route_manager.py
"""

from carla_route import Route
from lxml import etree
from carla_simulator_client import CarlaClient
from pathlib import Path
import config


class RouteManager:
    def __init__(self, carla_client):
        """
        Initialize the RouteManager.

        Args:
            carla_client (CarlaClient): The CARLA client instance.
        """
        self.carla_client = carla_client

        self.routes = {}
        self.selected_route_id = None
        self.weather = None
        self.route_file_paths = {}
        self.dirty_route_ids = set()
        # We assume every route file is only located in the same map. Technically, that's not necessarily true,
        # but all route files that we have are only located in the same map per file.
        self.map_name = None

    def empty_routes(self, map_name):
        """
        Clear existing routes, load the specified map, and add an empty route.

        Args:
            map_name (str): The name of the map to load.
        """
        self.map_name = map_name
        self.routes.clear()
        self.route_file_paths.clear()
        self.dirty_route_ids.clear()
        self.carla_client.load_map(map_name)
        self.weather = self.carla_client.get_weather()

        self.add_empty_route()
        self.mark_selected_route_dirty()

    def load_routes_from_file(self, file_path, append=False):
        """
        Load routes from an XML file.

        Args:
            file_path (str): The path to the XML file containing the routes.

        Returns:
            None if the file doesn't have an '.xml' extension.
        """
        if not file_path.endswith(".xml"):
            return None

        previous_selected_route_id = self.selected_route_id
        previously_dirty_route_ids = set(self.dirty_route_ids)

        if not append:
            self.routes.clear()
            self.route_file_paths.clear()
            self.dirty_route_ids.clear()
            self.selected_route_id = None

        root = etree.parse(file_path)
        route_elems = list(root.iter("route"))
        loaded_route_ids = []
        for i, route_elem in enumerate(route_elems):
            map_name = route_elem.get("town")
            if i == 0 and (not append or not self.routes):
                self.carla_client.load_map(map_name)
                self.weather = self.carla_client.get_weather()

            route_id = int(route_elem.get("id"))
            while route_id in self.routes:
                route_id += 1
            self.map_name = map_name
            weather_element = route_elem.find("weathers")
            waypoints = [
                [float(pos.get("x")), float(pos.get("y")), float(pos.get("z"))]
                for pos in route_elem.findall("./waypoints/position")
            ]

            scenarios = route_elem.findall("./scenarios/scenario")
            scenario_types = [scenario.get("type") for scenario in scenarios]
            scenario_trigger_points = [
                [float(trigger.get("x")), float(trigger.get("y")), float(trigger.get("z"))]
                for trigger in [scenario.find("trigger_point") for scenario in scenarios]
            ]

            self.routes[route_id] = Route(
                self.carla_client,
                route_id,
                map_name,
                weather_element,
                waypoints,
                scenarios,
                scenario_types,
                scenario_trigger_points,
            )
            self.route_file_paths[route_id] = str(Path(file_path))
            loaded_route_ids.append(route_id)

        if loaded_route_ids:
            self.selected_route_id = loaded_route_ids[0]
        elif self.routes:
            self.selected_route_id = next(iter(self.routes.keys()))

        if append:
            self.dirty_route_ids = previously_dirty_route_ids - set(loaded_route_ids)
            if previous_selected_route_id in self.routes and previous_selected_route_id not in loaded_route_ids:
                self.selected_route_id = previous_selected_route_id
        else:
            self.dirty_route_ids.clear()

    def _build_routes_xml_tree(self, routes):
        routes_elem = etree.Element("routes")

        for route in routes:
            route_elem = etree.SubElement(routes_elem, "route")
            route_elem.attrib["id"] = str(route.route_id)
            route_elem.attrib["town"] = route.map_name
            route_elem.append(self._clone_elem(route.weather_element))

            waypoints_elem = etree.SubElement(route_elem, "waypoints")
            for wp in route.waypoints:
                loc = etree.SubElement(waypoints_elem, "position")
                loc.attrib.update({coord: str(value) for coord, value in zip(["x", "y", "z"], wp)})

            scenarios_elem = etree.SubElement(route_elem, "scenarios")
            for scenario in route.scenarios:
                scenarios_elem.append(self._clone_elem(scenario))

        return etree.ElementTree(routes_elem)

    @staticmethod
    def _clone_elem(elem):
        return etree.fromstring(etree.tostring(elem))

    @staticmethod
    def _append_suffix_to_xml_path(file_path, suffix):
        path = Path(file_path)
        return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))

    def _build_baseline_variant_route(self, route):
        baseline_scenarios = []
        baseline_types = []
        baseline_trigger_points = []
        has_conversion = False

        type_counts = {}
        for scenario_elem, scenario_type, trigger_point in zip(
            route.scenarios, route.scenario_types, route.scenario_trigger_points
        ):
            # Pedestrian scenarios: if the walker is an animal, generate a
            # baseline with a generic pedestrian walker instead.
            if scenario_type in config.PEDESTRIAN_WALKER_SCENARIO_TYPES:
                walker_elem = scenario_elem.find("walker")
                walker_val = walker_elem.get("value", "") if walker_elem is not None else ""
                if walker_val.startswith("walker.animal."):
                    has_conversion = True
                    cloned_scenario = self._clone_elem(scenario_elem)
                    cloned_walker = cloned_scenario.find("walker")
                    if cloned_walker is not None:
                        cloned_walker.set("value", "walker.pedestrian.*")
                    baseline_scenarios.append(cloned_scenario)
                    baseline_types.append(scenario_type)
                    baseline_trigger_points.append(list(trigger_point))
                continue

            is_mapped_fail2drive = scenario_type in config.FAIL2DRIVE_EXPORT_BASELINE_BY_TYPE
            baseline_type = self._resolve_baseline_type(scenario_elem, scenario_type)

            if is_mapped_fail2drive:
                has_conversion = True

            # Baseline is route-without-this-scenario.
            if baseline_type is None:
                continue

            cloned_scenario = self._clone_elem(scenario_elem)
            if is_mapped_fail2drive:
                cloned_scenario.set("type", baseline_type)
                idx = type_counts.get(baseline_type, 0)
                cloned_scenario.set("name", f"{baseline_type}_{idx}")
                type_counts[baseline_type] = idx + 1
                for tag in ("objects", "baseline_overlay", "overlay_direction"):
                    elem = cloned_scenario.find(tag)
                    if elem is not None:
                        cloned_scenario.remove(elem)

            baseline_scenarios.append(cloned_scenario)
            baseline_types.append(baseline_type)
            baseline_trigger_points.append(list(trigger_point))

        if not has_conversion:
            return None

        return Route(
            self.carla_client,
            route.route_id,
            route.map_name,
            self._clone_elem(route.weather_element),
            [list(wp) for wp in route.waypoints],
            baseline_scenarios,
            baseline_types,
            baseline_trigger_points,
        )

    @staticmethod
    def _resolve_baseline_type(scenario_elem, scenario_type):
        if scenario_type not in ("CustomObstacle", "CustomObstacleTwoWays"):
            return config.FAIL2DRIVE_EXPORT_BASELINE_BY_TYPE.get(scenario_type, scenario_type)

        overlay_elem = scenario_elem.find("baseline_overlay")
        overlay = None
        if overlay_elem is not None:
            value = overlay_elem.get("value")
            if value is not None:
                overlay = value.strip().lower()

        if overlay is None:
            return config.FAIL2DRIVE_EXPORT_BASELINE_BY_TYPE.get(scenario_type, scenario_type)

        if overlay == "none":
            return None

        mapping_one_way = {
            "construction": "ConstructionObstacle",
            "accident": "Accident",
            "parkedvehicle": "ParkedObstacle",
            "parked": "ParkedObstacle",
        }
        mapping_two_way = {
            "construction": "ConstructionObstacleTwoWays",
            "accident": "AccidentTwoWays",
            "parkedvehicle": "ParkedObstacleTwoWays",
            "parked": "ParkedObstacleTwoWays",
        }
        mapping = mapping_two_way if scenario_type == "CustomObstacleTwoWays" else mapping_one_way
        return mapping.get(overlay, mapping["construction"])

    def save_routes_to_file(self, file_path):
        """
        Save routes to an XML file.

        Args:
            file_path (str): The path to save the XML file.
        """
        if not file_path.endswith(".xml"):
            file_path = file_path + ".xml"

        tree = self._build_routes_xml_tree([route for route in self.routes.values()])
        tree.write(file_path, pretty_print=True)

    def save_selected_route_to_file(self, file_path=None):
        if self.selected_route_id is None:
            return None

        route = self.routes[self.selected_route_id]
        if file_path is None:
            file_path = self.route_file_paths.get(self.selected_route_id, None)
            if file_path is None:
                return None

        if not file_path.endswith(".xml"):
            file_path = file_path + ".xml"

        baseline_path = self._write_single_route(route, file_path, write_paired_fail2drive=True)
        self.route_file_paths[self.selected_route_id] = file_path
        self.dirty_route_ids.discard(self.selected_route_id)
        return file_path, baseline_path

    def _write_single_route(self, route, file_path, write_paired_fail2drive=False):
        tree = self._build_routes_xml_tree([route])
        tree.write(file_path, pretty_print=True)
        if not write_paired_fail2drive:
            return None

        baseline_variant = self._build_baseline_variant_route(route)
        if baseline_variant is None:
            return None

        baseline_path = self._append_suffix_to_xml_path(file_path, "_baseline")
        baseline_tree = self._build_routes_xml_tree([baseline_variant])
        baseline_tree.write(baseline_path, pretty_print=True)
        return baseline_path

    def save_all_routes_to_directory(self, directory_path):
        directory = Path(directory_path)
        directory.mkdir(parents=True, exist_ok=True)

        previous_selected_route_id = self.selected_route_id
        for route_id in sorted(self.routes.keys()):
            self.selected_route_id = route_id
            existing = self.route_file_paths.get(route_id)
            if existing is not None:
                existing_path = Path(existing)
                file_path = existing_path if existing_path.parent == directory else directory / existing_path.name
            else:
                file_path = directory / f"route_{route_id}.xml"
            self.save_selected_route_to_file(str(file_path))
        self.selected_route_id = previous_selected_route_id

    def save_dirty_routes_to_directory(self, directory_path):
        directory = Path(directory_path)
        directory.mkdir(parents=True, exist_ok=True)

        for route_id in sorted(self.dirty_route_ids.copy()):
            route = self.routes.get(route_id)
            if route is None:
                continue
            file_path = directory / f"route_{route_id}.xml"
            self._write_single_route(route, str(file_path), write_paired_fail2drive=False)

    def load_routes_from_directory(self, directory_path):
        directory = Path(directory_path)
        xml_files = sorted(f for f in directory.glob("*.xml") if not f.stem.endswith("_baseline"))
        self.routes.clear()
        self.route_file_paths.clear()
        self.dirty_route_ids.clear()
        self.selected_route_id = None

        for i, xml_file in enumerate(xml_files):
            self.load_routes_from_file(str(xml_file), append=i > 0)

    def mark_selected_route_dirty(self):
        if self.selected_route_id is not None:
            self.dirty_route_ids.add(self.selected_route_id)

    def generate_random_weather_elem(self):
        """
        Generate a random weather XML element based on the current weather in the CARLA world.

        Returns:
            lxml.etree.Element: The generated weather XML element.
        """
        weather = self.weather

        weather_string = f"         <weather\n"
        weather_string += f'            route_percentage="0"\n'
        weather_string += f'            cloudiness="{weather.cloudiness}" '
        weather_string += f'precipitation="{weather.precipitation}" '
        weather_string += f'precipitation_deposits="{weather.precipitation_deposits}" '
        weather_string += f'wetness="{weather.wetness}"\n'
        weather_string += f'            wind_intensity="{weather.wind_intensity}" '
        weather_string += f'sun_azimuth_angle="{weather.sun_azimuth_angle}" '
        weather_string += f'sun_altitude_angle="{weather.sun_altitude_angle}"\n'
        weather_string += f'            fog_density="{weather.fog_density}" '
        weather_string += f'fog_distance="{weather.fog_distance}" '
        weather_string += f'fog_falloff="{round(weather.fog_falloff, 2)}" '
        weather_string += f'scattering_intensity="{weather.scattering_intensity}"\n'
        weather_string += f'            mie_scattering_scale="{round(weather.mie_scattering_scale, 2)}"/>'

        weather_elem1 = etree.fromstring(weather_string)
        weather_elem2 = etree.fromstring(weather_string.replace('route_percentage="0"', 'route_percentage="100"'))
        weathers_elem = etree.Element("weathers")
        weathers_elem.append(weather_elem1)
        weathers_elem.append(weather_elem2)

        return weathers_elem

    def add_empty_route(self):
        """
        Add an empty route with a unique ID.

        Returns:
            dict: The updated routes dictionary.
            int: The ID of the newly added route.
        """
        route_id = 0
        while route_id in self.routes:
            route_id += 1

        waypoints, scenarios, scenario_types, scenario_trigger_points = [], [], [], []
        weather_elem = self.generate_random_weather_elem()

        route = Route(
            self.carla_client,
            route_id,
            self.map_name,
            weather_elem,
            waypoints,
            scenarios,
            scenario_types,
            scenario_trigger_points,
        )
        self.routes[route_id] = route
        self.route_file_paths[route_id] = None
        self.selected_route_id = route_id
        self.dirty_route_ids.add(route_id)

        return self.routes, self.selected_route_id

    def remove_selected_route(self):
        """
        Remove the currently selected route.
        """
        if self.selected_route_id is None or self.selected_route_id not in self.routes:
            return

        del self.routes[self.selected_route_id]
        self.route_file_paths.pop(self.selected_route_id, None)
        self.dirty_route_ids.discard(self.selected_route_id)

        if self.routes:
            self.selected_route_id = next(iter(self.routes.keys()))
        else:
            self.selected_route_id = None


if __name__ == "__main__":
    carla_client = CarlaClient()

    route_manager = RouteManager(carla_client)
    route_manager.load_routes_from_file(
        "/home/jens/Desktop/Hiwi-Work/leaderboard2_human_data/leaderboard/data/routes_devtest.xml"
    )

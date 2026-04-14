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
        self.carla_client.load_map(map_name)
        self.weather = self.carla_client.carla_world.get_weather()

        self.add_empty_route()

    def load_routes_from_file(self, file_path):
        """
        Load routes from an XML file.

        Args:
            file_path (str): The path to the XML file containing the routes.

        Returns:
            None if the file doesn't have an '.xml' extension.
        """
        if not file_path.endswith(".xml"):
            return None

        self.routes.clear()
        self.selected_route_id = None

        root = etree.parse(file_path)
        for i, route_elem in enumerate(root.iter("route")):
            map_name = route_elem.get("town")
            if i == 0:
                self.carla_client.load_map(map_name)
                self.weather = self.carla_client.carla_world.get_weather()

            route_id = int(route_elem.get("id"))
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

        if self.routes:
            self.selected_route_id = next(iter(self.routes.keys()))

    def save_routes_to_file(self, file_path):
        """
        Save routes to an XML file.

        Args:
            file_path (str): The path to save the XML file.
        """
        if not file_path.endswith(".xml"):
            file_path = file_path + ".xml"

        routes_elem = etree.Element("routes")

        for route_id, route in self.routes.items():
            route_elem = etree.SubElement(routes_elem, "route")
            route_elem.attrib["id"] = str(route_id)
            route_elem.attrib["town"] = route.map_name
            route_elem.append(route.weather_element)

            waypoints_elem = etree.SubElement(route_elem, "waypoints")
            for wp in route.waypoints:
                loc = etree.SubElement(waypoints_elem, "position")
                loc.attrib.update({coord: str(value) for coord, value in zip(["x", "y", "z"], wp)})

            scenarios_elem = etree.SubElement(route_elem, "scenarios")
            for scenario in route.scenarios:
                scenarios_elem.append(scenario)

        tree = etree.ElementTree(routes_elem)
        tree.write(file_path, pretty_print=True)

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
        self.selected_route_id = route_id

        return self.routes, self.selected_route_id

    def remove_selected_route(self):
        """
        Remove the currently selected route.
        """
        del self.routes[self.selected_route_id]

        if self.routes:
            self.selected_route_id = next(iter(self.routes.keys()))


if __name__ == "__main__":
    carla_client = CarlaClient()

    route_manager = RouteManager(carla_client)
    route_manager.load_routes_from_file(
        "/home/jens/Desktop/Hiwi-Work/leaderboard2_human_data/leaderboard/data/routes_devtest.xml"
    )

"""
This file provides a class `CarlaClient` that establishes a connection with the CARLA simulator,
loads a specified map, and aggregates relevant map data such as stop sign centers, traffic light centers,
and waypoints. Additionally, it calculates the map dimensions and provides methods to access the available maps.
"""

import carla
import os
import pickle
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner


class ConnectionError(Exception):
    """Exception raised when a connection cannot be established with the CARLA simulator."""

    pass


class CarlaClient:
    def __init__(self, host="localhost", port=2000, carla_map_dir="carla_map_data"):
        """
        Initializes the CarlaClient and establishes a connection with the CARLA simulator.

        Args:
            host (str): The host address of the CARLA simulator.
            port (int): The port number of the CARLA simulator.
            carla_map_dir (str): The directory path where CARLA map data is stored.

        Raises:
            FileNotFoundError: If the specified CARLA map data directory does not exist.
            ConnectionError: If the connection to the CARLA simulator cannot be established.
        """
        self.client = carla.Client(host, port)
        self.carla_world = None
        self.carla_map = None
        self.global_route_planner = None
        self.carla_map_dir = carla_map_dir

        if not os.path.exists(carla_map_dir):
            raise FileNotFoundError("The path of the CARLA map data does not exist!")

        # Test if the CARLA simulator is running
        try:
            self.client.set_timeout(1)
            self.client.get_server_version()
            self.client.set_timeout(120)
        except:
            raise ConnectionError(
                "Failed to connect to the simulator. Make sure the simulator is running & the host and port are correct!"
            )

    def get_available_maps(self):
        """
        Returns a list of available maps in the CARLA simulator.

        Returns:
            list: A list of available map names.
        """
        available_maps = self.client.get_available_maps()
        available_maps = [x.split("/")[-1] for x in available_maps]
        return available_maps

    def load_map(self, map_name):
        """
        Loads the specified map in the CARLA simulator and initializes the global route planner.

        Args:
            map_name (str): The name of the map to load.
        """
        if self.carla_world is None:
            self.carla_world = self.client.get_world()
            self.carla_map = self.carla_world.get_map()
            self.global_route_planner = GlobalRoutePlanner(self.carla_map, 1.0)

        if map_name not in self.carla_map.name:
            self.carla_world = self.client.load_world(map_name)
            self.carla_map = self.carla_world.get_map()
            self.global_route_planner = GlobalRoutePlanner(self.carla_map, 1.0)

        self.aggregate_map_data(map_name)

    def aggregate_map_data(self, map_name):
        """
        Aggregates relevant map data for the specified map, such as stop sign centers, traffic light centers,
        and waypoints. Also calculates the map dimensions.

        Args:
            map_name (str): The name of the map for which to aggregate data.
        """
        with open(os.path.join(self.carla_map_dir, f"{map_name}.pkl"), "rb") as file:
            data = pickle.load(file)

        # Process stop sign centers
        stop_sign_centers_np = data["stop_sign_centers_np"]
        if stop_sign_centers_np.shape[0]:
            stop_sign_centers_np = stop_sign_centers_np[:, :2]
        stop_sign_centers_np = stop_sign_centers_np.reshape((-1, 2))

        # Process traffic light centers
        traffic_light_centers_np = data["traffic_light_centers_np"]
        if traffic_light_centers_np.shape[0]:
            traffic_light_centers_np = traffic_light_centers_np[:, :2]
        traffic_light_centers_np = traffic_light_centers_np.reshape((-1, 2))

        all_waypoints_np = data["all_waypoints_np"][:, :2]
        num_road_waypoints = data["num_road_waypoints"]

        self.stop_sign_centers_np = stop_sign_centers_np
        self.traffic_light_centers_np = traffic_light_centers_np

        # Calculate map dimensions
        self.road_waypoints_np = all_waypoints_np[:num_road_waypoints]
        self.parking_waypoints_np = all_waypoints_np[num_road_waypoints:]
        self.min_coords = all_waypoints_np.min(axis=0)
        self.max_coords = all_waypoints_np.max(axis=0)
        self.map_width, self.map_height = (self.max_coords - self.min_coords)[:2].astype("int")
        self.map_size = np.array([self.map_width, self.map_height])

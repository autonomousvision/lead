"""
This file provides a class `CarlaClient` that loads OpenDRIVE maps from disk,
builds a local CARLA map object for routing, and aggregates map metadata such as
stop sign centers, traffic light centers, and waypoints from precomputed files.
"""

import carla
import os
import pickle
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner


class CarlaClient:
    def __init__(self, carla_map_dir="carla_map_data", xodr_map_dir="carla_xodr"):
        """
        Initializes the CarlaClient in offline mode using local OpenDRIVE files.

        Args:
            carla_map_dir (str): The directory path where CARLA map data is stored.
            xodr_map_dir (str): The directory path where OpenDRIVE (.xodr) maps are stored.

        Raises:
            FileNotFoundError: If one of the required directories does not exist.
        """
        self.carla_map = None
        self.global_route_planner = None
        self.carla_map_dir = carla_map_dir
        self.xodr_map_dir = xodr_map_dir
        self.current_map_name = None

        if not os.path.exists(carla_map_dir):
            raise FileNotFoundError("The path of the CARLA map data does not exist!")

        if not os.path.exists(xodr_map_dir):
            raise FileNotFoundError("The path of the OpenDRIVE map data does not exist!")

        self.weather = carla.WeatherParameters.ClearNoon

    def get_available_maps(self):
        """
        Returns a list of map names available in both the OpenDRIVE and map-data directories.

        Returns:
            list: A list of available map names.
        """
        available_xodr_maps = {os.path.splitext(x)[0] for x in os.listdir(self.xodr_map_dir) if x.endswith(".xodr")}
        available_pkl_maps = {os.path.splitext(x)[0] for x in os.listdir(self.carla_map_dir) if x.endswith(".pkl")}
        return sorted(list(available_xodr_maps & available_pkl_maps))

    def load_map(self, map_name):
        """
        Loads the specified map from OpenDRIVE and initializes the global route planner.

        Args:
            map_name (str): The name of the map to load.
        """
        if self.current_map_name != map_name:
            xodr_path = os.path.join(self.xodr_map_dir, f"{map_name}.xodr")
            if not os.path.exists(xodr_path):
                raise FileNotFoundError(f"Missing OpenDRIVE file: {xodr_path}")

            with open(xodr_path, "r", encoding="utf-8") as f:
                xodr_data = f.read()

            self.carla_map = carla.Map(map_name, xodr_data)
            self.global_route_planner = GlobalRoutePlanner(self.carla_map, 1.0)
            self.current_map_name = map_name

        self.aggregate_map_data(map_name)

    def get_weather(self):
        """
        Returns weather data used for writing route XML files.
        """
        return self.weather

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
        self.biking_waypoints_np = self._collect_biking_waypoints()
        self.min_coords = all_waypoints_np.min(axis=0)
        self.max_coords = all_waypoints_np.max(axis=0)
        self.map_width, self.map_height = (self.max_coords - self.min_coords)[:2].astype("int")
        self.map_size = np.array([self.map_width, self.map_height])

    def _collect_biking_waypoints(self):
        """
        Collect representative biking-lane waypoint coordinates from the loaded map.
        """
        if self.carla_map is None:
            return np.zeros((0, 2), dtype="float")

        road_wps = self.carla_map.generate_waypoints(2.0)
        biking_points = []
        for wp in road_wps:
            left_wp = wp.get_left_lane()
            right_wp = wp.get_right_lane()
            if left_wp is not None and left_wp.lane_type == carla.LaneType.Biking:
                biking_points.append([left_wp.transform.location.x, left_wp.transform.location.y])
            if right_wp is not None and right_wp.lane_type == carla.LaneType.Biking:
                biking_points.append([right_wp.transform.location.x, right_wp.transform.location.y])

        if not biking_points:
            return np.zeros((0, 2), dtype="float")

        biking_np = np.array(biking_points, dtype="float")
        # Deduplicate to avoid overdraw from repeated neighborhood lookups.
        biking_np = np.unique(np.round(biking_np, 1), axis=0)
        return biking_np

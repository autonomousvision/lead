"""Lifecycle of the expert's world state: setup and lazy initialization."""

import logging
import os
import pathlib
import xml.etree.ElementTree as ET
from functools import cached_property

import carla
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.config import load_lead_config
from lead.expert.driving.privileged_route_planner import PrivilegedRoutePlanner
from lead.expert.utils import roadgraph

LOG = logging.getLogger(__name__)


class LifecycleMixin:
    """Sets up privileged access to the CARLA world and initializes the route."""

    def setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ):
        """
        Set up the expert agent for the CARLA simulation.

        Args:
            path_to_conf_file: Path to the configuration file.
            route_index: Index of the route to follow.
            traffic_manager: The traffic manager object.
        """
        super().setup(load_lead_config())
        LOG.info("Setup")
        self.initialized = False

        self.config_path = path_to_conf_file
        self.step = -1
        self.save_path = None
        self.route_index = route_index
        self.scenario_name = self._parse_scenario_type(path_to_conf_file)
        self.list_traffic_lights: list[
            tuple[carla.TrafficLight, carla.Location, list[carla.Waypoint]]
        ] = []
        self.close_traffic_lights: list[
            tuple[
                carla.TrafficLight,
                carla.BoundingBox,
                carla.TrafficLightState,
                int,
                bool,
            ]
        ] = []
        self.close_stop_signs = []

        self.track = autonomous_agent.Track.MAP
        # Privileged access
        self.ego_vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self.carla_world: carla.World = self.ego_vehicle.get_world()
        self.carla_world_map: carla.Map = CarlaDataProvider.get_map()

    def _init(self) -> None:
        """Lazily initialize world-dependent components on the first step."""
        LOG.info("Init")

        # Check if the vehicle starts from a parking spot
        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(
            self.ego_vehicle.get_location(),
        )

        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2
        LOG.info(
            f"Vehicle starts {'with' if starts_with_parking_exit else 'without'} parking exit.",
        )

        # Set up the route planner and extrapolation
        self.privileged_route_planner = PrivilegedRoutePlanner(self.config_expert)
        self.privileged_route_planner.setup_route(
            self.org_dense_route_world_coord,
            self.carla_world,
            self.carla_world_map,
            starts_with_parking_exit,
            self.ego_vehicle.get_location(),
        )
        self.privileged_route_planner.save()
        LOG.info(
            f"Route setup with {len(self.privileged_route_planner.route_waypoints)} waypoints.",
        )

        # Preprocess traffic lights
        all_actors = self.carla_world.get_actors()
        for actor in all_actors:
            if "traffic_light" in actor.type_id:
                center, waypoints = roadgraph.get_traffic_light_waypoints(
                    actor,
                    self.carla_world_map,
                )
                self.list_traffic_lights.append((actor, center, waypoints))

        # Remove bugged 2-wheelers
        # https://github.com/carla-simulator/carla/issues/3670
        for actor in all_actors:
            if "vehicle" in actor.type_id:
                extent = actor.bounding_box.extent
                if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
                    actor.destroy()

    @staticmethod
    def _parse_scenario_type(path_to_conf_file: str) -> str:
        """Scenario type of the first scenario in the routes XML, or the parent
        directory name if the file has none or is not a routes XML."""
        try:
            scenario = ET.parse(path_to_conf_file).getroot().find(".//scenario")
            scenario_type = scenario.get("type") if scenario is not None else None
            if scenario_type:
                return scenario_type
        except (ET.ParseError, OSError):
            pass
        return pathlib.Path(path_to_conf_file).parent.name

    @property
    def town(self):
        return self.carla_world.get_map().name.split("/")[-1]

    @cached_property
    def rep(self):
        return os.environ.get("REPETITION", "-1")

    @property
    def log_name(self):
        return f"{self.town}_Rep{self.rep}_{self.route_index}"

"""Debug drawing of bounding boxes, routes and traffic lights in CARLA."""

import logging
import typing

import carla

if typing.TYPE_CHECKING:
    from lead.lead.planning import forecast_kernels

LOG = logging.getLogger(__name__)


class DataVisualizationMixin:
    """Debug visualization of internal expert data via CARLA debug drawing."""

    def visualize_ego_bb(self, ego_bb_global: carla.BoundingBox):
        ego_vehicle_transform = self.ego_vehicle.get_transform()
        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(
            self.ego_vehicle.bounding_box.location,
        )
        ego_bb_global = carla.BoundingBox(
            center_ego_bb_global,
            self.ego_vehicle.bounding_box.extent,
        )
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.config_expert.visualization.visualize_bounding_boxes:
            self.carla_world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=carla.Color(
                    *self.config_expert.visualization.ego_vehicle_bb_color,
                ),
                life_time=self.config_expert.visualization.draw_life_time,
            )

    def visualize_lead_and_trailing_vehicles(self):
        if self.config_expert.visualization.visualize_internal_data:
            vehicle_list = ...

            leading_vehicle_ids = (
                self.privileged_route_planner.compute_leading_vehicles(
                    vehicle_list,
                    self.ego_vehicle.id,
                )
            )
            trailing_vehicle_ids = (
                self.privileged_route_planner.compute_trailing_vehicles(
                    vehicle_list,
                    self.ego_vehicle.id,
                )
            )

            for vehicle in vehicle_list:
                if vehicle.id in leading_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Leading Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.visualization.draw_life_time,
                        color=carla.Color(
                            *self.config_expert.visualization.leading_vehicle_color,
                        ),
                    )
                elif vehicle.id in trailing_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Trailing Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.visualization.draw_life_time,
                        color=carla.Color(
                            *self.config_expert.visualization.trailing_vehicle_color,
                        ),
                    )

    def visualize_forecasted_bounding_boxes(
        self,
        predicted_bounding_boxes: "dict[int, forecast_kernels.ActorForecast]",
    ):
        if self.config_expert.visualization.visualize_bounding_boxes:
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for _actor_idx, forecast in predicted_bounding_boxes.items():
                for i in range(forecast.centers.shape[0]):
                    color = carla.Color(
                        *self.config_expert.visualization.other_vehicles_forecasted_bbs_color,
                    )
                    if (
                        _actor_idx in dangerous_adversarial_actors_ids
                        or _actor_idx in safe_adversarial_actors_ids
                    ):
                        color = carla.Color(
                            *self.config_expert.visualization.adversarial_color,
                        )
                    bb = carla.BoundingBox(
                        carla.Location(
                            x=forecast.centers[i, 0],
                            y=forecast.centers[i, 1],
                            z=forecast.centers[i, 2],
                        ),
                        carla.Vector3D(
                            x=forecast.extents[i, 0],
                            y=forecast.extents[i, 1],
                            z=forecast.extents[i, 2],
                        ),
                    )
                    bb.rotation = carla.Rotation(
                        pitch=0,
                        yaw=forecast.yaws_deg[i],
                        roll=0,
                    )
                    self.carla_world.debug.draw_box(
                        box=bb,
                        rotation=bb.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=self.config_expert.visualization.draw_life_time,
                    )

                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if (
                        vehicle_id in self.leading_vehicle_ids
                        and not self.near_lane_change
                    ):
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=carla.Color(
                                *self.config_expert.visualization.leading_vehicle_color,
                            ),
                            life_time=self.config_expert.visualization.draw_life_time,
                        )
                    elif vehicle_id in self.trailing_vehicle_ids:
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0,
                            yaw=vehicle.get_transform().rotation.yaw,
                            roll=0,
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=carla.Color(
                                *self.config_expert.visualization.trailing_vehicle_color,
                            ),
                            life_time=self.config_expert.visualization.draw_life_time,
                        )

    def visualize_pedestrian_bounding_boxes(
        self,
        nearby_pedestrians_bbs: "list[forecast_kernels.WalkerForecast]",
    ):
        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.config_expert.visualization.visualize_bounding_boxes:
            for walker_forecast in nearby_pedestrians_bbs:
                rotation = carla.Rotation(
                    pitch=walker_forecast.rotation_deg[0],
                    yaw=walker_forecast.rotation_deg[1],
                    roll=walker_forecast.rotation_deg[2],
                )
                extent = carla.Vector3D(
                    x=walker_forecast.extent[0],
                    y=walker_forecast.extent[1],
                    z=walker_forecast.extent[2],
                )
                for i in range(walker_forecast.centers.shape[0]):
                    bbox = carla.BoundingBox(
                        carla.Location(
                            walker_forecast.centers[i, 0],
                            walker_forecast.centers[i, 1],
                            walker_forecast.centers[i, 2],
                        ),
                        extent,
                    )
                    bbox.rotation = rotation
                    self.carla_world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=carla.Color(
                            *self.config_expert.visualization.pedestrian_forecasted_bbs_color,
                        ),
                        life_time=self.config_expert.visualization.draw_life_time,
                    )

    def visualize_traffic_lights(
        self,
        traffic_light: carla.TrafficLight,
        wp: carla.Waypoint,
        bounding_box: carla.BoundingBox,
    ):
        if self.config_expert.visualization.visualize_traffic_lights_bounding_boxes:
            if traffic_light.state == carla.TrafficLightState.Red:
                color = carla.Color(
                    *self.config_expert.visualization.red_traffic_light_color,
                )
            elif traffic_light.state == carla.TrafficLightState.Yellow:
                color = carla.Color(
                    *self.config_expert.visualization.yellow_traffic_light_color,
                )
            elif traffic_light.state == carla.TrafficLightState.Green:
                color = carla.Color(
                    *self.config_expert.visualization.green_traffic_light_color,
                )
            elif traffic_light.state == carla.TrafficLightState.Off:
                color = carla.Color(
                    *self.config_expert.visualization.off_traffic_light_color,
                )
            else:  # unknown
                color = carla.Color(
                    *self.config_expert.visualization.unknown_traffic_light_color,
                )

            self.carla_world.debug.draw_box(
                box=bounding_box,
                rotation=bounding_box.rotation,
                thickness=0.1,
                color=color,
                life_time=0.051,
            )

            self.carla_world.debug.draw_point(
                wp.transform.location
                + carla.Location(z=traffic_light.trigger_volume.location.z),
                size=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.simulation.carla_fps) + 1e-6,
            )

            self.carla_world.debug.draw_box(
                box=traffic_light.bounding_box,
                rotation=traffic_light.bounding_box.rotation,
                thickness=0.1,
                color=color,
                life_time=0.051,
            )

    def visualize_stop_signs(
        self,
        bounding_box_stop_sign: carla.BoundingBox,
        affects_ego: bool,
    ):
        if self.config_expert.visualization.visualize_bounding_boxes:
            color = carla.Color(0, 1, 0) if affects_ego else carla.Color(1, 0, 0)
            self.carla_world.debug.draw_box(
                box=bounding_box_stop_sign,
                rotation=bounding_box_stop_sign.rotation,
                thickness=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.simulation.carla_fps) + 1e-6,
            )

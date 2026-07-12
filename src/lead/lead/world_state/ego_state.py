"""Cached per-step view of the ego vehicle's state."""

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.lead.utils.expert_utils import step_cached_property


class EgoStateMixin:
    """Cached per-step ego pose, speed and lane properties."""

    @step_cached_property
    def privileged_ego_past_positions(self) -> list[list[float]]:
        # World positions were recorded at enqueue time; only the projection
        # into the current ego frame happens here.
        ego_matrix_current = self.transform_queue[-1].get_matrix()
        T_world_to_current_ego = np.linalg.inv(ego_matrix_current)
        past_positions = []
        for position in self.ego_position_queue:
            pos_world = np.append(np.array(position), 1.0)
            pos_current_ego = T_world_to_current_ego @ pos_world
            past_positions.append(pos_current_ego[:2].tolist())
        return past_positions

    @step_cached_property
    def last_encountered_speed_limit_sign(self) -> float:
        ret = self.ego_vehicle.get_speed_limit()
        if ret is not None:
            ret /= 3.6
        return ret

    @step_cached_property
    def speed_limit(self) -> float:
        if self.last_encountered_speed_limit_sign is not None:
            return self.last_encountered_speed_limit_sign
        return 30.0 / 3.6

    @step_cached_property
    def ego_lane_width(self) -> float:
        """
        Returns the width of the lane the ego vehicle is currently on.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        return self.ego_lane.lane_width

    @step_cached_property
    def ego_lane(self) -> carla.Waypoint:
        """
        Returns the current lane of the ego vehicle as a CARLA waypoint.
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized. Call setup() first.")
        return self.carla_world_map.get_waypoint(
            self.ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    @step_cached_property
    def ego_lane_id(self) -> int:
        """
        Returns the current lane ID of the ego vehicle.
        """
        return self.ego_lane.lane_id

    @step_cached_property
    def ego_transform(self) -> carla.Transform:
        return self.ego_vehicle.get_transform()

    @step_cached_property
    def ego_location(self) -> carla.Location:
        return self.ego_vehicle.get_location()

    @property
    def ego_location_array(self) -> jt.Float[npt.NDArray, " 3"]:
        """
        Returns the ego vehicle's location as a numpy array.
        """
        location = self.ego_location
        return np.array([location.x, location.y, location.z])

    @step_cached_property
    def ego_speed(self) -> float:
        return self.ego_vehicle.get_velocity().length()

    def _get_actor_forward_speed(self, actor: carla.Actor) -> float:
        return self._get_forward_speed(
            transform=actor.get_transform(),
            velocity=actor.get_velocity(),
        )

    def _get_forward_speed(
        self,
        transform: carla.Transform | None = None,
        velocity: carla.Vector3D | None = None,
    ) -> float:
        """
        Calculate the forward speed of the vehicle based on its transform and velocity.

        Args:
            transform: The transform of the vehicle. If not provided, it will be obtained from ego.
            velocity: The velocity of the vehicle. If not provided, it will be obtained from ego.

        Returns:
            The forward speed of the vehicle in m/s.
        """
        if not velocity:
            velocity = self.ego_vehicle.get_velocity()

        if not transform:
            transform = self.ego_vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array(
            [
                np.cos(pitch_rad) * np.cos(yaw_rad),
                np.cos(pitch_rad) * np.sin(yaw_rad),
                np.sin(pitch_rad),
            ],
        )

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return float(forward_speed)

    @step_cached_property
    def ego_yaw_degree(self) -> float:
        return self.ego_vehicle.get_transform().rotation.yaw

    @step_cached_property
    def ego_orientation_rad(self) -> float | None:
        return self.compass

    @step_cached_property
    def ego_matrix(self) -> jt.Float[npt.NDArray, "4 4"]:
        return np.array(self.ego_vehicle.get_transform().get_matrix())

    @step_cached_property
    def inv_ego_matrix(self) -> jt.Float[npt.NDArray, "4 4"]:
        return np.linalg.inv(self.ego_matrix)

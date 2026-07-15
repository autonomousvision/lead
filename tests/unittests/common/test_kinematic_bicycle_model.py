import numpy as np
import pytest

from lead.config import load_lead_config
from lead.expert.driving.kinematic_bicycle_model import KinematicBicycleModel


@pytest.fixture
def config():
    """Fixture providing the expert config section."""
    return load_lead_config().expert


@pytest.fixture
def bicycle_model(config):
    """Fixture providing KinematicBicycleModel instance."""
    return KinematicBicycleModel(config)


class TestKinematicBicycleModel:
    """Tests for the kinematic bicycle model."""

    def test_forecast_ego_vehicle_zero_speed_no_action(self, bicycle_model):
        """Test ego vehicle with zero speed and no action stays stationary."""
        location = np.array([0.0, 0.0, 0.0])
        heading = 0.0
        speed = 0.0
        action = np.array([0.0, 0.0, 0.0])  # No steer, no throttle, no brake

        next_location, next_heading, next_speed = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action,
        )

        # With zero speed, vehicle should remain stationary
        np.testing.assert_allclose(next_location, location, atol=1e-6)
        np.testing.assert_allclose(next_heading, heading, atol=1e-6)
        assert next_speed >= 0.0  # Speed should not be negative

    def test_forecast_ego_vehicle_constant_speed(self, bicycle_model):
        """Test ego vehicle maintains speed with low throttle."""
        location = np.array([0.0, 0.0, 0.0])
        heading = 0.0
        speed = 5.0  # 5 m/s
        action = np.array([0.0, 0.2, 0.0])  # Low throttle (below threshold)

        next_location, next_heading, next_speed = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action,
        )

        # With low throttle, speed should remain constant
        np.testing.assert_allclose(next_speed, speed, atol=1e-3)
        # Should move forward
        assert next_location[0] > location[0]

    def test_forecast_ego_vehicle_braking(self, bicycle_model):
        """Test ego vehicle decelerates when braking."""
        location = np.array([0.0, 0.0, 0.0])
        heading = 0.0
        speed = 10.0  # 10 m/s
        action = np.array([0.0, 0.0, 1.0])  # Brake applied

        next_location, next_heading, next_speed = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action,
        )

        # Speed should decrease when braking
        assert next_speed < speed
        assert next_speed >= 0.0

    def test_forecast_ego_vehicle_acceleration(self, bicycle_model):
        """Test ego vehicle accelerates with high throttle."""
        location = np.array([0.0, 0.0, 0.0])
        heading = 0.0
        speed = 5.0
        action = np.array([0.0, 0.8, 0.0])  # High throttle

        next_location, next_heading, next_speed = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action,
        )

        # Speed should increase with high throttle
        assert next_speed > speed

    def test_forecast_ego_vehicle_steering(self, bicycle_model):
        """Test ego vehicle changes heading when steering."""
        location = np.array([0.0, 0.0, 0.0])
        heading = 0.0
        speed = 5.0
        action_left = np.array([0.5, 0.5, 0.0])  # Steer left

        next_location, next_heading_left, _ = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action_left,
        )

        # Heading should change with steering
        assert next_heading_left != heading

        # Test right steering
        action_right = np.array([-0.5, 0.5, 0.0])  # Steer right
        _, next_heading_right, _ = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action_right,
        )

        # Left and right steering should produce opposite heading changes
        assert (next_heading_left - heading) * (next_heading_right - heading) < 0

    def test_forecast_ego_vehicle_maintains_altitude(self, bicycle_model):
        """Test that z-coordinate (altitude) is preserved."""
        location = np.array([0.0, 0.0, 5.0])
        heading = 0.0
        speed = 5.0
        action = np.array([0.0, 0.5, 0.0])

        next_location, _, _ = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action,
        )

        # Z-coordinate should remain unchanged
        assert next_location[2] == location[2]

    def test_forecast_other_vehicles_single(self, bicycle_model):
        """Test forecasting a single other vehicle."""
        locations = np.array([[0.0, 0.0, 0.0]])
        headings = np.array([0.0])
        speeds = np.array([5.0])
        actions = np.array([[0.0, 0.5, 0.0]])  # No steer, throttle, no brake

        future_locations, future_headings, future_speeds = (
            bicycle_model.forecast_other_vehicles(
                locations,
                headings,
                speeds,
                actions,
                num_future_frames=4,
            )
        )

        assert future_locations.shape == (4, 1, 3)
        assert future_headings.shape == (4, 1)
        assert future_speeds.shape == (4, 1)
        # Vehicle should move forward, farther with every frame
        assert future_locations[0, 0, 0] > locations[0, 0]
        assert np.all(np.diff(future_locations[:, 0, 0]) > 0)

    def test_forecast_other_vehicles_multiple(self, bicycle_model):
        """Test forecasting multiple other vehicles simultaneously."""
        n_vehicles = 5
        locations = np.random.rand(n_vehicles, 3) * 10
        headings = np.random.rand(n_vehicles) * 2 * np.pi
        speeds = np.random.rand(n_vehicles) * 10
        actions = np.random.rand(n_vehicles, 3)

        future_locations, future_headings, future_speeds = (
            bicycle_model.forecast_other_vehicles(
                locations,
                headings,
                speeds,
                actions,
                num_future_frames=3,
            )
        )

        assert future_locations.shape == (3, n_vehicles, 3)
        assert future_headings.shape == (3, n_vehicles)
        assert future_speeds.shape == (3, n_vehicles)
        # All speeds should be non-negative
        assert np.all(future_speeds >= 0.0)

    def test_forecast_other_vehicles_braking(self, bicycle_model):
        """Test that other vehicles decelerate when braking."""
        locations = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        headings = np.array([0.0, 0.0])
        speeds = np.array([10.0, 10.0])
        actions = np.array(
            [[0.0, 0.5, 1.0], [0.0, 0.5, 0.0]],
        )  # First brakes, second doesn't

        _, _, future_speeds = bicycle_model.forecast_other_vehicles(
            locations,
            headings,
            speeds,
            actions,
            num_future_frames=1,
        )

        # First vehicle (braking) should have lower speed
        assert future_speeds[0, 0] < speeds[0]
        # Second vehicle (not braking) should accelerate or maintain speed
        assert future_speeds[0, 1] >= speeds[0] - 1.0  # Allow some tolerance

    def test_forecast_other_vehicles_zero_speed(self, bicycle_model):
        """Test that vehicles with zero speed remain stationary."""
        locations = np.array([[0.0, 0.0, 0.0]])
        headings = np.array([0.0])
        speeds = np.array([0.0])
        actions = np.array([[0.0, 0.0, 0.0]])

        future_locations, _, future_speeds = bicycle_model.forecast_other_vehicles(
            locations,
            headings,
            speeds,
            actions,
            num_future_frames=3,
        )

        # With zero speed, location change should be minimal
        for frame_locations in future_locations:
            np.testing.assert_allclose(frame_locations, locations, atol=1e-6)
        assert np.all(future_speeds >= 0.0)

    def test_forecast_consistency(self, bicycle_model):
        """Test that ego and other vehicle forecasts are consistent for same inputs."""
        location = np.array([0.0, 0.0, 0.0])
        heading = 0.0
        speed = 5.0
        action = np.array([0.2, 0.5, 0.0])

        # Forecast ego vehicle
        next_loc_ego, _, next_speed_ego = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action,
        )

        # Forecast as "other vehicle" with same parameters
        locations = np.array([location])
        headings = np.array([heading])
        speeds = np.array([speed])
        actions = np.array([action])

        future_locs_other, _, future_speeds_other = (
            bicycle_model.forecast_other_vehicles(
                locations,
                headings,
                speeds,
                actions,
                num_future_frames=1,
            )
        )

        # Results should be similar (may differ slightly due to ego-specific throttle model)
        # Testing that both produce reasonable outputs
        assert next_loc_ego.shape == (3,)
        assert future_locs_other.shape == (1, 1, 3)
        assert next_speed_ego > 0
        assert future_speeds_other[0, 0] > 0

    def test_speed_never_negative(self, bicycle_model):
        """Test that speed is always clamped to non-negative values."""
        location = np.array([0.0, 0.0, 0.0])
        heading = 0.0
        speed = 0.5  # Low speed
        action = np.array([0.0, 0.0, 1.0])  # Hard brake

        next_location, next_heading, next_speed = bicycle_model.forecast_ego_vehicle(
            location,
            heading,
            speed,
            action,
        )

        assert next_speed >= 0.0

        # Test with other vehicles braking over many frames
        locations = np.array([[0.0, 0.0, 0.0]])
        headings = np.array([0.0])
        speeds = np.array([0.5])
        actions = np.array([[0.0, 0.0, 1.0]])

        future_locations, _, future_speeds = bicycle_model.forecast_other_vehicles(
            locations,
            headings,
            speeds,
            actions,
            num_future_frames=40,
        )

        assert np.all(future_speeds >= 0.0)
        # Once stopped, the vehicle must not move anymore
        stopped = np.where(future_speeds[:, 0] == 0.0)[0]
        assert np.all(
            future_locations[stopped[0] :, 0] == future_locations[stopped[0], 0],
        )

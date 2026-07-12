"""Kinematic bicycle model configuration shared by forecasting and state estimation."""

from lead.config.node import ConfigNode


class BicycleModelConfig(ConfigNode):
    """Kinematic bicycle model parameters of the ego and other vehicles."""

    # Frame rate used for the bicycle models in the autopilot
    bicycle_frame_rate: int = 20
    # Time step for the model (20 frames per second)
    time_step: float = 1.0 / 20.0
    # Distance from the rear axle to the front axle of the vehicle
    front_wheel_base: float = -0.090769015
    # Distance from the rear axle to the center of the rear wheels
    rear_wheel_base: float = 1.4178275
    # Gain factor for steering angle to wheel angle conversion
    steering_gain: float = 0.36848336
    # Deceleration rate when braking (m/s^2) of other vehicles
    brake_acceleration: float = -4.952399
    # Acceleration rate when throttling (m/s^2) of other vehicles
    throttle_acceleration: float = 0.5633837
    # Minimum throttle value that has an affect during forecasting the ego vehicle
    throttle_threshold_during_forecasting: float = 0.3

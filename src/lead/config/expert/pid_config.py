"""Controller (PID / longitudinal regression) configuration of the expert."""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.config.node import ConfigNode, overridable_property


class ExpertPidConfig(ConfigNode):
    """Lateral PID and longitudinal linear regression controller parameters."""

    @property
    def _ppm(self) -> int:
        """Route interpolation density used to convert meters to route points."""
        return self._root.expert.simulation.points_per_meter

    # --- Longitudinal Linear Regression Controller ---
    # Minimum threshold for target speed (< 1 km/h) for longitudinal linear regression controller
    longitudinal_linear_regression_minimum_target_speed: float = 0.278
    # Maximum acceleration rate (approximately 1.9 m/tick) for the longitudinal linear regression controller
    longitudinal_linear_regression_maximum_acceleration: float = 1.89
    # Maximum deceleration rate (approximately -4.82 m/tick) for the longitudinal linear regression controller
    longitudinal_linear_regression_maximum_deceleration: float = -4.82

    # --- Lateral PID Controller ---
    # The proportional gain for the lateral PID controller
    lateral_pid_kp: float = 3.118357247806046
    # The derivative gain for the lateral PID controller
    lateral_pid_kd: float = 1.3782508892109167
    # The integral gain for the lateral PID controller
    lateral_pid_ki: float = 0.6406067986034124
    # The scaling factor used in the calculation of the lookahead distance based on the current speed
    lateral_pid_speed_scale: float = 0.9755321901954155
    # The offset used in the calculation of the lookahead distance based on the current speed
    lateral_pid_speed_offset: float = 1.9152884533402488
    # The size of the sliding window used to store the error history for the lateral PID controller
    lateral_pid_window_size: int = 6

    @overridable_property
    def lateral_pid_minimum_lookahead_distance(self) -> float:
        """The minimum allowed lookahead distance for the lateral PID controller (route points)."""
        return 2.4 * self._ppm

    @overridable_property
    def lateral_pid_maximum_lookahead_distance(self) -> float:
        """The maximum allowed lookahead distance for the lateral PID controller (route points)."""
        return 10.5 * self._ppm

    # Linear regression parameters for longitudinal control as numpy array
    longitudinal_linear_regression_params: jt.Float[npt.NDArray, "7"] = np.array(
        [
            1.1990342347353184,
            -0.8057602384167799,
            1.710818710950062,
            0.921890257450335,
            1.556497522998393,
            -0.7013479734904027,
            1.031266635497984,
        ],
    )
    # Coefficients for polynomial equation estimating speed change with throttle input for ego model
    throttle_values: jt.Float[npt.NDArray, "8"] = np.array(
        [
            9.63873001e-01,
            4.37535692e-04,
            -3.80192912e-01,
            1.74950069e00,
            9.16787414e-02,
            -7.05461530e-02,
            -1.05996152e-03,
            6.71079346e-04,
        ],
    )
    # Coefficients for polynomial equation estimating speed change with brake input for the ego model
    brake_values: jt.Float[npt.NDArray, "7"] = np.array(
        [
            9.31711370e-03,
            8.20967431e-02,
            -2.83832427e-03,
            5.06587474e-05,
            -4.90357228e-07,
            2.44419284e-09,
            -4.91381935e-12,
        ],
    )

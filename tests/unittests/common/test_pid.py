import numpy as np

from lead.common.pid import PIDController


class TestPIDController:
    """Unit tests for PID controller."""

    def test_proportional_response(self):
        """Test proportional control reduces error linearly."""
        controller = PIDController(k_p=0.5, k_i=0.0, k_d=0.0, n=5)
        errors = [10.0, 5.0, 2.0, 1.0]
        outputs = [controller.step(e) for e in errors]
        expected = [e * 0.5 for e in errors]
        assert np.allclose(outputs, expected)

    def test_integral_accumulation(self):
        """Test integral term accumulates persistent error."""
        controller = PIDController(k_p=0.0, k_i=1.0, k_d=0.0, n=3)
        # Feed constant error - integral should grow
        outputs = [controller.step(1.0) for _ in range(5)]
        # Integral grows as window fills, then stabilizes at average
        assert outputs[-1] > outputs[0]
        assert outputs[-1] == 1.0  # Average of window [1, 1, 1]

    def test_derivative_detects_change(self):
        """Test derivative term responds to error rate of change."""
        controller = PIDController(k_p=0.0, k_i=0.0, k_d=1.0, n=5)
        controller.step(0.0)
        output = controller.step(5.0)  # Sudden error increase
        assert output == 5.0  # D-term = k_d * (5.0 - 0.0)

        output = controller.step(5.0)  # Error unchanged
        assert output == 0.0  # D-term = k_d * (5.0 - 5.0)

    def test_combined_pid_response(self):
        """Test full PID controller tracking setpoint."""
        controller = PIDController(k_p=0.8, k_i=0.2, k_d=0.1, n=10)
        # Simulate approaching target with decreasing error
        errors = [10.0, 8.0, 5.0, 3.0, 1.0, 0.5]
        outputs = [controller.step(e) for e in errors]

        # All outputs should be positive (correcting)
        assert all(o > 0 for o in outputs)
        # Output should decrease as error decreases
        assert outputs[-1] < outputs[0]

    def test_window_size_limits_history(self):
        """Test sliding window maintains fixed size."""
        n = 3
        controller = PIDController(k_p=0.0, k_i=1.0, k_d=0.0, n=n)
        # Fill beyond window size
        for i in range(10):
            controller.step(float(i))

        # Integral should only consider last n values
        output = controller.step(10.0)
        # Window: [8.0, 9.0, 10.0], average = 9.0
        assert output == 9.0

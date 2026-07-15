import pytest
import torch
import torch.testing as tt

from lead.config import TransfuserConfig, load_lead_config
from lead.policy.transfuser.decoder.planning_decoder import (
    decode_two_hot,
    encode_two_hot,
)


@pytest.fixture
def config():
    """Fixture providing the transfuser config section."""
    return load_lead_config().policy.transfuser


@pytest.fixture
def device():
    """Fixture providing CPU device."""
    return torch.device("cpu")


class TestPlanningDecoder:
    """Tests for planning decoder two-hot encoding/decoding functions."""

    def helper(
        self,
        config: TransfuserConfig,
        device: torch.device,
        input_speed: float,
        input_brake: bool,
        expected_encoded: list[float],
    ):
        """Helper function to test encode/decode round-trip.

        Args:
            config: Training configuration.
            device: Torch device.
            input_speed: Input speed value to encode.
            input_brake: Whether brake is applied.
            expected_encoded: Expected encoded two-hot distribution.
        """
        input_tensor = torch.tensor([input_speed])
        brake = torch.tensor([input_brake]).bool()

        # Encode and decode
        encoded = encode_two_hot(input_tensor, config.target_speed_classes, brake=brake)
        decoded = decode_two_hot(encoded, config.target_speed_classes, device)

        # Verify encoding
        tt.assert_close(
            encoded,
            torch.tensor(expected_encoded).unsqueeze(0),
            rtol=1e-4,
            atol=1e-4,
        )

        # Verify decoding
        if not input_brake:
            input_tensor = torch.clamp(
                input_tensor,
                0.0,
                config.target_speed_classes[-1],
            )
            tt.assert_close(decoded, input_tensor, rtol=1e-4, atol=1e-4)
        else:
            tt.assert_close(decoded, torch.tensor([0.0]), rtol=1e-4, atol=1e-4)

    def test_encode_decode_zero_speed_no_brake(self, config, device):
        """Test encoding/decoding speed 0.0 without brake."""
        self.helper(
            config,
            device,
            0.0,
            False,
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        )

    def test_encode_decode_mid_speed_no_brake(self, config, device):
        """Test encoding/decoding speed 2.0 without brake."""
        self.helper(
            config,
            device,
            2.0,
            False,
            [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        )

    def test_encode_decode_max_speed_no_brake(self, config, device):
        """Test encoding/decoding speed 20.0 (max) without brake."""
        self.helper(
            config,
            device,
            20.0,
            False,
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        )

    def test_encode_decode_max_speed_with_brake(self, config, device):
        """Test encoding/decoding speed 20.0 with brake applied."""
        self.helper(
            config,
            device,
            20.0,
            True,
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        )

    def test_encode_decode_above_max_speed_no_brake(self, config, device):
        """Test encoding/decoding speed 25.0 (above max) without brake."""
        self.helper(
            config,
            device,
            25.0,
            False,
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        )

    def test_encode_decode_above_max_speed_with_brake(self, config, device):
        """Test encoding/decoding speed 25.0 (above max) with brake applied."""
        self.helper(
            config,
            device,
            25.0,
            True,
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        )

    def test_two_hot_interpolation(self, config, device):
        """Test that two-hot encoding correctly interpolates between bins."""
        speed = torch.tensor([6.0])  # Should interpolate between bins
        brake = torch.tensor([False]).bool()

        encoded = encode_two_hot(speed, config.target_speed_classes, brake=brake)

        # Check that exactly two adjacent bins are non-zero
        non_zero = (encoded > 0).sum()
        assert non_zero == 2, f"Expected 2 non-zero bins, got {non_zero}"

        # Check that weights sum to 1
        assert torch.allclose(encoded.sum(), torch.tensor(1.0)), (
            "Encoded weights should sum to 1"
        )

    def test_brake_overrides_speed(self, config, device):
        """Test that brake flag overrides any speed value."""
        for speed_val in [0.0, 5.0, 10.0, 20.0]:
            speed = torch.tensor([speed_val])
            brake = torch.tensor([True]).bool()

            encoded = encode_two_hot(speed, config.target_speed_classes, brake=brake)

            # When brake is True, should encode as first bin only
            expected = torch.zeros(len(config.target_speed_classes))
            expected[0] = 1.0

            tt.assert_close(encoded[0], expected, rtol=1e-4, atol=1e-4)

    def test_decode_round_trip(self, config, device):
        """Test that decode(encode(x)) ≈ x for various speeds."""
        test_speeds = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]

        for speed_val in test_speeds:
            speed = torch.tensor([speed_val])
            brake = torch.tensor([False]).bool()

            encoded = encode_two_hot(speed, config.target_speed_classes, brake=brake)
            decoded = decode_two_hot(encoded, config.target_speed_classes, device)

            tt.assert_close(decoded, speed, rtol=1e-4, atol=1e-4)

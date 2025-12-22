import unittest

import torch
import torch.testing as tt
from config_training import TrainingConfig
from model.decoders.planning_decoder import decode_two_hot, encode_two_hot


class TestPlanningDecoder(unittest.TestCase):
    config = TrainingConfig()
    device = torch.device("cpu")

    def helper(self, input_speed: float, input_brake: bool, expected_encoded: list[float]):
        input_tensor = torch.tensor([input_speed])
        brake = torch.tensor([input_brake]).bool()

        # Use new API with class_values and optional brake_mask
        encoded = encode_two_hot(input_tensor, self.config.target_speed_classes, brake=brake)
        decoded = decode_two_hot(encoded, self.config.target_speed_classes, self.device)

        tt.assert_close(encoded, torch.tensor(expected_encoded).unsqueeze(0), rtol=1e-4, atol=1e-4)
        if not input_brake:
            input_tensor = torch.clamp(input_tensor, 0.0, self.config.target_speed_classes[-1])
            tt.assert_close(decoded, input_tensor, rtol=1e-4, atol=1e-4)
        else:
            tt.assert_close(decoded, torch.tensor([0.0]), rtol=1e-4, atol=1e-4)

    def test_encode_decode_target_speed(self):
        self.helper(0.0, False, [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        self.helper(2.0, False, [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        self.helper(20.0, False, [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000])
        self.helper(20.0, True, [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        self.helper(25.0, False, [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000])
        self.helper(25.0, True, [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])


if __name__ == "__main__":
    unittest.main()

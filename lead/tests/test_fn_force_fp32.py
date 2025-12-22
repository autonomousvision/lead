import unittest

import torch
import torch.testing as tt
from model.fn import force_fp32


class TestForceFP32Decorator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with different tensor types."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create test tensors of different types
        self.fp16_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device=self.device)
        self.fp32_tensor = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device=self.device)
        self.int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32, device=self.device)
        self.non_tensor = "hello"

    def test_force_fp32_all_args(self):
        """Test force_fp32 decorator without apply_to parameter (converts all tensor args)."""

        @force_fp32()
        def test_function(x, y, z):
            # Check that tensors are converted to fp32
            if isinstance(x, torch.Tensor):
                self.assertEqual(x.dtype, torch.float32)
            if isinstance(y, torch.Tensor):
                self.assertEqual(y.dtype, torch.float32)
            # z should remain unchanged if it's not a tensor
            return x, y, z

        # Test with mixed tensor types
        result_x, result_y, result_z = test_function(self.fp16_tensor, self.fp32_tensor, self.non_tensor)

        # Check that fp16 tensor was converted to fp32
        self.assertEqual(result_x.dtype, torch.float32)
        tt.assert_close(result_x, self.fp16_tensor.float())

        # Check that fp32 tensor remains fp32
        self.assertEqual(result_y.dtype, torch.float32)
        tt.assert_close(result_y, self.fp32_tensor)

        # Check that non-tensor argument is unchanged
        self.assertEqual(result_z, self.non_tensor)

    def test_force_fp32_specific_args(self):
        """Test force_fp32 decorator with apply_to parameter (converts only specified args)."""

        @force_fp32(apply_to=("x", "z"))
        def test_function(x, y, z):
            # Only x and z should be converted to fp32
            if isinstance(x, torch.Tensor):
                self.assertEqual(x.dtype, torch.float32)
            if isinstance(z, torch.Tensor):
                self.assertEqual(z.dtype, torch.float32)
            # y should remain unchanged
            return x, y, z

        # Test with fp16 tensors
        fp16_tensor_2 = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float16, device=self.device)
        result_x, result_y, result_z = test_function(self.fp16_tensor, self.fp16_tensor, fp16_tensor_2)

        # x should be converted
        self.assertEqual(result_x.dtype, torch.float32)
        tt.assert_close(result_x, self.fp16_tensor.float())

        # y should remain fp16
        self.assertEqual(result_y.dtype, torch.float16)
        tt.assert_close(result_y, self.fp16_tensor)

        # z should be converted
        self.assertEqual(result_z.dtype, torch.float32)
        tt.assert_close(result_z, fp16_tensor_2.float())

    def test_force_fp32_kwargs(self):
        """Test force_fp32 decorator with keyword arguments."""

        @force_fp32(apply_to=("x", "y"))
        def test_function(x, y=None, z=None):
            if isinstance(x, torch.Tensor):
                self.assertEqual(x.dtype, torch.float32)
            if isinstance(y, torch.Tensor):
                self.assertEqual(y.dtype, torch.float32)
            return x, y, z

        # Test with keyword arguments
        result_x, result_y, result_z = test_function(x=self.fp16_tensor, y=self.fp16_tensor, z=self.fp16_tensor)

        # x and y should be converted
        self.assertEqual(result_x.dtype, torch.float32)
        self.assertEqual(result_y.dtype, torch.float32)

        # z should remain fp16 since it's not in apply_to
        self.assertEqual(result_z.dtype, torch.float16)

    def test_force_fp32_non_tensor_args(self):
        """Test force_fp32 decorator with non-tensor arguments."""

        @force_fp32()
        def test_function(x, y, z):
            return x, y, z

        # Test with non-tensor arguments
        result_x, result_y, result_z = test_function(self.non_tensor, 42, [1, 2, 3])

        # All should remain unchanged
        self.assertEqual(result_x, self.non_tensor)
        self.assertEqual(result_y, 42)
        self.assertEqual(result_z, [1, 2, 3])

    def test_force_fp32_mixed_args(self):
        """Test force_fp32 decorator with mixed tensor and non-tensor arguments."""

        @force_fp32(apply_to=("tensor_arg",))
        def test_function(tensor_arg, int_arg, string_arg):
            if isinstance(tensor_arg, torch.Tensor):
                self.assertEqual(tensor_arg.dtype, torch.float32)
            return tensor_arg, int_arg, string_arg

        result_tensor, result_int, result_string = test_function(tensor_arg=self.fp16_tensor, int_arg=42, string_arg="test")

        # Only tensor should be converted
        self.assertEqual(result_tensor.dtype, torch.float32)
        tt.assert_close(result_tensor, self.fp16_tensor.float())

        # Others should remain unchanged
        self.assertEqual(result_int, 42)
        self.assertEqual(result_string, "test")

    def test_force_fp32_autocast_disabled(self):
        """Test that force_fp32 decorator disables autocast."""

        @force_fp32()
        def test_function(x):
            # Inside the function, autocast should be disabled
            # We can test this by checking if operations maintain fp32 precision
            result = torch.softmax(x, dim=-1)
            self.assertEqual(result.dtype, torch.float32)
            return result

        # Enable autocast and test
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
            # Even though autocast is enabled, the function should run in fp32
            result = test_function(self.fp16_tensor)
            self.assertEqual(result.dtype, torch.float32)

    def test_force_fp32_return_values(self):
        """Test that force_fp32 decorator doesn't affect return values."""

        @force_fp32()
        def test_function(x):
            # The decorator should not change the return type
            return x  # This should still be fp32 since input was converted

        result = test_function(self.fp16_tensor)

        # Result should be fp32 since input was converted to fp32
        self.assertEqual(result.dtype, torch.float32)
        tt.assert_close(result, self.fp16_tensor.float())

    def test_force_fp32_mathematical_operations(self):
        """Test force_fp32 decorator with mathematical operations that benefit from fp32 precision."""

        @force_fp32()
        def softmax_function(logits):
            return torch.softmax(logits, dim=-1)

        # Create logits that might have precision issues in fp16
        logits_fp16 = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float16, device=self.device)

        result = softmax_function(logits_fp16)

        # Result should be computed in fp32 precision
        expected = torch.softmax(logits_fp16.float(), dim=-1)
        tt.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_force_fp32_preserves_function_signature(self):
        """Test that force_fp32 decorator preserves function metadata."""

        @force_fp32()
        def documented_function(x, y):
            """This is a test function with documentation."""
            return x + y

        # Check that function name and docstring are preserved
        self.assertEqual(documented_function.__name__, "documented_function")
        self.assertEqual(documented_function.__doc__, "This is a test function with documentation.")


if __name__ == "__main__":
    unittest.main()

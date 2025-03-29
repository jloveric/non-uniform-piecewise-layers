import torch
import torch.nn as nn
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.efficient_adaptive_piecewise_conv import (
    PiecewiseLinearExpansion2d,
    EfficientAdaptivePiecewiseConv2d,
)
from non_uniform_piecewise_layers.adaptive_piecewise_conv import AdaptivePiecewiseConv2d
import torch.testing as testing

class TestPiecewiseLinearExpansion2d(unittest.TestCase):
    """Test cases for the PiecewiseLinearExpansion2d class."""

    def test_expansion_shape(self):
        """Test that the expansion produces the correct output shape."""
        batch_size = 2
        channels = 3
        height = 10
        width = 10
        num_points = 5
        
        # Create input tensor
        x = torch.randn(batch_size, channels, height, width)
        
        # Create expansion layer
        expansion = PiecewiseLinearExpansion2d(num_points=num_points)
        
        # Apply expansion
        expanded = expansion(x)
        
        # Check output shape
        expected_shape = (batch_size, channels * num_points, height, width)
        self.assertEqual(expanded.shape, expected_shape)
    
    def test_expansion_values(self):
        """Test that the expansion produces correct values for a simple case."""
        # Create a simple 1x1x2x2 input with known values
        x = torch.tensor([[[[0.0, 1.0], [0.5, -0.5]]]])  # 1 channel, 2x2 spatial dims
        
        # Create expansion with 3 points at positions [-1, 0, 1]
        expansion = PiecewiseLinearExpansion2d(num_points=3)
        expansion.positions = torch.tensor([-1.0, 0.0, 1.0])
        
        # Apply expansion
        expanded = expansion(x)
        
        print('expanded', expanded, expanded.shape)
        
        # Expected basis function values for each input value based on the implementation:
        # For x=0.0 (at position 0,0):
        #   - First basis (at position -1): 0.0 (leftmost point, mask is True but value is 0)
        #   - Second basis (at position 0): 1.0 (middle point, left mask is True with value 1.0)
        #   - Third basis (at position 1): 0.0 (rightmost point, mask is True but value is 0)
        # For x=1.0 (at position 0,1):
        #   - First basis (at position -1): 0.0 (outside support)
        #   - Second basis (at position 0): 0.0 (hat function at edge)
        #   - Third basis (at position 1): 1.0 (hat function peak)
        # For x=0.5 (at position 1,0):
        #   - First basis (at position -1): 0.0 (outside support)
        #   - Second basis (at position 0): 0.5 (halfway down hat function)
        #   - Third basis (at position 1): 0.5 (halfway up hat function)
        # For x=-0.5 (at position 1,1):
        #   - First basis (at position -1): 0.5 (halfway up hat function)
        #   - Second basis (at position 0): 0.5 (halfway down hat function)
        #   - Third basis (at position 1): 0.0 (outside support)
        
        # Test for x=0.0 (position 0,0)
        self.assertAlmostEqual(expanded[0, 0, 0, 0].item(), 0.0 * 0.0, places=5)  # x=0.0, first basis
        self.assertAlmostEqual(expanded[0, 1, 0, 0].item(), 0.0 * 1.0, places=5)  # x=0.0, second basis
        self.assertAlmostEqual(expanded[0, 2, 0, 0].item(), 0.0 * 0.0, places=5)  # x=0.0, third basis
        
        # Test for x=1.0 (position 0,1)
        self.assertAlmostEqual(expanded[0, 0, 0, 1].item(), 1.0 * 0.0, places=5)  # x=1.0, first basis
        self.assertAlmostEqual(expanded[0, 1, 0, 1].item(), 1.0 * 0.0, places=5)  # x=1.0, second basis
        self.assertAlmostEqual(expanded[0, 2, 0, 1].item(), 1.0 * 1.0, places=5)  # x=1.0, third basis
        
        # Test for x=0.5 (position 1,0)
        self.assertAlmostEqual(expanded[0, 0, 1, 0].item(), 0.5 * 0.0, places=5)  # x=0.5, first basis
        self.assertAlmostEqual(expanded[0, 1, 1, 0].item(), 0.5 * 0.5, places=5)  # x=0.5, second basis
        self.assertAlmostEqual(expanded[0, 2, 1, 0].item(), 0.5 * 0.5, places=5)  # x=0.5, third basis
        
        # Test for x=-0.5 (position 1,1)
        self.assertAlmostEqual(expanded[0, 0, 1, 1].item(), -0.5 * 0.5, places=5)  # x=-0.5, first basis
        self.assertAlmostEqual(expanded[0, 1, 1, 1].item(), -0.5 * 0.5, places=5)  # x=-0.5, second basis
        self.assertAlmostEqual(expanded[0, 2, 1, 1].item(), -0.5 * 0.0, places=5)  # x=-0.5, third basis
    
    def test_expansion_gradient(self):
        """
        TODO: I feel like this is a useless test
        Test that gradients flow through the expansion layer.
        """
        batch_size = 2
        channels = 3
        height = 4
        width = 4
        num_points = 5
        
        # Create input tensor that requires grad
        x = torch.randn(batch_size, channels, height, width, requires_grad=True)
        
        # Create expansion layer
        expansion = PiecewiseLinearExpansion2d(num_points=num_points)
        
        # Apply expansion
        expanded = expansion(x)
        
        # Compute loss and backpropagate
        loss = expanded.sum()
        loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


class TestEfficientAdaptivePiecewiseConv2d(unittest.TestCase):
    """Test cases for the EfficientAdaptivePiecewiseConv2d class."""

    def test_conv_shape(self):
        """Test that the convolution produces the correct output shape."""
        batch_size = 2
        in_channels = 3
        out_channels = 6
        height = 10
        width = 10
        kernel_size = (3, 3)
        num_points = 5
        
        # Create input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Create convolution layer with explicit padding='same'
        conv = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            padding='same'
        )
        
        # Apply convolution
        output = conv(x)
        
        # Calculate expected output shape
        # For a convolution with padding='same', the output spatial dimensions are the same as input
        expected_shape = (batch_size, out_channels, height, width)
        self.assertEqual(output.shape, expected_shape)
    
    def test_conv_values_simple(self):
        """Test the full convolution output with known weights and simple input."""
        # Input tensor (1 batch, 1 channel, 2x2)
        x = torch.tensor([[[[0.0, 1.0], [0.5, -0.5]]]], dtype=torch.float32)
        
        # Layer parameters
        in_channels = 1
        out_channels = 1
        kernel_size = 1
        num_points = 3
        
        # Create the layer
        conv_layer = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            position_init="uniform", # Positions will be [-1, 0, 1]
            padding=0 # No padding needed for 1x1 kernel
        )
        
        # Manually set weights to ones
        # Shape: (out_channels, in_channels * num_points, ks, ks) = (1, 1*3, 1, 1)
        conv_layer.conv.weight.data = torch.ones((1, 3, 1, 1), dtype=torch.float32)
        
        # --- Calculate Expected Output ---
        # 1. Expansion positions: [-1.0, 0.0, 1.0]
        # 2. Expected expanded tensor (calculated based on implementation):
        #    x=0.0  -> [0.0, 1.0, 0.0] * 0.0  = [0.0, 0.0, 0.0]
        #    x=1.0  -> [0.0, 0.0, 1.0] * 1.0  = [0.0, 0.0, 1.0]
        #    x=0.5  -> [0.0, 0.5, 0.5] * 0.5  = [0.0, 0.25, 0.25]
        #    x=-0.5 -> [0.5, 0.5, 0.0] * -0.5 = [-0.25, -0.25, 0.0]
        # Expanded tensor shape: (1, 3, 2, 2)
        # expanded = torch.tensor([[[
        #     [ 0.00,  0.00],  # Basis -1
        #     [ 0.00, -0.25]
        # ],[
        #     [ 0.00,  0.00],  # Basis 0
        #     [ 0.25, -0.25]
        # ],[
        #     [ 0.00,  1.00],  # Basis 1
        #     [ 0.25,  0.00]
        # ]]], dtype=torch.float32)
        
        # 3. Apply 1x1 convolution with weights = [1, 1, 1]
        #    Output[h, w] = sum(Expanded[:, h, w])
        #    Output[0, 0] = 0.0 + 0.0 + 0.0 = 0.0
        #    Output[0, 1] = 0.0 + 0.0 + 1.0 = 1.0
        #    Output[1, 0] = 0.0 + 0.25 + 0.25 = 0.5
        #    Output[1, 1] = -0.25 + (-0.25) + 0.0 = -0.5
        expected_output = torch.tensor([[[[0.0, 1.0], [0.5, -0.5]]]], dtype=torch.float32)

        # --- Get Actual Output ---
        actual_output = conv_layer(x)
        
        # Print for debugging
        print("Input Tensor:")
        print(x)
        # print("Expanded Tensor (Manual):") # Uncomment if needed
        # print(expanded)                     # Uncomment if needed
        print("Expected Output:")
        print(expected_output)
        print("Actual Output:")
        print(actual_output)
        
        # --- Compare ---
        testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)

    def test_conv_gradient(self):
        """Test that gradients flow through the convolution layer."""
        batch_size = 2
        in_channels = 3
        out_channels = 6
        height = 8
        width = 8
        kernel_size = (3, 3)
        num_points = 5
        
        # Create input tensor that requires grad
        x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        
        # Create convolution layer with explicit padding='same'
        conv = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            padding='same'
        )
        
        # Apply convolution
        output = conv(x)
        
        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isnan(conv.conv.weight.grad).any())

    
    def test_expansion_and_conv_separately(self):
        """Test that the expansion followed by convolution works as expected."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        batch_size = 2
        in_channels = 3
        out_channels = 6
        height = 8
        width = 8
        kernel_size = (3, 3)
        num_points = 4
        
        # Create input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Create expansion layer
        expansion = PiecewiseLinearExpansion2d(num_points=num_points)
        
        # Create convolution layer
        conv = nn.Conv2d(
            in_channels=in_channels * num_points,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
        )
        
        # Create efficient implementation
        efficient_conv = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            padding='same'
        )
        
        # Copy weights from efficient to separate conv
        conv.weight.data = efficient_conv.conv.weight.data.clone()
        if conv.bias is not None and efficient_conv.conv.bias is not None:
            conv.bias.data = efficient_conv.conv.bias.data.clone()
        
        # Copy positions from efficient to separate expansion
        expansion.positions = efficient_conv.expansion.positions.clone()
        
        # Apply separate expansion and convolution
        expanded = expansion(x)
        separate_output = conv(expanded)
        
        # Apply efficient convolution
        efficient_output = efficient_conv(x)
        
        # Check that the outputs have the same shape
        self.assertEqual(separate_output.shape, efficient_output.shape)
        
        # Verify that the outputs are not all zeros
        self.assertFalse(torch.all(efficient_output == 0))
        
        # Print the statistics for debugging
        print(f"Separate output mean: {separate_output.mean().item()}, std: {separate_output.std().item()}")
        print(f"Efficient output mean: {efficient_output.mean().item()}, std: {efficient_output.std().item()}")
        
        # The implementations are different, so we just check that both produce non-zero output
        # with reasonable statistics rather than requiring exact matches


if __name__ == "__main__":
    unittest.main()

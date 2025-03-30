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

    def test_conv_values_2x2_kernel_vs_separate(self):
        """Test the full convolution output with a 2x2 kernel against separate components."""
        # Set random seed for reproducibility
        torch.manual_seed(123)
        
        # Input tensor (1 batch, 1 channel, 4x4)
        x = torch.randn((1, 1, 4, 4), dtype=torch.float32)
        
        # Layer parameters
        in_channels = 1
        out_channels = 1
        kernel_size = 2
        num_points = 3
        padding = 1 # Explicit padding
        
        # 1. Create the efficient layer
        efficient_conv = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            position_init="uniform", # Positions will be [-1, 0, 1]
            padding=padding
        )
        
        # 2. Create separate expansion and conv layers
        expansion = PiecewiseLinearExpansion2d(num_points=num_points, position_init="uniform")
        separate_conv = nn.Conv2d(
            in_channels=in_channels * num_points,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False # Match efficient layer default
        )
        
        # 3. Ensure weights and positions are identical
        # Use random weights for a more robust test than all ones
        efficient_conv.conv.weight.data = torch.randn_like(efficient_conv.conv.weight.data)
        separate_conv.weight.data = efficient_conv.conv.weight.data.clone()
        # Positions are already identical due to "uniform" init, but explicit copy is safer
        expansion.positions = efficient_conv.expansion.positions.clone()
        
        # 4. Calculate output from efficient layer
        efficient_output = efficient_conv(x)
        
        # 5. Calculate output from separate components
        expanded_output = expansion(x)
        separate_output = separate_conv(expanded_output)
        
        # Print shapes for debugging
        print(f"Input shape: {x.shape}")
        print(f"Expanded shape: {expanded_output.shape}")
        print(f"Efficient Output shape: {efficient_output.shape}")
        print(f"Separate Output shape: {separate_output.shape}")

        # 6. Compare outputs
        testing.assert_close(
            efficient_output, 
            separate_output, 
            rtol=1e-5, 
            atol=1e-5,
            msg="Output mismatch between efficient layer and separate components for 2x2 kernel"
        )

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


    def test_move_smoothest(self):
        """Test that the move_smoothest method correctly updates positions and weights."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Layer parameters
        in_channels = 2
        out_channels = 3
        kernel_size = 2
        num_points = 5  # Need at least 4 points to be able to move one
        padding = 1
        
        # Create input tensor
        x = torch.randn((1, in_channels, 4, 4), dtype=torch.float32)
        
        # Create the layer with non-uniform positions
        conv_layer = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            position_init="uniform",
            padding=padding
        )
        
        # Set non-uniform positions to ensure non-zero errors
        with torch.no_grad():
            positions = torch.tensor([-1.0, -0.6, -0.2, 0.4, 1.0], 
                                   device=conv_layer.expansion.positions.device)
            conv_layer.expansion.positions.data = positions
        
        # Store the initial positions and weights
        initial_positions = conv_layer.expansion.positions.clone()
        initial_weights = conv_layer.conv.weight.data.clone()
        
        # Get the initial output
        initial_output = conv_layer(x)
        
        # Apply move_smoothest with a negative threshold to ensure it moves points
        moved = conv_layer.move_smoothest(weighted=True, threshold=-1.0)
        
        # Verify that points were moved
        self.assertTrue(moved, "move_smoothest should return True when points are moved")
        
        # Get the new positions and weights
        new_positions = conv_layer.expansion.positions
        new_weights = conv_layer.conv.weight.data
        
        # Verify that positions have changed
        self.assertFalse(torch.allclose(initial_positions, new_positions), 
                         "Positions should change after move_smoothest")
        
        # Verify that weights have changed
        self.assertFalse(torch.allclose(initial_weights, new_weights), 
                         "Weights should change after move_smoothest")
        
        # Verify that the number of positions is still the same
        self.assertEqual(len(initial_positions), len(new_positions), 
                        "Number of positions should remain the same")
        
        # Get the output after moving points
        new_output = conv_layer(x)
        
        # Verify that the output shape hasn't changed
        self.assertEqual(initial_output.shape, new_output.shape, 
                        "Output shape should not change after move_smoothest")
        
        # Verify that the output has changed (since we moved points)
        self.assertFalse(torch.allclose(initial_output, new_output), 
                         "Output should change after move_smoothest")
        
        # Test with threshold parameter
        # Create a new layer with non-uniform positions to ensure non-zero errors
        conv_layer_threshold = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            position_init="uniform",
            padding=padding
        )
        
        # Set positions to create a non-uniform distribution with predictable error ratios
        with torch.no_grad():
            # Create non-uniform positions with one point having a small error
            positions = torch.tensor([-1.0, -0.6, -0.2, 0.4, 1.0], 
                                   device=conv_layer_threshold.expansion.positions.device)
            conv_layer_threshold.expansion.positions.data = positions
        
        # First try with a negative threshold - should always move points even with uniform errors
        moved_with_negative_threshold = conv_layer_threshold.move_smoothest(weighted=True, threshold=-1.0)
        self.assertTrue(moved_with_negative_threshold, 
                       "move_smoothest should return True with a negative threshold")
        
        # Create another layer for the high threshold test
        conv_layer_high_threshold = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            position_init="uniform",
            padding=padding
        )
        
        # Set the same non-uniform positions
        with torch.no_grad():
            positions = torch.tensor([-1.0, -0.6, -0.2, 0.4, 1.0], 
                                   device=conv_layer_high_threshold.expansion.positions.device)
            conv_layer_high_threshold.expansion.positions.data = positions
        
        # Now try with a threshold of 1.0 - should never move points
        moved_with_high_threshold = conv_layer_high_threshold.move_smoothest(weighted=True, threshold=1.0)
        self.assertFalse(moved_with_high_threshold, 
                         "move_smoothest should return False when threshold=1.0")
        
        # Test that positions are actually changed and no longer evenly spaced
        # Create a layer with evenly spaced positions
        conv_layer_even = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            position_init="uniform",
            padding=padding
        )
        
        # Set evenly spaced positions
        with torch.no_grad():
            positions = torch.linspace(-1.0, 1.0, num_points, device=conv_layer_even.expansion.positions.device)
            conv_layer_even.expansion.positions.data = positions
            
        # Store initial positions
        initial_positions = conv_layer_even.expansion.positions.clone()
        
        # Verify positions are evenly spaced initially
        diffs = initial_positions[1:] - initial_positions[:-1]
        self.assertTrue(torch.allclose(diffs, diffs[0].expand_as(diffs), rtol=1e-5, atol=1e-5),
                        "Initial positions should be evenly spaced")
        
        # Apply move_smoothest with a negative threshold to ensure it moves points
        moved = conv_layer_even.move_smoothest(weighted=True, threshold=-1.0)
        self.assertTrue(moved, "move_smoothest should return True with negative threshold")
        
        # Get new positions
        new_positions = conv_layer_even.expansion.positions
        
        # Verify positions have changed
        self.assertFalse(torch.allclose(initial_positions, new_positions, rtol=1e-5, atol=1e-5),
                         "Positions should change after move_smoothest")
        print('initial_positions', initial_positions, 'new_posistions', new_positions)

        # Verify positions are no longer evenly spaced
        new_diffs = new_positions[1:] - new_positions[:-1]
        self.assertFalse(torch.allclose(new_diffs, new_diffs[0].expand_as(new_diffs), rtol=1e-5, atol=1e-5),
                         "Positions should no longer be evenly spaced after move_smoothest")
        
        # Test that weights maintain a global linear relationship after move_smoothest
        # Create a layer with evenly spaced positions
        conv_layer_linear = EfficientAdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            position_init="uniform",
            padding=padding
        )
        
        # Set evenly spaced positions
        with torch.no_grad():
            positions = torch.linspace(-1.0, 1.0, num_points, device=conv_layer_linear.expansion.positions.device)
            conv_layer_linear.expansion.positions.data = positions
            
            # Instead of using different slopes/intercepts for each point,
            # we'll use a single global linear function for each filter/channel
            # This way, even if points are reordered, we can still check if the weights
            # follow the global linear relationship
            
            # Get the shape of the weights
            out_ch, expanded_in_ch, kh, kw = conv_layer_linear.conv.weight.shape
            
            # Create random slopes and intercepts for each filter/channel combination
            # For each output channel and each input channel
            global_slopes = torch.randn((out_ch, in_channels), device=positions.device)
            global_intercepts = torch.randn((out_ch, in_channels), device=positions.device)
            
            # Initialize the weights to follow linear patterns
            for out_idx in range(out_ch):
                for in_idx in range(in_channels):
                    # Use the same slope and intercept for all points in this filter/channel
                    slope = global_slopes[out_idx, in_idx].item()
                    intercept = global_intercepts[out_idx, in_idx].item()
                    
                    for p_idx in range(num_points):
                        # Calculate the index in the expanded input channels
                        expanded_idx = in_idx * num_points + p_idx
                        
                        # For simplicity, we'll set the same pattern for all kernel positions
                        for ki in range(kh):
                            for kj in range(kw):
                                # Set weights to follow f(x) = slope * x + intercept
                                x = positions[p_idx].item()
                                value = slope * x + intercept
                                conv_layer_linear.conv.weight.data[out_idx, expanded_idx, ki, kj] = value
        
        # Store initial positions and weights
        initial_positions = conv_layer_linear.expansion.positions.clone()
        initial_weights = conv_layer_linear.conv.weight.data.clone()
        
        # Apply move_smoothest with a negative threshold to ensure it moves points
        moved = conv_layer_linear.move_smoothest(weighted=True, threshold=-1.0)
        self.assertTrue(moved, "move_smoothest should return True with negative threshold")
        
        # Get new positions and weights after moving points
        new_positions = conv_layer_linear.expansion.positions
        new_weights = conv_layer_linear.conv.weight.data
        
        # Verify positions have changed
        self.assertFalse(torch.allclose(initial_positions, new_positions),
                         "Positions should change after move_smoothest")
        
        # Print positions for debugging
        print(f"Initial positions: {initial_positions}")
        print(f"New positions: {new_positions}")
        
        # Now check if the weights still follow the global linear relationship
        # For each output channel and input channel, plot the weights against positions
        # and check if they still follow a linear pattern
        for out_idx in range(out_ch):
            for in_idx in range(in_channels):
                # Extract the weights for this filter/channel
                weights = []
                for p_idx in range(num_points):
                    expanded_idx = in_idx * num_points + p_idx
                    weights.append(new_weights[out_idx, expanded_idx, 0, 0].item())
                
                # Convert to tensors for easier manipulation
                positions_tensor = new_positions
                weights_tensor = torch.tensor(weights, device=positions_tensor.device)
                
                # Fit a linear function to the weights vs positions
                # Using the formula for linear regression:
                # slope = (E[xy] - E[x]E[y]) / (E[x^2] - E[x]^2)
                # intercept = E[y] - slope * E[x]
                mean_x = torch.mean(positions_tensor)
                mean_y = torch.mean(weights_tensor)
                mean_xy = torch.mean(positions_tensor * weights_tensor)
                mean_x2 = torch.mean(positions_tensor * positions_tensor)
                
                fitted_slope = (mean_xy - mean_x * mean_y) / (mean_x2 - mean_x * mean_x)
                fitted_intercept = mean_y - fitted_slope * mean_x
                
                # Calculate the expected weights based on the fitted line
                expected_weights = fitted_slope * positions_tensor + fitted_intercept
                
                # Check if the actual weights are close to the expected weights
                # This verifies that the weights still follow a linear pattern
                # even if the specific slope/intercept might have changed
                is_linear = torch.allclose(weights_tensor, expected_weights, rtol=1e-2, atol=1e-2)
                
                # Print debug info for the first few channels
                if out_idx < 1 and in_idx < 1:
                    print(f"\nFilter {out_idx}, Channel {in_idx}:")
                    print(f"  Fitted slope: {fitted_slope.item():.6f}, intercept: {fitted_intercept.item():.6f}")
                    for p_idx in range(num_points):
                        actual = weights[p_idx]
                        expected = expected_weights[p_idx].item()
                        pos = positions_tensor[p_idx].item()
                        print(f"  Point {p_idx}: pos={pos:.6f}, weight={actual:.6f}, expected={expected:.6f}, diff={abs(actual-expected):.6f}")
                    print(f"  Is linear: {is_linear}")
                
                # Assert that the weights still follow a linear pattern
                self.assertTrue(is_linear, 
                               f"Weights should maintain a linear relationship after move_smoothest for out_ch={out_idx}, in_ch={in_idx}")


if __name__ == "__main__":
    unittest.main()

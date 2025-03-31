import torch
import torch.nn as nn
import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.custom_positions_conv import CustomPositionsPiecewiseConv2d
import torch.testing as testing

def test_conv_shape():
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
    conv = CustomPositionsPiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points,
        padding=1  # Padding for 'same' with 3x3 kernel
    )
    
    # Apply convolution
    output = conv(x)
    
    # Calculate expected output shape
    # For a convolution with padding=1 and 3x3 kernel, the output spatial dimensions are the same as input
    expected_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_shape

def test_conv_values_simple():
    """Test the full convolution output with known weights and simple input."""
    # Input tensor (1 batch, 1 channel, 2x2)
    x = torch.tensor([[[[0.0, 1.0], [0.5, -0.5]]]], dtype=torch.float32)
    
    # Layer parameters
    in_channels = 1
    out_channels = 1
    kernel_size = 1
    num_points = 3
    
    # Create the layer
    conv_layer = CustomPositionsPiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points,
        position_init="uniform",  # Positions will be [-1, 0, 1]
        padding=0  # No padding needed for 1x1 kernel
    )
    
    # Print positions to verify they are [-1, 0, 1]
    print("Positions:", conv_layer.positions[0, 0, 0, 0])
    
    # Manually set weights to ones
    # Shape: (out_channels, in_channels, num_points, ks, ks) = (1, 1, 3, 1, 1)
    conv_layer.weights.data = torch.ones((1, 1, 3, 1, 1), dtype=torch.float32)
    
    # Let's also check the interpolation constants
    print("Variable coefficients:", conv_layer.variable_coefficients[0, 0, 0, 0])
    print("Constant terms:", conv_layer.constant_terms[0, 0, 0, 0])
    
    # Apply the convolution
    output = conv_layer(x)
    
    # Print the actual output for debugging
    print("Actual output:", output)
    
    # Expected output based on the analytical solution
    # For input tensor [[[[0.0, 1.0], [0.5, -0.5]]]]
    # With positions [-1, 0, 1] and all weights = 1.0:
    # Output[0,0] = 0.0 (input = 0.0)
    # Output[0,1] = 1.0 (input = 1.0)
    # Output[1,0] = 0.5 (input = 0.5)
    # Output[1,1] = -0.5 (input = -0.5)
    expected_output = torch.tensor([[[[0.0, 1.0], [0.5, -0.5]]]], dtype=torch.float32)
    print("Expected output:", expected_output)
    
    # Check that the output matches the expected values
    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)

def test_position_initialization():
    """Test that positions are correctly initialized."""
    # Create layer with uniform positions
    conv_uniform = CustomPositionsPiecewiseConv2d(
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        num_points=5,
        position_init="uniform",
        position_range=(-1, 1)
    )
    
    # Check that positions have the correct shape
    expected_shape = (2, 2, 2, 2, 5)  # (out_ch, in_ch, kh, kw, num_points)
    assert conv_uniform.positions.shape == expected_shape
    
    # Check that uniform positions are the same for all kernel elements
    positions_uniform = conv_uniform.positions
    first_positions = positions_uniform[0, 0, 0, 0]
    for out_ch in range(2):
        for in_ch in range(2):
            for kh in range(2):
                for kw in range(2):
                    torch.testing.assert_close(
                        positions_uniform[out_ch, in_ch, kh, kw],
                        first_positions,
                        rtol=1e-5, atol=1e-5
                    )
    
    # Create layer with random positions
    conv_random = CustomPositionsPiecewiseConv2d(
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        num_points=5,
        position_init="random",
        position_range=(-1, 1)
    )
    
    # Check that positions have the correct shape
    assert conv_random.positions.shape == expected_shape
    
    # Check that random positions are different for different kernel elements
    positions_random = conv_random.positions
    different_positions = False
    
    for out_ch in range(2):
        for in_ch in range(2):
            for kh in range(2):
                for kw in range(2):
                    if out_ch == 0 and in_ch == 0 and kh == 0 and kw == 0:
                        continue
                    
                    if not torch.allclose(
                        positions_random[out_ch, in_ch, kh, kw],
                        positions_random[0, 0, 0, 0],
                        rtol=1e-5, atol=1e-5
                    ):
                        different_positions = True
                        break
    
    assert different_positions, "Random positions should be different for different kernel elements"
    
    # Check that positions are sorted and endpoints are fixed
    for out_ch in range(2):
        for in_ch in range(2):
            for kh in range(2):
                for kw in range(2):
                    pos = positions_random[out_ch, in_ch, kh, kw]
                    # Check sorted
                    assert torch.all(pos[1:] >= pos[:-1]), "Positions should be sorted"
                    # Check endpoints
                    assert pytest.approx(pos[0].item(), abs=1e-5) == -1.0, "First position should be -1.0"
                    assert pytest.approx(pos[-1].item(), abs=1e-5) == 1.0, "Last position should be 1.0"

def test_interpolation_constants():
    """Test that interpolation constants are correctly computed."""
    # Create layer
    conv = CustomPositionsPiecewiseConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        num_points=3,
        position_init="uniform",
        position_range=(-1, 1)
    )
    
    # Get positions, variable coefficients, and constant terms
    positions = conv.positions[0, 0, 0, 0]
    var_coefs = conv.variable_coefficients[0, 0, 0, 0]
    const_terms = conv.constant_terms[0, 0, 0, 0]
    
    # Expected positions: [-1, 0, 1]
    expected_positions = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(positions, expected_positions, rtol=1e-5, atol=1e-5)
    
    # Expected variable coefficients: [1/(0-(-1)), 1/(1-0)] = [1.0, 1.0]
    expected_var_coefs = torch.tensor([1.0, 1.0])
    torch.testing.assert_close(var_coefs, expected_var_coefs, rtol=1e-5, atol=1e-5)
    
    # Expected constant terms: [-(-1)/(0-(-1)), -0/(1-0)] = [1.0, 0.0]
    expected_const_terms = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(const_terms, expected_const_terms, rtol=1e-5, atol=1e-5)

def test_forward_pass_with_known_values():
    """Test the forward pass with known input, positions, and weights."""
    # Input tensor (1 batch, 1 channel, 1x1)
    x = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    
    # Create layer
    conv = CustomPositionsPiecewiseConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        num_points=3,
        position_init="uniform",
        position_range=(-1, 1),
        bias=False
    )
    
    # Set positions to [-1, 0, 1]
    with torch.no_grad():
        conv.positions[0, 0, 0, 0] = torch.tensor([-1.0, 0.0, 1.0])
        
        # Recompute interpolation constants
        conv._precompute_interpolation_constants()
        
        # Set weights to [1.0, 2.0, 3.0]
        conv.weights[0, 0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    
    # Apply the convolution
    output = conv(x)
    
    # Expected output:
    # For x=0.5 (between positions 0 and 1):
    # - Left weight = 2.0 (at position 0)
    # - Right weight = 3.0 (at position 1)
    # - t = (0.5 - 0) / (1 - 0) = 0.5
    # - Output = 2.0 + 0.5 * (3.0 - 2.0) = 2.5
    expected_output = torch.tensor([[[[2.5]]]], dtype=torch.float32)
    
    # Check that the output matches the expected value
    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)

def test_move_smoothest():
    """Test the move_smoothest operation."""
    # Create layer
    conv = CustomPositionsPiecewiseConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        num_points=5,
        position_init="uniform",
        position_range=(-1, 1)
    )
    
    # Get original positions
    original_positions = conv.positions.clone()
    
    # Apply move_smoothest
    moved = conv.move_smoothest()
    
    # Check that points were moved
    assert moved, "move_smoothest should return True"
    
    # Check that positions have changed
    assert not torch.allclose(conv.positions, original_positions, rtol=1e-5, atol=1e-5), \
        "Positions should change after move_smoothest"
    
    # Check that positions are still sorted and endpoints are fixed
    positions = conv.positions[0, 0, 0, 0]
    assert torch.all(positions[1:] >= positions[:-1]), "Positions should be sorted"
    assert pytest.approx(positions[0].item(), abs=1e-5) == -1.0, "First position should be -1.0"
    assert pytest.approx(positions[-1].item(), abs=1e-5) == 1.0, "Last position should be 1.0"
    
    # Check that the number of points is still the same
    assert positions.size(0) == 5, "Number of points should remain the same"

def test_gradient_flow():
    """Test that gradients flow through the layer."""
    # Create input tensor that requires grad
    x = torch.randn(2, 1, 4, 4, requires_grad=True)
    
    # Create layer
    conv = CustomPositionsPiecewiseConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding=1,
        num_points=5,
        position_init="uniform"
    )
    
    # Forward pass
    output = conv(x)
    
    # Compute loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check that gradients were computed for input
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    
    # Check that gradients were computed for weights
    assert conv.weights.grad is not None
    assert not torch.isnan(conv.weights.grad).any()
    
    # Check that gradients were computed for bias
    assert conv.bias.grad is not None
    assert not torch.isnan(conv.bias.grad).any()

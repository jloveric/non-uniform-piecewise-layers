import torch
import pytest
from non_uniform_piecewise_layers import AdaptivePiecewiseConv2d
from non_uniform_piecewise_layers.utils import largest_error

def test_conv_initialization():
    """Test that Conv2d is initialized with correct dimensions and points"""
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    num_points = 3
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    )
    
    # Check piecewise layer dimensions
    assert conv.piecewise.num_inputs == in_channels * kernel_size * kernel_size
    assert conv.piecewise.num_outputs == out_channels
    assert conv.piecewise.num_points == num_points
    
    # Check positions and values shapes
    assert conv.piecewise.positions.shape == (in_channels * kernel_size * kernel_size, out_channels, num_points)
    assert conv.piecewise.values.shape == (in_channels * kernel_size * kernel_size, out_channels, num_points)

def test_conv_forward():
    """Test that forward pass maintains correct shapes"""
    batch_size = 32
    in_channels = 3
    out_channels = 16
    height = 28
    width = 28
    kernel_size = 3
    padding = 1
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    y = conv(x)
    
    # Output should maintain height and width due to padding
    assert y.shape == (batch_size, out_channels, height, width)
    
    # Test without padding
    conv_no_pad = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=0
    )
    
    y_no_pad = conv_no_pad(x)
    expected_size = height - kernel_size + 1
    
    # Output should be smaller due to no padding
    assert y_no_pad.shape == (batch_size, out_channels, expected_size, expected_size)

def test_move_smoothest():
    """Test that move_smoothest works correctly"""
    in_channels = 2
    out_channels = 3
    kernel_size = 3
    num_points = 5
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    )
    
    # Get initial number of points
    initial_num_points = conv.piecewise.num_points
    
    # Should be able to move points since we have more than 2 points
    success = conv.move_smoothest()
    
    # Number of points should remain the same
    assert conv.piecewise.num_points == initial_num_points
    assert success

def test_move_smoothest_edge_case():
    """Test that move_smoothest handles edge cases correctly"""
    in_channels = 2
    out_channels = 3
    kernel_size = 3
    num_points = 2  # Only 2 points, can't move
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    )
    
    # Should not be able to move points when only 2 points exist
    success = conv.move_smoothest()
    
    # Operation should fail
    assert not success
    
    # Number of points should remain the same
    assert conv.piecewise.num_points == num_points

def test_move_smoothest_weighted():
    """Test that move_smoothest works with weighted parameter"""
    in_channels = 2
    out_channels = 3
    kernel_size = 3
    num_points = 5
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    )
    
    # Get initial number of points
    initial_num_points = conv.piecewise.num_points
    
    # Try with weighted=False
    success = conv.move_smoothest(weighted=False)
    
    # Number of points should remain the same
    assert conv.piecewise.num_points == initial_num_points
    assert success

def test_stride():
    """Test that stride parameter works correctly"""
    batch_size = 4
    in_channels = 2
    out_channels = 4
    height = 8
    width = 8
    kernel_size = 3
    stride = 2
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    y = conv(x)
    
    expected_size = (height - kernel_size) // stride + 1
    assert y.shape == (batch_size, out_channels, expected_size, expected_size)

def test_invalid_initialization():
    """Test that invalid initialization raises appropriate errors"""
    with pytest.raises(ValueError):
        # kernel_size must be positive
        AdaptivePiecewiseConv2d(in_channels=3, out_channels=16, kernel_size=0)
    
    with pytest.raises(ValueError):
        # num_points must be at least 2
        AdaptivePiecewiseConv2d(in_channels=3, out_channels=16, kernel_size=3, num_points=1)

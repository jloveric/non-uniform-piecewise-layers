import torch
import pytest
from non_uniform_piecewise_layers import AdaptivePiecewiseConv2d

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

def test_largest_error():
    """Test that largest_error returns valid points"""
    batch_size = 4
    in_channels = 2
    out_channels = 4
    height = 8
    width = 8
    kernel_size = 3
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    y = conv(x)
    error = torch.abs(y)  # Use output as mock error
    
    x_error = conv.largest_error(error, x)
    
    # Should return a tensor of input values
    assert isinstance(x_error, torch.Tensor)
    assert x_error.shape == (in_channels * kernel_size * kernel_size,)
    
    # Values should be within position range
    assert torch.all(x_error >= conv.piecewise.position_min)
    assert torch.all(x_error <= conv.piecewise.position_max)

def test_insert_points():
    """Test that insert_points adds points correctly"""
    batch_size = 4
    in_channels = 2
    out_channels = 4
    height = 8
    width = 8
    kernel_size = 3
    initial_points = 3
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=initial_points
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    initial_num_points = conv.piecewise.num_points
    
    conv.insert_points(x)
    
    # Number of points should increase
    assert conv.piecewise.num_points > initial_num_points
    
    # Shapes should update accordingly
    assert conv.piecewise.positions.shape[2] == conv.piecewise.num_points
    assert conv.piecewise.values.shape[2] == conv.piecewise.num_points

def test_insert_nearby_point():
    """Test that insert_nearby_point adds points correctly"""
    batch_size = 4
    in_channels = 2
    out_channels = 4
    height = 8
    width = 8
    kernel_size = 3
    initial_points = 3
    
    conv = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=initial_points
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    y = conv(x)
    error = torch.abs(y)  # Use output as mock error
    x_error = conv.largest_error(error, x)
    
    initial_num_points = conv.piecewise.num_points
    conv.insert_nearby_point(x_error)
    
    # Number of points should increase by out_channels (one point per output channel)
    assert conv.piecewise.num_points == initial_num_points + out_channels
    
    # Shapes should update accordingly
    assert conv.piecewise.positions.shape[2] == conv.piecewise.num_points
    assert conv.piecewise.values.shape[2] == conv.piecewise.num_points

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

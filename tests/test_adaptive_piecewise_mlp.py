import torch
import pytest
from non_uniform_piecewise_layers import AdaptivePiecewiseMLP

def test_mlp_initialization():
    """Test that MLP is initialized with correct layer widths and points"""
    width = [2, 4, 3, 1]  # 3 layers with varying widths
    num_points = 3
    
    mlp = AdaptivePiecewiseMLP(
        width=width,
        num_points=num_points
    )
    
    # Check number of layers
    assert len(mlp.layers) == len(width) - 1
    
    # Check each layer has correct input/output dimensions and number of points
    for i, layer in enumerate(mlp.layers):
        assert layer.positions.shape[0] == width[i]  # num_inputs
        assert layer.positions.shape[1] == width[i+1]  # num_outputs
        assert layer.positions.shape[2] == num_points  # num_points

def test_mlp_forward():
    """Test that forward pass maintains correct shapes"""
    width = [2, 3, 1]  # 2 layers
    mlp = AdaptivePiecewiseMLP(width=width)
    
    # Test single input
    x_single = torch.tensor([[0.5, 0.3]])  # (1, 2)
    out_single = mlp(x_single)
    assert out_single.shape == (1, 1)  # Should output (1, 1)
    
    # Test batch input
    batch_size = 32
    x_batch = torch.randn(batch_size, 2)  # (32, 2)
    out_batch = mlp(x_batch)
    assert out_batch.shape == (batch_size, 1)  # Should output (32, 1)

def test_largest_error():
    """Test that largest_error returns valid points"""
    width = [2, 1]  # Single layer, 2 inputs -> 1 output
    mlp = AdaptivePiecewiseMLP(width=width)
    
    # Create batch of inputs
    x = torch.tensor([
        [0.5, 0.3],
        [-0.2, 0.1],
        [0.7, -0.4]
    ])
    error = torch.ones(3, 1)  # Uniform error for each output
    
    # Should return a point not too close to existing points
    x_at_error = mlp.largest_error(error, x)
    
    assert x_at_error is not None
    assert x_at_error.shape == (1, 2)  # Should return (1, num_inputs)
    assert torch.all((-1 <= x_at_error) & (x_at_error <= 1))

def test_insert_points():
    """Test that insert_points adds points correctly"""
    width = [2, 3, 1]  # 2 layers
    mlp = AdaptivePiecewiseMLP(width=width)
    
    initial_points = [layer.positions.shape[-1] for layer in mlp.layers]
    
    # Insert a point
    x = torch.tensor([[0.5, 0.3]])
    success = mlp.insert_points(x)
    
    # Should successfully insert point
    assert success
    
    # Check that points were added
    final_points = [layer.positions.shape[-1] for layer in mlp.layers]
    assert all(f > i for f, i in zip(final_points, initial_points))

def test_insert_nearby_point():
    """Test that insert_nearby_point adds points correctly"""
    width = [2, 3, 1]  # 2 layers
    mlp = AdaptivePiecewiseMLP(width=width)
    
    initial_points = [layer.positions.shape[-1] for layer in mlp.layers]
    
    # Insert a nearby point
    x = torch.tensor([[0.5, 0.3]])
    success = mlp.insert_nearby_point(x)
    
    # Should successfully insert point
    assert success
    
    # Check that points were added
    final_points = [layer.positions.shape[-1] for layer in mlp.layers]
    assert all(f > i for f, i in zip(final_points, initial_points))

def test_invalid_initialization():
    """Test that invalid initialization raises appropriate errors"""
    # Width list too short
    with pytest.raises(ValueError):
        AdaptivePiecewiseMLP(width=[2])
    
    # Invalid number of points
    with pytest.raises(ValueError):
        AdaptivePiecewiseMLP(width=[2, 1], num_points=0)

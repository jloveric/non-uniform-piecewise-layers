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

def test_remove_add():
    """Test that remove_add correctly modifies points in all layers."""
    torch.manual_seed(42)
    num_inputs = 2
    hidden_size = 3
    num_outputs = 2
    num_points = 4
    layer_sizes = [num_inputs, hidden_size, num_outputs]
    mlp = AdaptivePiecewiseMLP(layer_sizes, num_points, clamp=True)  # Enable clamping
    
    # Set specific positions and values for each layer
    for layer in mlp.layers:
        positions = torch.zeros(layer.num_inputs, layer.num_outputs, num_points)
        values = torch.zeros(layer.num_inputs, layer.num_outputs, num_points)
        
        # Set positions for all inputs/outputs
        positions[:, :] = torch.tensor([-1.0, -0.33, 0.33, 1.0])  # Use full [-1, 1] range
        
        # Set values to create regions of high and low error
        # Point at x=0.33 should be smoothest (removed)
        for i in range(layer.num_inputs):
            for j in range(layer.num_outputs):
                values[i, j] = torch.tensor([-1.0, -0.2 + 0.1*i + 0.2*j, 0.7, 1.0])
        
        layer.positions.data = positions
        layer.values.data = values
    
    # Store original number of points for each layer
    original_num_points = [layer.num_points for layer in mlp.layers]
    
    # Choose a point to add (midpoint between x=-0.33 and x=0.33)
    batch_size = 1
    point = torch.tensor([[0.0, 0.0]])  # Shape: (batch_size, num_inputs)
    
    # Perform remove_add operation
    success = mlp.remove_add(point)
    assert success, "Failed to perform remove_add operation"
    
    # Check that number of points remained the same in each layer
    for i, layer in enumerate(mlp.layers):
        assert layer.num_points == original_num_points[i], \
            f"Number of points changed in layer {i} from {original_num_points[i]} to {layer.num_points}"
    
    # Check that points were actually moved in each layer
    for i, layer in enumerate(mlp.layers):
        positions_changed = False
        for j in range(layer.num_inputs):
            for k in range(layer.num_outputs):
                if not torch.allclose(layer.positions[j, k], torch.tensor([-1.0, -0.33, 0.33, 1.0])):
                    positions_changed = True
                    break
        assert positions_changed, f"No positions were changed in layer {i}"
    
    # Test edge case: only 2 points
    mlp = AdaptivePiecewiseMLP([1, 1], 2, clamp=True)  # Enable clamping
    success = mlp.remove_add(torch.tensor([[0.0]]))  # Shape: (batch_size, num_inputs)
    assert not success, "Should not be able to add/remove points when only 2 points exist"
    assert all(layer.num_points == 2 for layer in mlp.layers), \
        "Number of points should not change when operation fails"


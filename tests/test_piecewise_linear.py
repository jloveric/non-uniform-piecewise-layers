import torch
import pytest
from non_uniform_piecewise_layers import NonUniformPiecewiseLinear

def test_add_point_at_max_error():
    # Create a simple layer with 1 input, 1 output, and 2 points
    layer = NonUniformPiecewiseLinear(num_inputs=1, num_outputs=1, num_points=2)
    
    # Set specific positions and values for testing
    layer.positions.data = torch.tensor([[[0.0, 1.0]]])
    layer.values.data = torch.tensor([[[0.0, 1.0]]])
    
    # Create input and run forward pass
    x = torch.tensor([[0.5]])
    y = layer(x)
    
    # Set artificial gradients (maximum at the first point)
    layer.values.grad = torch.tensor([[[2.0, 1.0]]])
    
    # Add point using different strategies
    for strategy in range(3):
        # Reset layer for each strategy
        layer = NonUniformPiecewiseLinear(num_inputs=1, num_outputs=1, num_points=2)
        layer.positions.data = torch.tensor([[[0.0, 1.0]]])
        layer.values.data = torch.tensor([[[0.0, 1.0]]])
        layer.values.grad = torch.tensor([[[2.0, 1.0]]])
        
        # Compute absolute gradients
        abs_grad = torch.abs(layer.values.grad)
        success = layer.add_point_at_max_error(abs_grad=abs_grad, split_strategy=strategy)
        
        assert success, "Failed to add point"
        assert layer.num_points == 3, "Number of points did not increase"
        
        # Check new positions are properly ordered
        positions = layer.positions[0, 0]
        assert torch.all(positions[:-1] < positions[1:])

def test_add_point_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Create layer on GPU
    layer = NonUniformPiecewiseLinear(num_inputs=1, num_outputs=1, num_points=2).cuda()
    layer.positions.data = torch.tensor([[[0.0, 1.0]]], device='cuda')
    layer.values.data = torch.tensor([[[0.0, 1.0]]], device='cuda')
    
    # Run forward pass on GPU
    x = torch.tensor([[0.5]], device='cuda')
    y = layer(x)
    
    # Set gradients on GPU
    layer.values.grad = torch.tensor([[[2.0, 1.0]]], device='cuda')
    
    # Compute absolute gradients
    abs_grad = torch.abs(layer.values.grad)
    # Add point
    success = layer.add_point_at_max_error(abs_grad=abs_grad, split_strategy=0)
    
    assert success, "Failed to add point"
    assert layer.num_points == 3, "Number of points did not increase"
    assert layer.positions.device.type == 'cuda'
    assert layer.values.device.type == 'cuda'

def test_no_gradients():
    layer = NonUniformPiecewiseLinear(num_inputs=1, num_outputs=1, num_points=2)
    with pytest.raises(ValueError, match="No gradients available"):
        abs_grad = torch.zeros_like(layer.values)  # Zero gradients should trigger the error
        layer.add_point_at_max_error(abs_grad=abs_grad)

def test_multiple_dimensions():
    # Test with multiple inputs and outputs
    layer = NonUniformPiecewiseLinear(num_inputs=2, num_outputs=3, num_points=2)
    
    # Set gradients with known maximum
    grads = torch.zeros(2, 3, 2)
    grads[1, 2, 0] = 5.0  # Maximum at input=1, output=2, point=0
    layer.values.grad = grads
    
    # Compute absolute gradients
    abs_grad = torch.abs(layer.values.grad)
    success = layer.add_point_at_max_error(abs_grad=abs_grad)
    
    assert success, "Failed to add point"
    assert layer.num_points == 3, "Number of points did not increase"
    
    # Check that point was added in correct input-output pair
    original_positions = layer.positions[1, 2]  # Check positions for input=1, output=2
    assert len(original_positions) == 3  # Should now have 3 points

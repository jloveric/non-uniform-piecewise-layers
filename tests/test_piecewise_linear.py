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

def test_insert_existing_point():
    import torch
    from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear

    torch.manual_seed(42)
    num_inputs = 2
    num_outputs = 1
    num_points = 3
    layer = AdaptivePiecewiseLinear(num_inputs, num_outputs, num_points)

    # Choose a test input for which to check the output
    test_input = torch.tensor([[0.0, 0.0]])
    output_before = layer(test_input).detach().clone()
    elements_start = layer.positions.numel()

    # For each input dimension, choose an existing point from the layer's positions
    existing_points = torch.stack([layer.positions[i, 0, 1] for i in range(num_inputs)])

    # Insert the existing points
    inserted = layer.insert_points(existing_points)

    # After insertion:
    # 1. The output should remain exactly the same since we inserted existing points
    output_after = layer(test_input).detach().clone()
    assert torch.allclose(output_before, output_after, atol=1e-5), "Output changed after inserting existing points"
    
    # 2. The number of elements should increase by num_inputs (one duplicate per input dimension)
    elements_end = layer.positions.numel()
    assert elements_end == elements_start + num_inputs, f"Expected {elements_start + num_inputs} elements after insertion, got {elements_end}"
    
    # 3. The duplicated points should have identical values
    for i in range(num_inputs):
        pos = layer.positions[i, 0]
        vals = layer.values[i, 0]
        # Find where duplicates exist
        for p in existing_points:
            matches = (pos == p)
            if matches.sum() > 1:  # If duplicates exist
                duplicate_vals = vals[matches]
                # All duplicates should have the same value
                assert torch.allclose(duplicate_vals, duplicate_vals[0]), f"Duplicate points have different values in dimension {i}"

def test_insert_new_point_2input():
    import torch
    from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear

    torch.manual_seed(42)
    num_inputs = 2
    num_outputs = 1
    num_points = 3
    layer = AdaptivePiecewiseLinear(num_inputs, num_outputs, num_points)

    # Save initial number of points and output
    initial_points = layer.num_points
    test_input = torch.tensor([[0.0, 0.1]])  # Test point between 0.0 and 0.25
    output_before = layer(test_input).detach().clone()

    # New point to insert (for 2 inputs, specifying a new value for each dimension)
    new_point = torch.tensor([0.0, 0.25])
    inserted = layer.insert_points(new_point)

    # Compute output after new point insertion
    output_after = layer(test_input).detach().clone()

    # Check that the number of points increased by one (i.e. even if the point is new, it is inserted)
    assert layer.num_points == initial_points + 1, "Number of points did not increase after inserting new point"

    # Since we're using linear interpolation, the output at 0.1 should be the same
    # even after inserting a point at 0.25
    assert torch.allclose(output_before, output_after, atol=1e-5), "Output changed after insertion of new point, but it should stay the same due to linear interpolation"

    # Test that the output at 0.25 matches the interpolated value
    test_at_new = torch.tensor([[0.0, 0.25]])
    output_at_new = layer(test_at_new).detach()
    
    # Verify that the value at 0.25 in dimension 1 is what we interpolated
    assert torch.allclose(layer.values[1, 0, 2], torch.tensor(0.0885385051369667), atol=1e-5), "Value at new point does not match interpolated value"
    
    # Since we have duplicates at x=0.0 in dimension 0, we expect the output to be different
    # from what we interpolated in dimension 1. This is because the duplicates in dimension 0
    # affect the final output through summation.

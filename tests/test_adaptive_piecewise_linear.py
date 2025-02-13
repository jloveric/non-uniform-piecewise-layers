import torch
import pytest
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear

def test_insert_existing_point():
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

def test_compute_removal_errors():
    torch.manual_seed(42)
    num_inputs = 1
    num_outputs = 1
    num_points = 4  # We'll have 2 internal points that can be removed
    layer = AdaptivePiecewiseLinear(num_inputs, num_outputs, num_points)
    
    # Set specific positions and values for testing
    layer.positions.data = torch.tensor([[[0.0, 0.33, 0.67, 1.0]]])
    layer.values.data = torch.tensor([[[0.0, 0.5, 0.8, 1.0]]])
    
    # Compute removal errors
    errors, indices = layer.compute_removal_errors()
    
    # Check shapes
    assert errors.shape == (1, 1, 2), f"Expected error shape (1,1,2), got {errors.shape}"
    assert indices.shape == (1, 1, 2), f"Expected indices shape (1,1,2), got {indices.shape}"
    
    # Check indices are correct (should be [1,2] for internal points)
    expected_indices = torch.tensor([[[1, 2]]])
    assert torch.all(indices == expected_indices), f"Expected indices {expected_indices}, got {indices}"
    
    # Calculate expected errors manually
    # For point at x=0.33:
    # Current value = 0.5
    # If removed, interpolated between x=0 and x=0.67:
    # t = (0.33 - 0) / (0.67 - 0) = 0.4925
    # interpolated_val = 0.0 + 0.4925 * (0.8 - 0.0) = 0.394
    # Error = |0.5 - 0.394| = 0.106
    t1 = 0.33 / 0.67  # = 0.4925
    interpolated_val1 = 0.0 + t1 * (0.8 - 0.0)  # = 0.394
    expected_error_1 = abs(0.5 - interpolated_val1)  # ≈ 0.106
    
    # For point at x=0.67:
    # Current value = 0.8
    # If removed, interpolated between x=0.33 and x=1.0:
    # t = (0.67 - 0.33) / (1.0 - 0.33) = 0.507
    # interpolated_val = 0.5 + 0.507 * (1.0 - 0.5) = 0.7535
    # Error = |0.8 - 0.7535| = 0.0465
    t2 = (0.67 - 0.33) / (1.0 - 0.33)  # = 0.507
    interpolated_val2 = 0.5 + t2 * (1.0 - 0.5)  # = 0.7535
    expected_error_2 = abs(0.8 - interpolated_val2)  # ≈ 0.0465
    
    expected_errors = torch.tensor([[[expected_error_1, expected_error_2]]])
    assert torch.allclose(errors, expected_errors, atol=1e-6), f"Expected errors {expected_errors}, got {errors}"
    
    # Test with multiple inputs/outputs
    layer = AdaptivePiecewiseLinear(num_inputs=2, num_outputs=2, num_points=4)
    errors, indices = layer.compute_removal_errors()
    assert errors.shape == (2, 2, 2), f"Expected error shape (2,2,2), got {errors.shape}"
    assert indices.shape == (2, 2, 2), f"Expected indices shape (2,2,2), got {indices.shape}"

def test_compute_removal_errors_edge_cases():
    # Test with minimum number of points (2)
    layer = AdaptivePiecewiseLinear(num_inputs=1, num_outputs=1, num_points=2)
    errors, indices = layer.compute_removal_errors()
    assert errors.numel() == 0, "Expected empty tensor for errors with only 2 points"
    assert indices.numel() == 0, "Expected empty tensor for indices with only 2 points"
    
    # Test with 3 points (1 removable point)
    layer = AdaptivePiecewiseLinear(num_inputs=1, num_outputs=1, num_points=3)
    errors, indices = layer.compute_removal_errors()
    assert errors.shape == (1, 1, 1), f"Expected error shape (1,1,1), got {errors.shape}"
    assert indices.shape == (1, 1, 1), f"Expected indices shape (1,1,1), got {indices.shape}"
    assert indices[0,0,0] == 1, "Middle point should have index 1"

def test_compute_removal_errors_3in_2out():
    torch.manual_seed(42)
    num_inputs = 3
    num_outputs = 2
    num_points = 4
    layer = AdaptivePiecewiseLinear(num_inputs, num_outputs, num_points)
    
    # Set specific positions and values for testing
    # We'll test input 0, output 0 and input 2, output 1
    positions = torch.zeros(num_inputs, num_outputs, num_points)
    values = torch.zeros(num_inputs, num_outputs, num_points)
    
    # Set positions for all inputs/outputs
    positions[:, :] = torch.tensor([0.0, 0.33, 0.67, 1.0])
    
    # Set specific values for input 0, output 0
    values[0, 0] = torch.tensor([0.0, 0.5, 0.8, 1.0])
    
    # Set specific values for input 2, output 1
    values[2, 1] = torch.tensor([0.0, 0.3, 0.9, 1.2])
    
    # Set the rest to linear interpolation
    for i in range(num_inputs):
        for j in range(num_outputs):
            if (i == 0 and j == 0) or (i == 2 and j == 1):
                continue
            values[i, j] = torch.linspace(0, 1, num_points)
    
    layer.positions.data = positions
    layer.values.data = values
    
    # Compute removal errors
    errors, indices = layer.compute_removal_errors()
    
    # Check shapes
    assert errors.shape == (3, 2, 2), f"Expected error shape (3,2,2), got {errors.shape}"
    assert indices.shape == (3, 2, 2), f"Expected indices shape (3,2,2), got {indices.shape}"
    
    # Check indices are correct for all dimensions (should be [1,2] for internal points)
    expected_indices = torch.zeros((3, 2, 2), dtype=torch.long)
    expected_indices[..., 0] = 1  # First internal point index
    expected_indices[..., 1] = 2  # Second internal point index
    assert torch.all(indices == expected_indices), f"Expected indices {expected_indices}, got {indices}"
    
    # Test specific error calculations for input 0, output 0
    # For point at x=0.33:
    t1 = 0.33 / 0.67  # = 0.4925
    interpolated_val1 = 0.0 + t1 * (0.8 - 0.0)  # = 0.394
    expected_error_1 = abs(0.5 - interpolated_val1)  # ≈ 0.106
    
    # For point at x=0.67:
    t2 = (0.67 - 0.33) / (1.0 - 0.33)  # = 0.507
    interpolated_val2 = 0.5 + t2 * (1.0 - 0.5)  # = 0.7535
    expected_error_2 = abs(0.8 - interpolated_val2)  # ≈ 0.0465
    
    # Check errors for input 0, output 0
    assert torch.allclose(errors[0, 0], torch.tensor([expected_error_1, expected_error_2]), atol=1e-6), \
        f"Expected errors for input 0, output 0 to be [{expected_error_1}, {expected_error_2}], got {errors[0, 0]}"
    
    # Test specific error calculations for input 2, output 1
    # For point at x=0.33:
    t3 = 0.33 / 0.67
    interpolated_val3 = 0.0 + t3 * (0.9 - 0.0)  # = 0.44325
    expected_error_3 = abs(0.3 - interpolated_val3)  # ≈ 0.14325
    
    # For point at x=0.67:
    t4 = (0.67 - 0.33) / (1.0 - 0.33)
    interpolated_val4 = 0.3 + t4 * (1.2 - 0.3)  # = 0.7567
    expected_error_4 = abs(0.9 - interpolated_val4)  # ≈ 0.1433
    
    # Check errors for input 2, output 1
    assert torch.allclose(errors[2, 1], torch.tensor([expected_error_3, expected_error_4]), atol=1e-6), \
        f"Expected errors for input 2, output 1 to be [{expected_error_3}, {expected_error_4}], got {errors[2, 1]}"
    
    # For the linear interpolation cases, errors should be very small
    # since removing a point from a linear function shouldn't change the values much
    # Note: Due to numerical precision in torch.linspace, we use a larger tolerance
    assert torch.all(errors[1, 0] < 0.01), f"Expected small errors for linear case, got {errors[1, 0]}"
    assert torch.all(errors[1, 1] < 0.01), f"Expected small errors for linear case, got {errors[1, 1]}"

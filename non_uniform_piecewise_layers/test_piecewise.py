import torch
from non_uniform_piecewise_layers.piecewise_linear import NonUniformPiecewiseLinear

def test_piecewise_layer():
    # Create a layer with 2 inputs, 3 outputs, and 5 points per function
    layer = NonUniformPiecewiseLinear(num_inputs=2, num_outputs=3, num_points=5)
    
    # Create a sample input
    x = torch.tensor([[0.5, -0.3], [0.1, 0.8]], dtype=torch.float32)
    
    # Forward pass
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nInput:\n{x}")
    print(f"\nOutput:\n{output}")
    
    # Test monotonic enforcement
    print("\nPositions before enforcing monotonic:")
    print(layer.positions[0, 0])  # Show first input-output pair's positions
    
    layer.enforce_monotonic()
    
    print("\nPositions after enforcing monotonic:")
    print(layer.positions[0, 0])  # Show first input-output pair's positions

if __name__ == "__main__":
    test_piecewise_layer()

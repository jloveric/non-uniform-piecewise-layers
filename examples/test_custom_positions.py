import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.custom_positions_conv import CustomPositionsPiecewiseConv2d

def test_custom_positions_initialization():
    """Test that custom positions are properly initialized for each kernel element"""
    print("Testing custom positions initialization...")
    
    # Create a convolutional layer with random positions
    conv = CustomPositionsPiecewiseConv2d(
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        stride=1,
        padding=0,
        num_points=5,
        position_range=(-1, 1),
        position_init="random",  # Use random position initialization
        weight_init="random"
    )
    
    # Get positions
    positions = conv.positions
    
    # Print positions shape
    print(f"Positions shape: {positions.shape}")
    
    # Check that positions are different for each kernel element
    different_positions = True
    for out_ch in range(2):
        for in_ch in range(2):
            for kh in range(2):
                for kw in range(2):
                    # Compare with other kernel elements
                    for out_ch2 in range(2):
                        for in_ch2 in range(2):
                            for kh2 in range(2):
                                for kw2 in range(2):
                                    if out_ch != out_ch2 or in_ch != in_ch2 or kh != kh2 or kw != kw2:
                                        if torch.allclose(positions[out_ch, in_ch, kh, kw], positions[out_ch2, in_ch2, kh2, kw2]):
                                            different_positions = False
                                            print(f"Positions for kernel elements ({out_ch}, {in_ch}, {kh}, {kw}) and ({out_ch2}, {in_ch2}, {kh2}, {kw2}) are identical.")
    
    if different_positions:
        print("✅ All kernel elements have different positions.")
    else:
        print("❌ Some kernel elements have identical positions.")
    
    # Visualize positions for each kernel element
    plt.figure(figsize=(12, 8))
    
    # Use different colors and markers for each kernel element
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*']
    
    # Plot positions for each kernel element
    for out_ch in range(2):
        for in_ch in range(2):
            for kh in range(2):
                for kw in range(2):
                    idx = out_ch * 8 + in_ch * 4 + kh * 2 + kw
                    color = colors[idx % len(colors)]
                    marker = markers[idx % len(markers)]
                    
                    pos = positions[out_ch, in_ch, kh, kw].detach().cpu().numpy()
                    plt.plot(np.arange(len(pos)), pos, color=color, marker=marker, 
                             label=f'out_{out_ch}_in_{in_ch}_k_{kh}_{kw}')
    
    plt.title('Custom Positions for Each Kernel Element')
    plt.xlabel('Index')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('custom_positions.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    
    print("Positions visualization saved to 'custom_positions.png'")
    
    return conv

def test_forward_pass(conv):
    """Test the forward pass with custom positions"""
    print("\nTesting forward pass...")
    
    # Create a simple input tensor
    batch_size = 2
    in_channels = conv.in_channels
    height = 3
    width = 3
    
    # Create a tensor with values increasing from -1 to 1
    x = torch.linspace(-1, 1, batch_size * in_channels * height * width)
    x = x.reshape(batch_size, in_channels, height, width)
    
    # Forward pass
    output = conv(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check that output is not all zeros or NaNs
    if torch.isnan(output).any():
        print("❌ Output contains NaN values.")
    else:
        print("✅ Output does not contain NaN values.")
    
    if torch.all(output == 0):
        print("❌ Output is all zeros.")
    else:
        print("✅ Output is not all zeros.")
    
    return x, output

def test_move_smoothest(conv, x):
    """Test the move_smoothest operation"""
    print("\nTesting move_smoothest operation...")
    
    # Get original positions
    original_positions = conv.positions.clone()
    
    # Forward pass before moving points
    output_before = conv(x)
    
    # Move points
    moved = conv.move_smoothest()
    
    if moved:
        print("✅ Points were moved successfully.")
    else:
        print("❌ Points were not moved.")
    
    # Get updated positions
    updated_positions = conv.positions
    
    # Check that positions have changed
    if torch.allclose(original_positions, updated_positions):
        print("❌ Positions did not change after move_smoothest.")
    else:
        print("✅ Positions changed after move_smoothest.")
    
    # Forward pass after moving points
    output_after = conv(x)
    
    # Check that output has changed
    if torch.allclose(output_before, output_after):
        print("❌ Output did not change after move_smoothest.")
    else:
        print("✅ Output changed after move_smoothest.")
    
    # Visualize positions before and after moving
    plt.figure(figsize=(12, 8))
    
    # Use different colors for each kernel element
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    
    # Plot positions for the first output and input channel
    out_ch = 0
    in_ch = 0
    
    for kh in range(2):
        for kw in range(2):
            idx = kh * 2 + kw
            color = colors[idx % len(colors)]
            
            # Original positions
            pos_orig = original_positions[out_ch, in_ch, kh, kw].detach().cpu().numpy()
            plt.plot(np.arange(len(pos_orig)), pos_orig, color=color, linestyle='--', 
                     label=f'Original k_{kh}_{kw}')
            
            # Updated positions
            pos_updated = updated_positions[out_ch, in_ch, kh, kw].detach().cpu().numpy()
            plt.plot(np.arange(len(pos_updated)), pos_updated, color=color, linestyle='-', 
                     label=f'Updated k_{kh}_{kw}')
    
    plt.title('Positions Before and After move_smoothest')
    plt.xlabel('Index')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('positions_before_after.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    
    print("Positions visualization saved to 'positions_before_after.png'")

def test_interpolation_accuracy():
    """Test the accuracy of the piecewise linear interpolation"""
    print("\nTesting interpolation accuracy...")
    
    # Create a simple 1D function to approximate
    def target_function(x):
        return torch.sin(2 * np.pi * x)
    
    # Create a convolutional layer with uniform positions
    conv = CustomPositionsPiecewiseConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=0,
        num_points=10,  # Use more points for better approximation
        position_range=(-1, 1),
        position_init="uniform",  # Use uniform initialization for this test
        weight_init="random"
    )
    
    # Set weights to match the target function at the positions
    with torch.no_grad():
        positions = conv.positions[0, 0, 0, 0]
        target_values = target_function(positions)
        conv.weights.data[0, 0, :, 0, 0] = target_values
    
    # Create input tensor with values from -1 to 1
    num_samples = 100
    x = torch.linspace(-1, 1, num_samples).reshape(1, 1, num_samples, 1)
    
    # Forward pass
    output = conv(x)
    
    # Compute target values
    target = target_function(x.reshape(num_samples))
    
    # Compute error
    error = torch.abs(output.reshape(num_samples) - target)
    mean_error = error.mean().item()
    max_error = error.max().item()
    
    print(f"Mean absolute error: {mean_error:.6f}")
    print(f"Maximum absolute error: {max_error:.6f}")
    
    # Visualize the approximation
    plt.figure(figsize=(10, 6))
    
    # Plot target function
    x_np = x.reshape(num_samples).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    plt.plot(x_np, target_np, 'b-', label='Target function')
    
    # Plot approximation
    output_np = output.reshape(num_samples).detach().cpu().numpy()
    plt.plot(x_np, output_np, 'r--', label='Piecewise approximation')
    
    # Plot positions and weights
    positions_np = positions.detach().cpu().numpy()
    weights_np = conv.weights[0, 0, :, 0, 0].detach().cpu().numpy()
    plt.plot(positions_np, weights_np, 'go', label='Positions and weights')
    
    plt.title('Piecewise Linear Approximation of sin(2πx)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('interpolation_accuracy.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    
    print("Interpolation accuracy visualization saved to 'interpolation_accuracy.png'")

def main():
    """Run all tests"""
    # Test custom positions initialization
    conv = test_custom_positions_initialization()
    
    # Test forward pass
    x, output = test_forward_pass(conv)
    
    # Test move_smoothest operation
    test_move_smoothest(conv, x)
    
    # Test interpolation accuracy
    test_interpolation_accuracy()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()

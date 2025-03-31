import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.efficient_adaptive_piecewise_conv import EfficientAdaptivePiecewiseConv2d

def test_random_positions():
    """
    Test that random positions are properly initialized for each kernel element
    """
    # Create a convolutional layer with random positions
    conv = EfficientAdaptivePiecewiseConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        num_points=5,
        position_range=(-1, 1),
        position_init="random",
        weight_init="random"
    )
    
    # Check if custom_positions exists and has the right shape
    if hasattr(conv, "custom_positions") and conv.custom_positions is not None:
        print(f"Custom positions shape: {conv.custom_positions.shape}")
        # Should be [out_channels, in_channels, kernel_height, kernel_width, num_points]
        # = [1, 1, 2, 2, 5]
    else:
        print("No custom positions found!")
        return
    
    # Create a figure to visualize the positions
    plt.figure(figsize=(12, 8))
    
    # Use different colors and markers for each kernel element
    colors = ['b', 'g', 'r', 'c']
    markers = ['o', 's', '^', 'v']
    
    # Plot the positions for each kernel element
    for kh in range(2):
        for kw in range(2):
            # Get the positions for this kernel element
            positions = conv.custom_positions[0, 0, kh, kw].cpu().numpy()
            
            # Plot the positions
            plt.plot(
                range(len(positions)), 
                positions, 
                f"{colors[kh*2+kw]}{markers[kh*2+kw]}-",
                label=f"Kernel position ({kh}, {kw})",
                linewidth=2,
                markersize=10
            )
    
    plt.title("Random Positions for Each Kernel Element")
    plt.xlabel("Point Index")
    plt.ylabel("Position Value")
    plt.grid(True)
    plt.legend()
    plt.savefig("random_positions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Now create a test input and run a forward pass
    x = torch.randn(1, 1, 3, 3)  # Batch size 1, 1 channel, 3x3 input
    y = conv(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Now visualize the weights
    plt.figure(figsize=(12, 8))
    
    # Get the weights
    weights = conv.conv.weight.data.cpu()
    
    # Reshape weights to [out_channels, in_channels, num_points, kernel_height, kernel_width]
    weights_reshaped = weights.view(1, 1, 5, 2, 2)
    
    # Plot the weights for each kernel element
    for kh in range(2):
        for kw in range(2):
            # Get the positions and weights for this kernel element
            positions = conv.custom_positions[0, 0, kh, kw].cpu().numpy()
            kernel_weights = weights_reshaped[0, 0, :, kh, kw].numpy()
            
            # Plot the weights vs positions
            plt.plot(
                positions,
                kernel_weights,
                f"{colors[kh*2+kw]}{markers[kh*2+kw]}-",
                label=f"Kernel position ({kh}, {kw})",
                linewidth=2,
                markersize=10
            )
    
    plt.title("Weights vs Positions for Each Kernel Element")
    plt.xlabel("Position")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.legend()
    plt.savefig("weights_vs_positions.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    test_random_positions()

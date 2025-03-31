import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.efficient_adaptive_piecewise_conv import EfficientAdaptivePiecewiseConv2d

def visualize_positions_and_weights():
    """
    Create a simple model with random positions and visualize the weights
    """
    # Create a convolutional layer with random positions
    conv = EfficientAdaptivePiecewiseConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        stride=1,
        padding=0,
        num_points=5,
        position_range=(-1, 1),
        position_init="random",  # Use random position initialization
        weight_init="random"
    )
    
    # Get positions and reshape weights for visualization
    positions = conv.expansion.positions.data.cpu()
    weights = conv.conv.weight.data.cpu()
    
    # Reshape weights to [out_channels, in_channels, num_points, kernel_height, kernel_width]
    out_channels = weights.size(0)
    in_channels = conv.in_channels
    num_points = positions.size(0)
    kernel_size = conv.kernel_size
    weights_reshaped = weights.view(out_channels, in_channels, num_points, *kernel_size)
    
    # Create figure for visualization
    plt.figure(figsize=(12, 8))
    
    # Use different colors and markers for each kernel element
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']
    linestyles = ['-', '--', '-.', ':']
    
    # Plot positions vs weights for each kernel position
    for out_ch in range(out_channels):
        for in_ch in range(in_channels):
            for kh in range(kernel_size[0]):
                for kw in range(kernel_size[1]):
                    # Create a unique color/marker/linestyle combination for each kernel element
                    color_idx = (out_ch * in_channels + in_ch) % len(colors)
                    marker_idx = (kh * kernel_size[1] + kw) % len(markers)
                    linestyle_idx = (out_ch + in_ch + kh + kw) % len(linestyles)
                    
                    label = f'out_{out_ch}_in_{in_ch}_k_{kh}_{kw}'
                    plt.plot(positions.numpy(), 
                              weights_reshaped[out_ch, in_ch, :, kh, kw].numpy(), 
                              color=colors[color_idx],
                              marker=markers[marker_idx],
                              linestyle=linestyles[linestyle_idx],
                              label=label,
                              linewidth=2,
                              markersize=8,
                              alpha=0.7)
    
    plt.title('Convolutional Layer Piecewise Approximations')
    plt.xlabel('Position')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig('random_positions_visualization.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    
    print("Visualization saved to 'random_positions_visualization.png'")
    
    # Test if move_smoothest works with random positions
    print("Testing move_smoothest operation...")
    print(f"Original positions: {positions.numpy()}")
    
    # Move points
    conv.move_smoothest()
    
    # Get updated positions
    updated_positions = conv.expansion.positions.data.cpu()
    print(f"Updated positions: {updated_positions.numpy()}")
    
    # Visualize updated positions and weights
    updated_weights = conv.conv.weight.data.cpu()
    updated_weights_reshaped = updated_weights.view(out_channels, in_channels, updated_positions.size(0), *kernel_size)
    
    plt.figure(figsize=(12, 8))
    
    # Plot updated positions vs weights
    for out_ch in range(out_channels):
        for in_ch in range(in_channels):
            for kh in range(kernel_size[0]):
                for kw in range(kernel_size[1]):
                    color_idx = (out_ch * in_channels + in_ch) % len(colors)
                    marker_idx = (kh * kernel_size[1] + kw) % len(markers)
                    linestyle_idx = (out_ch + in_ch + kh + kw) % len(linestyles)
                    
                    label = f'out_{out_ch}_in_{in_ch}_k_{kh}_{kw}'
                    plt.plot(updated_positions.numpy(), 
                              updated_weights_reshaped[out_ch, in_ch, :, kh, kw].numpy(), 
                              color=colors[color_idx],
                              marker=markers[marker_idx],
                              linestyle=linestyles[linestyle_idx],
                              label=label,
                              linewidth=2,
                              markersize=8,
                              alpha=0.7)
    
    plt.title('Convolutional Layer After move_smoothest')
    plt.xlabel('Position')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig('after_move_smoothest.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    
    print("Visualization after move_smoothest saved to 'after_move_smoothest.png'")

if __name__ == "__main__":
    visualize_positions_and_weights()

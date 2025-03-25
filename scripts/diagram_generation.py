import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from non_uniform_piecewise_layers import AdaptivePiecewiseLinear
from matplotlib.colors import LinearSegmentedColormap

def create_test_function():
    """Create a test function for visualization."""
    # Define a simple function to approximate
    def func(x):
        return torch.cos(1/(torch.abs(x)+0.075))
    
    # Create input data
    x = torch.linspace(-1, 1, 1000).reshape(-1, 1)
    y = func(x)
    
    return x, y

def initialize_model(num_points=5):
    """Initialize the AdaptivePiecewiseLinear model."""
    model = AdaptivePiecewiseLinear(
        num_inputs=1,
        num_outputs=1,
        num_points=num_points
    )
    
    # Set initial values to approximate our test function
    x, y = create_test_function()
    with torch.no_grad():
        # Initialize positions uniformly
        positions = torch.linspace(-1, 1, num_points).reshape(1, 1, -1)
        model.positions = positions
        
        # Initialize values based on the test function
        values = torch.zeros(1, 1, num_points)
        for i in range(num_points):
            pos = positions[0, 0, i].item()
            idx = torch.argmin(torch.abs(x - pos))
            values[0, 0, i] = y[idx]
        
        model.values = nn.Parameter(values)
    
    return model

def evaluate_model(model, x):
    """Evaluate the model on the given input."""
    with torch.no_grad():
        return model(x)

def visualize_move_smoothest(model, x, y, output_path=None):
    """
    Visualize how the move_smoothest algorithm works.
    
    Args:
        model: The AdaptivePiecewiseLinear model
        x: Input tensor
        y: Target tensor
        output_path: Path to save the visualization
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), dpi=100)
    
    # 1. Original function approximation
    with torch.no_grad():
        y_pred_original = model(x)
    
    # Get positions and values
    positions_original = model.positions[0, 0].clone()
    values_original = model.values[0, 0].clone()
    
    # Plot original function and approximation
    axes[0].plot(x.detach().numpy(), y.detach().numpy(), 'b-', label='True Function', alpha=0.7)
    axes[0].plot(x.detach().numpy(), y_pred_original.detach().numpy(), 'r--', label='Piecewise Linear Approximation')
    axes[0].scatter(positions_original.detach().numpy(), values_original.detach().numpy(), c='g', s=100, label='Control Points')
    axes[0].set_title('Original Function Approximation')
    axes[0].legend()
    # Remove axis labels and grid
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # 2. Compute removal errors and identify smoothest point
    with torch.no_grad():
        # Compute removal errors
        errors, indices = model.compute_removal_errors()
        
        # Find the index of the point with minimum error (smoothest point)
        min_error_idx = torch.argmin(errors[0, 0])
        min_error = errors[0, 0][min_error_idx]
        smoothest_point_idx = indices[0, 0][min_error_idx]
        
        # Find the index of the point with maximum error
        max_error_idx = torch.argmax(errors[0, 0])
        max_error = errors[0, 0][max_error_idx]
        max_error_point_idx = indices[0, 0][max_error_idx]
        
        # Create a copy of the model with the smoothest point removed
        model_removed = AdaptivePiecewiseLinear(1, 1, model.num_points - 1)
        
        # Create mask for all points except the smoothest one
        keep_mask = torch.ones(model.num_points, dtype=torch.bool)
        keep_mask[smoothest_point_idx] = False
        
        # Copy positions and values, excluding the smoothest point
        model_removed.positions = model.positions[:, :, keep_mask].clone()
        model_removed.values = nn.Parameter(model.values[:, :, keep_mask].clone())
        
        # Evaluate the model with the point removed
        y_pred_removed = model_removed(x)
        
        # Create a copy of the model with the max error point removed
        model_max_removed = AdaptivePiecewiseLinear(1, 1, model.num_points - 1)
        
        # Create mask for all points except the max error one
        max_keep_mask = torch.ones(model.num_points, dtype=torch.bool)
        max_keep_mask[max_error_point_idx] = False
        
        # Copy positions and values, excluding the max error point
        model_max_removed.positions = model.positions[:, :, max_keep_mask].clone()
        model_max_removed.values = nn.Parameter(model.values[:, :, max_keep_mask].clone())
        
        # Evaluate the model with the max error point removed
        y_pred_max_removed = model_max_removed(x)
    
    # Plot function with smoothest point removed
    axes[1].plot(x.detach().numpy(), y.detach().numpy(), 'b-', label='True Function', alpha=0.7)
    axes[1].plot(x.detach().numpy(), y_pred_original.detach().numpy(), 'r--', label='Original Approximation', alpha=0.4)
    axes[1].plot(x.detach().numpy(), y_pred_removed.detach().numpy(), 'g--', label='Approximation with Min Error Point Removed')
    axes[1].plot(x.detach().numpy(), y_pred_max_removed.detach().numpy(), 'c--', label='Approximation with Max Error Point Removed')
    
    # Plot all original points
    axes[1].scatter(positions_original.detach().numpy(), values_original.detach().numpy(), c='gray', s=80, alpha=0.5, label='Original Points')
    
    # Highlight the removed points
    removed_pos = positions_original[smoothest_point_idx].item()
    removed_val = values_original[smoothest_point_idx].item()
    axes[1].scatter([removed_pos], [removed_val], c='r', s=150, label=f'Min Error Point (Error: {min_error:.4f})')
    
    # Highlight the max error point
    max_error_pos = positions_original[max_error_point_idx].item()
    max_error_val = values_original[max_error_point_idx].item()
    axes[1].scatter([max_error_pos], [max_error_val], c='orange', s=150, label=f'Max Error Point (Error: {max_error:.4f})')
    
    # Shade the area between the curves to show removal error for min error point
    x_np = x.detach().numpy().flatten()
    y_original = y_pred_original.detach().numpy().flatten()
    y_removed = y_pred_removed.detach().numpy().flatten()
    y_max_removed = y_pred_max_removed.detach().numpy().flatten()
    
    # Create custom colormaps
    colors_blue = [(0.8, 0.8, 1.0), (0.5, 0.5, 1.0)]
    colors_orange = [(1.0, 0.9, 0.8), (1.0, 0.7, 0.4)]
    cmap_blue = LinearSegmentedColormap.from_list('custom_blue', colors_blue, N=100)
    cmap_orange = LinearSegmentedColormap.from_list('custom_orange', colors_orange, N=100)
    
    # Fill between for min error point (blue)
    axes[1].fill_between(x_np, y_original, y_removed, where=(y_original != y_removed), 
                         alpha=0.3, color='blue', label='Min Error Removal Error')
    
    # Fill between for max error point (orange)
    axes[1].fill_between(x_np, y_original, y_max_removed, where=(y_original != y_max_removed), 
                         alpha=0.3, color='orange', label='Max Error Removal Error')
    
    axes[1].set_title('Function Approximation with Smoothest Point Removed')
    axes[1].legend()
    # Remove axis labels and grid
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # 3. Apply move_smoothest to get the final result
    with torch.no_grad():
        # Make a copy of the original model
        model_final = AdaptivePiecewiseLinear(1, 1, model.num_points)
        model_final.positions = model.positions.clone()
        model_final.values = nn.Parameter(model.values.clone())
        
        # Apply move_smoothest
        moved_pairs, total_pairs = model_final.move_smoothest()
        
        # Evaluate the final model
        y_pred_final = model_final(x)
    
    # Plot the final function approximation
    axes[2].plot(x.detach().numpy(), y.detach().numpy(), 'b-', label='True Function', alpha=0.7)
    axes[2].plot(x.detach().numpy(), y_pred_original.detach().numpy(), 'r--', label='Original Approximation', alpha=0.4)
    axes[2].plot(x.detach().numpy(), y_pred_final.detach().numpy(), 'm--', label='Final Approximation (After move_smoothest)')
    
    # Plot original points
    axes[2].scatter(positions_original.detach().numpy(), values_original.detach().numpy(), c='gray', s=80, alpha=0.5, label='Original Points')
    
    # Highlight the removed point
    axes[2].scatter([removed_pos], [removed_val], c='r', s=150, label=f'Removed Point (Min Error: {min_error:.4f})')
    
    # Highlight the new point (the one that's in final but not in original)
    final_positions = model_final.positions[0, 0].detach().numpy()
    final_values = model_final.values[0, 0].detach().numpy()
    
    # Find the new point by comparing original and final positions
    original_pos_set = set(positions_original.detach().numpy().tolist())
    final_pos_set = set(final_positions.tolist())
    new_pos_set = final_pos_set - original_pos_set
    
    if new_pos_set:
        new_pos = list(new_pos_set)[0]
        new_pos_idx = np.where(final_positions == new_pos)[0][0]
        new_val = final_values[new_pos_idx]
        
        # Highlight the point with max error (where we inserted nearby)
        max_error_pos = positions_original[max_error_point_idx].item()
        max_error_val = values_original[max_error_point_idx].item()
        
        axes[2].scatter([max_error_pos], [max_error_val], c='orange', s=150, 
                        label=f'Max Error Point (Error: {max_error:.4f})')
        
        # Highlight the new point
        axes[2].scatter([new_pos], [new_val], c='g', s=150, label='Newly Inserted Point')
    
    axes[2].set_title('Final Function Approximation After move_smoothest')
    axes[2].legend()
    # Remove axis labels and grid
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # Add more space between subplots
    plt.tight_layout(pad=3.0)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate diagrams for AdaptivePiecewiseLinear layer')
    parser.add_argument('--output', type=str, default='move_smoothest_visualization.png',
                        help='Output file path for the visualization')
    parser.add_argument('--num_points', type=int, default=7,
                        help='Number of initial points in the piecewise linear function')
    args = parser.parse_args()
    
    # Create test data
    x, y = create_test_function()
    
    # Initialize model
    model = initialize_model(args.num_points)
    
    # Visualize move_smoothest algorithm
    visualize_move_smoothest(model, x, y, args.output)

if __name__ == "__main__":
    main()
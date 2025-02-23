import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_piecewise_functions(layer, num_pixels_y=256, save_path=None):
    """
    Create a heatmap visualization of all piecewise functions in an AdaptivePiecewiseLinear layer.
    Functions are grouped by output neuron. This implementation is vectorized for speed.
    Each column represents an independent piecewise function with no interpolation between columns.
    The y-axis shows the input values, and the x-axis shows different functions grouped by output neuron.
    
    Args:
        layer: AdaptivePiecewiseLinear layer instance
        num_pixels_y: Number of pixels in the y direction (vertical resolution)
        save_path: If provided, save the figure to this path instead of displaying it
        
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Get layer dimensions
    n_inputs = layer.num_inputs
    n_outputs = layer.num_outputs
    total_functions = n_inputs * n_outputs
    
    # Create evaluation points
    y_points = torch.linspace(layer.position_min, layer.position_max, num_pixels_y)
    
    with torch.no_grad():
        # Initialize output data
        data = torch.zeros(n_outputs, n_inputs, num_pixels_y)
        
        # Process each function separately to handle searchsorted correctly
        for j in range(n_outputs):
            for i in range(n_inputs):
                # Get positions and values for this function
                pos = layer.positions[i, j]  # [num_points]
                vals = layer.values[i, j]  # [num_points]
                
                # Find indices of segments for each y point
                indices = torch.searchsorted(pos, y_points)  # [num_pixels_y]
                indices = torch.clamp(indices, 1, pos.size(0))
                
                # Get left and right positions/values for interpolation
                left_idx = indices - 1
                x0 = pos[left_idx]
                x1 = pos[indices.clamp(0, pos.size(0)-1)]
                y0 = vals[left_idx]
                y1 = vals[indices.clamp(0, pos.size(0)-1)]
                
                # Linear interpolation
                slope = (y1 - y0) / (x1 - x0).clamp(min=1e-6)
                data[j, i] = y0 + slope * (y_points - x0)
    
    # Convert to numpy and reshape for visualization
    # Reshape to [total_functions, num_pixels_y] and transpose to get [num_pixels_y, total_functions]
    data = data.reshape(total_functions, num_pixels_y).T.numpy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, aspect='auto', interpolation='none',
                   extent=[0, total_functions, layer.position_max, layer.position_min],  # Flip y-axis to have -1 at bottom
                   cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Function Value')
    
    # Add labels and title
    ax.set_xlabel('Function Index (grouped by output neuron)')
    ax.set_ylabel('Input Value')
    ax.set_title('Piecewise Linear Functions Heatmap')
    
    # Add vertical lines to separate output neurons (only if more than one output)
    if n_outputs > 1:
        for i in range(1, n_outputs):
            x = i * n_inputs  # Place lines at integer boundaries between groups
            ax.axvline(x=x, color='red', linestyle='--', alpha=0.5)
    
    # Add text labels for output neurons
    for i in range(n_outputs):
        x = i * n_inputs + n_inputs/2  # Center of each group
        ax.text(x, layer.position_max, f'Out {i}', 
                horizontalalignment='center', verticalalignment='bottom')
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig, ax
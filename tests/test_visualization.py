import os
import torch
import pytest
import numpy as np
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear
from non_uniform_piecewise_layers.visualization import visualize_piecewise_functions

# Create test output directory if it doesn't exist
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_output')
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

def test_visualize_piecewise_functions():
    # Create a small test layer
    n_inputs = 3
    n_outputs = 2
    n_points = 20
    layer = AdaptivePiecewiseLinear(
        num_inputs=n_inputs,
        num_outputs=n_outputs,
        num_points=n_points,
        position_range=(-1, 1)
    )
    
    # Test with different resolutions
    for num_pixels_y in [32, 64, 256]:
        output_path = os.path.join(TEST_OUTPUT_DIR, f'test_visualization_{num_pixels_y}px.png')
        
        # Test visualization with saving
        fig, ax = visualize_piecewise_functions(
            layer,
            num_pixels_y=num_pixels_y,
            save_path=output_path
        )
        
        # Check that the file was created
        assert os.path.exists(output_path)
        
        # Check the figure dimensions
        assert fig.get_size_inches()[1] == 8  # Height should be 8 inches
        
        # Check that the plot has the correct number of functions
        total_functions = n_inputs * n_outputs
        assert len(ax.get_xticks()) >= total_functions
        
        # Check that we have the correct number of vertical separator lines
        # (n_outputs - 1) separators since we don't need a line at the start
        lines = [line for line in ax.get_lines() if line.get_linestyle() == '--']
        assert len(lines) == n_outputs - 1
        
        # Check line positions and color
        for i, line in enumerate(lines, 1):
            assert line.get_xdata()[0] == i * n_inputs  # Lines should be at integer positions
            assert line.get_color() == 'red'  # Lines should be red
        
        # Check that text labels for outputs exist and are centered in their groups
        texts = [t for t in ax.texts if t.get_text().startswith('Out')]
        assert len(texts) == n_outputs
        for i, text in enumerate(texts):
            assert text.get_position()[0] == i * n_inputs + n_inputs/2  # Text should be centered
        
        # Check y-axis orientation
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # y-axis should be inverted (-1 at bottom, 1 at top)

def test_visualize_edge_cases():
    # Test with minimum values
    layer_min = AdaptivePiecewiseLinear(
        num_inputs=1,
        num_outputs=1,
        num_points=20,
        position_range=(-1, 1)
    )
    
    output_path = os.path.join(TEST_OUTPUT_DIR, 'test_min.png')
    fig, ax = visualize_piecewise_functions(
        layer_min,
        num_pixels_y=32,
        save_path=output_path
    )
    assert os.path.exists(output_path)
    
    # With single output, should have no separator lines
    lines = [line for line in ax.get_lines() if line.get_linestyle() == '--']
    assert len(lines) == 0
    
    # Test with larger values
    layer_large = AdaptivePiecewiseLinear(
        num_inputs=10,
        num_outputs=5,
        num_points=20,
        position_range=(-2, 2)
    )
    
    output_path = os.path.join(TEST_OUTPUT_DIR, 'test_large.png')
    fig, ax = visualize_piecewise_functions(
        layer_large,
        num_pixels_y=128,
        save_path=output_path
    )
    assert os.path.exists(output_path)
    
    # Should have 4 separator lines for 5 outputs
    lines = [line for line in ax.get_lines() if line.get_linestyle() == '--']
    assert len(lines) == 4
    # Check line positions
    for i, line in enumerate(lines, 1):
        assert line.get_xdata()[0] == i * 10  # Lines at multiples of n_inputs (10)
        assert line.get_color() == 'red'

def test_visualization_data_range():
    # Create a layer with known values
    layer = AdaptivePiecewiseLinear(
        num_inputs=2,
        num_outputs=2,
        num_points=3,
        position_range=(-1, 1)
    )
    
    # Set specific values for testing
    with torch.no_grad():
        # Set positions to [-1, 0, 1] for all functions
        layer.positions[:] = torch.tensor([-1., 0., 1.]).view(1, 1, -1)
        
        # Set values to create simple linear functions
        layer.values[0, 0] = torch.tensor([-1., 0., 1.])  # y = x
        layer.values[0, 1] = torch.tensor([1., 0., -1.])  # y = -x
        layer.values[1, 0] = torch.tensor([0., 0., 0.])   # y = 0
        layer.values[1, 1] = torch.tensor([2., 2., 2.])   # y = 2
    
    output_path = os.path.join(TEST_OUTPUT_DIR, 'test_known_functions.png')
    fig, ax = visualize_piecewise_functions(
        layer,
        num_pixels_y=64,
        save_path=output_path
    )
    
    # Get the plotted data
    image = ax.get_images()[0]
    data = image.get_array()
    
    # Check data dimensions
    assert data.shape == (64, 4)  # [num_pixels_y, total_functions]
    
    # Get the actual y values from the plot
    y_min, y_max = ax.get_ylim()
    y_values = np.linspace(y_max, y_min, 64)  # y-axis is inverted
    
    # Check some known values at specific y positions
    y_idx_neg1 = np.abs(y_values - (-1)).argmin()  # Find index closest to y = -1
    y_idx_pos1 = np.abs(y_values - 1).argmin()     # Find index closest to y = 1
    
    # For y = x function (first column)
    assert abs(data[y_idx_neg1, 0] + 1) < 1e-6  # At y = -1, f(y) should be -1
    assert abs(data[y_idx_pos1, 0] - 1) < 1e-6  # At y = 1, f(y) should be 1
    
    # For y = 2 function (last column)
    assert abs(data[y_idx_neg1, -1] - 2) < 1e-6  # Should be 2 everywhere
    assert abs(data[y_idx_pos1, -1] - 2) < 1e-6

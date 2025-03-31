import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.efficient_adaptive_piecewise_conv import EfficientAdaptivePiecewiseConv2d
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear

class HypersphereConvNet(nn.Module):
    """
    Neural network to approximate a hypersphere function using convolutional layers.
    """
    def __init__(
        self,
        num_points=5,
        position_range=(-1, 1),
        weight_init="random"
    ):
        """
        Initialize the hypersphere network.
        
        Args:
            num_points (int): Number of points in the piecewise linear function
            position_range (tuple): Range of positions for the piecewise linear function
            weight_init (str): Weight initialization method
        """
        super().__init__()
        
        # Create a convolutional layer to process the 2x2 input
        # Input: 1 channel, 2x2 grid
        # Output: 1 channel, 1x1 grid
        self.conv = EfficientAdaptivePiecewiseConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,  # 2x2 kernel to process the entire input
            stride=1,
            padding=0,
            num_points=num_points,
            position_range=position_range,
            position_init="random",  # Use random position initialization
            weight_init=weight_init
        )
        
        # Create a single adaptive piecewise linear layer to process the output of the convolutional layer
        # Input: 1 value (output of conv layer)
        # Output: 1 value
        self.linear = AdaptivePiecewiseLinear(
            num_inputs=1,
            num_outputs=1,
            num_points=num_points,
            position_range=position_range,
            anti_periodic=False,
            position_init="random"  # Use random position initialization
        )
    
    def forward(self, x):
        # Apply convolutional layer
        # x shape: [batch_size, 1, 2, 2]
        conv_out = self.conv(x)
        
        # Flatten the output
        # conv_out shape: [batch_size, 1, 1, 1]
        flat_out = conv_out.view(conv_out.size(0), -1)
        
        # Apply linear layer
        # flat_out shape: [batch_size, 1]
        output = self.linear(flat_out)
        
        return output

def hypersphere_function(inputs):
    """
    Compute the hypersphere function for 4D inputs.
    
    Args:
        inputs (torch.Tensor): Input tensor of shape [batch_size, 4]
        
    Returns:
        torch.Tensor: Output tensor of shape [batch_size, 1]
    """
    # Compute the sum of squares
    sum_squares = torch.sum(inputs**2, dim=1, keepdim=True)
    
    # Compute the hypersphere function: 1 if inside the unit sphere, 0 otherwise
    outputs = (sum_squares <= 1.0).float()
    
    return outputs.unsqueeze(1)

def generate_grid_data(grid_size=10, range_min=-1.0, range_max=1.0):
    """
    Generate a grid of points in 4D space.
    
    Args:
        grid_size (int): Number of points along each dimension
        range_min (float): Minimum value for each dimension
        range_max (float): Maximum value for each dimension
    
    Returns:
        torch.Tensor: Grid points of shape [grid_size^4, 4]
        torch.Tensor: Reshaped grid for visualization [grid_size, grid_size, grid_size, grid_size, 4]
    """
    # Create 1D grid for each dimension
    grid_1d = torch.linspace(range_min, range_max, grid_size)
    
    # Create 4D grid
    grid_points = torch.cartesian_prod(grid_1d, grid_1d, grid_1d, grid_1d)
    
    # Reshape for visualization
    grid_reshaped = grid_points.reshape(grid_size, grid_size, grid_size, grid_size, 4)
    
    return grid_points, grid_reshaped

def save_piecewise_plots(model, epoch, dpi=100):
    """Save plots showing the piecewise approximations for each layer"""
    # Create figure for convolutional layer
    plt.figure(figsize=(12, 8))
    
    # Get weights for visualization
    weights = model.conv.conv.weight.data.cpu()
    
    # Reshape weights to [out_channels, in_channels, num_points, kernel_height, kernel_width]
    out_channels = weights.size(0)
    in_channels = model.conv.in_channels
    num_points = model.conv.expansion.positions.size(0)
    kernel_size = model.conv.kernel_size
    weights_reshaped = weights.view(out_channels, in_channels, num_points, *kernel_size)
    
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
                    
                    # Get the positions for this kernel element
                    if hasattr(model.conv, "custom_positions") and model.conv.custom_positions is not None:
                        # Use custom positions for this specific kernel element
                        positions = model.conv.custom_positions[out_ch, in_ch, kh, kw].cpu().numpy()
                    else:
                        # Use the uniform positions from the expansion layer
                        positions = model.conv.expansion.positions.data.cpu().numpy()
                    
                    label = f'out_{out_ch}_in_{in_ch}_k_{kh}_{kw}'
                    plt.plot(positions, 
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
    plt.savefig(f'conv_weights_epoch_{epoch:04d}.png', dpi=dpi, bbox_inches='tight')
    plt.close('all')
    
    # Create figure for the linear layer
    plt.figure(figsize=(12, 5))
    
    # Get positions and values
    positions = model.linear.positions.data.cpu()
    values = model.linear.values.data.cpu()
    
    # Plot each input-output mapping
    for in_dim in range(positions.shape[0]):  # For each input dimension
        for out_dim in range(positions.shape[1]):  # For each output dimension
            pos = positions[in_dim, out_dim].numpy()
            val = values[in_dim, out_dim].numpy()
            plt.plot(pos, val, 'o-', label=f'in_{in_dim}_out_{out_dim}')
    
    plt.title('Linear Layer Piecewise Approximation')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'linear_weights_epoch_{epoch:04d}.png', dpi=dpi, bbox_inches='tight')
    plt.close('all')

def train_hypersphere(cfg):
    """
    Train a neural network to approximate the hypersphere function.
    
    Args:
        cfg (DictConfig): Configuration object from Hydra
    """
    # Set random seed for reproducibility
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    
    # Create directories for outputs
    os.makedirs(cfg.output.dir, exist_ok=True)
    
    # Create the model
    model = HypersphereConvNet(
        num_points=cfg.model.num_points,
        position_range=tuple(cfg.model.position_range),
        weight_init=cfg.model.weight_init
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Generate training data
    train_inputs, _ = generate_grid_data(
        grid_size=cfg.data.grid_size,
        range_min=cfg.data.range_min,
        range_max=cfg.data.range_max
    )
    
    # Compute target outputs
    train_targets = hypersphere_function(train_inputs)
    
    # Reshape inputs for convolutional layer
    # Each 4D point becomes a 2x2 grid with 1 channel
    train_inputs_reshaped = train_inputs.view(-1, 1, 2, 2)
    
    # Training loop
    for epoch in range(cfg.training.max_epochs):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_inputs_reshaped)
        
        # Compute loss
        loss = criterion(outputs, train_targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % cfg.logging.print_every == 0:
            logging.info(f'Epoch {epoch+1}/{cfg.training.max_epochs}, Loss: {loss.item():.6f}')
        
        # Save plots
        if (epoch + 1) % cfg.visualization.save_every == 0 or epoch == 0:
            logging.info(f'Saving plots for epoch {epoch+1}')
            save_piecewise_plots(model, epoch, dpi=cfg.visualization.dpi)
            
        # Move points if enabled
        if cfg.model.move_points and (epoch + 1) % cfg.model.move_every == 0:
            logging.info(f'Moving points for epoch {epoch+1}')
            model.conv.move_smoothest()
            model.linear.move_smoothest()
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(cfg.output.dir, 'hypersphere_model.pt'))
    
    # Save the final plots
    save_piecewise_plots(model, cfg.training.max_epochs-1, dpi=cfg.visualization.dpi)
    
    logging.info('Training completed!')

@hydra.main(config_path="../config", config_name="hypersphere_conv")
def main(cfg: DictConfig):
    """
    Main function to train the hypersphere model.
    
    Args:
        cfg (DictConfig): Configuration object from Hydra
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Log the configuration
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Train the model
    train_hypersphere(cfg)

if __name__ == "__main__":
    main()

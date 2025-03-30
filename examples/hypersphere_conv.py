import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from non_uniform_piecewise_layers import AdaptivePiecewiseLinear, EfficientAdaptivePiecewiseConv2d
from non_uniform_piecewise_layers.utils import largest_error
from lion_pytorch import Lion
import imageio
import logging
from itertools import combinations
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class HypersphereConvNet(nn.Module):
    """
    A network with a single convolutional layer followed by an MLP.
    The convolutional layer uses EfficientAdaptivePiecewiseConv2d.
    """
    def __init__(self, num_points=5, position_range=(-1, 1), weight_init="linear"):
        super().__init__()
        
        # Create a single convolutional layer
        # Input: 1 channel, 2x2 grid (4 values total)
        # Output: 1 channel
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
        # output shape: [batch_size, 1]
        output = self.linear(flat_out)
        
        return output

def generate_hypersphere_data(inputs, radius=0.5, center=None):
    """
    Generate a hypersphere in 4D space.
    Points inside the hypersphere get 0.5, outside get -0.5.
    
    Args:
        inputs (torch.Tensor): Input tensor of shape [batch_size, 4]
        radius (float): Radius of the hypersphere
        center (list, optional): Center of the hypersphere. If None, center is at origin.
    
    Returns:
        torch.Tensor: Output tensor of shape [batch_size, 1]
    """
    if center is None:
        center = [0.0, 0.0, 0.0, 0.0]
    
    # Calculate squared distance from each point to hypersphere center
    squared_distances = torch.zeros(inputs.size(0), device=inputs.device)
    for i in range(4):
        squared_distances += (inputs[:, i] - center[i])**2
    
    # Points inside hypersphere (distance <= radius) get 0.5, outside get -0.5
    outputs = torch.where(squared_distances <= radius**2, 0.5, -0.5)
    
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

def save_piecewise_plots(model, epoch, dpi=100, initialize_weights=False):
    """Save plots showing the piecewise approximations for each layer"""
    # Create figure for convolutional layer
    plt.figure(figsize=(12, 8))
    
    # Get positions and reshape weights for visualization
    positions = model.conv.expansion.positions.data.cpu()
    weights = model.conv.conv.weight.data.cpu()
    
    # Reshape weights to [out_channels, in_channels, num_points, kernel_height, kernel_width]
    out_channels = weights.size(0)
    in_channels = model.conv.in_channels
    num_points = positions.size(0)
    kernel_size = model.conv.kernel_size
    weights_reshaped = weights.view(out_channels, in_channels, num_points, *kernel_size)
    
    # If requested, initialize the convolutional layer with different patterns for better visualization
    if initialize_weights:
        with torch.no_grad():
            for out_ch in range(out_channels):
                for in_ch in range(in_channels):
                    for kh in range(kernel_size[0]):
                        for kw in range(kernel_size[1]):
                            # Create a unique pattern for each kernel element
                            # Add a small random offset to each weight to ensure they're visually distinct
                            offset = 0.2 * (out_ch + in_ch + kh + kw)
                            scale = 0.1 * (1 + out_ch + in_ch * 0.5 + kh * 0.3 + kw * 0.2)
                            for p_idx in range(num_points):
                                # Create a sinusoidal pattern with different frequencies
                                pos = positions[p_idx].item()
                                model.conv.conv.weight.data[out_ch, in_ch * num_points + p_idx, kh, kw] = \
                                    scale * torch.sin(torch.tensor(3.0 * pos + offset)) + offset
        
        # Get the updated weights after initialization
        weights = model.conv.conv.weight.data.cpu()
        weights_reshaped = weights.view(out_channels, in_channels, num_points, *kernel_size)
    
    # Plot positions vs weights for each kernel position
    for out_ch in range(out_channels):
        for in_ch in range(in_channels):
            for kh in range(kernel_size[0]):
                for kw in range(kernel_size[1]):
                    label = f'out_{out_ch}_in_{in_ch}_k_{kh}_{kw}'
                    plt.plot(positions.numpy(), 
                              weights_reshaped[out_ch, in_ch, :, kh, kw].numpy(), 
                              'o-', label=label)
    
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
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout(pad=3.0)
    plt.savefig(f'mlp_weights_epoch_{epoch:04d}.png', dpi=dpi, bbox_inches='tight')
    plt.close('all')
    
    # Clear any remaining figures and free memory
    plt.clf()
    plt.close('all')

def save_2d_slice_plots(model, grid_points, grid_size, epoch, loss, dpi=100):
    """
    Save 2D slice plots for visualization of the hypersphere.
    
    Args:
        model (nn.Module): The trained model
        grid_points (torch.Tensor): Grid points of shape [grid_size^4, 4]
        grid_size (int): Number of points along each dimension
        epoch (int): Current epoch number
        loss (float): Current loss value
        dpi (int): Resolution for saved images
    """
    # Get model predictions
    with torch.no_grad():
        # Reshape grid points to [batch_size, 1, 2, 2] for the convolutional layer
        inputs_reshaped = grid_points.view(-1, 1, 2, 2)
        predictions = model(inputs_reshaped).cpu().numpy().flatten()
    
    # Reshape predictions to 4D grid
    pred_grid = predictions.reshape(grid_size, grid_size, grid_size, grid_size)
    
    # Create 2D slices by fixing two dimensions
    # We'll create plots for all combinations of dimensions
    dim_pairs = list(combinations(range(4), 2))
    
    # Create a figure with subplots for each dimension pair
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (dim1, dim2) in enumerate(dim_pairs):
        # Fixed values for the other two dimensions (middle of the range)
        fixed_dims = [d for d in range(4) if d != dim1 and d != dim2]
        fixed_idx = grid_size // 2  # Middle index
        
        # Create slice indices
        slice_indices = [slice(None)] * 4
        for dim in fixed_dims:
            slice_indices[dim] = fixed_idx
        
        # Extract 2D slice
        slice_2d = pred_grid[tuple(slice_indices)]
        
        # Create coordinate grids for plotting
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Plot the 2D slice
        ax = axes[idx]
        levels = np.linspace(-0.75, 0.75, 21)
        cs = ax.contourf(X.numpy(), Y.numpy(), slice_2d, levels=levels, cmap='coolwarm', extend='both')
        ax.contour(X.numpy(), Y.numpy(), slice_2d, levels=[0], colors='k', linestyles='dashed')
        
        # Draw the actual circle for this 2D slice
        # Calculate the radius of the circle in this 2D slice
        # For a 4D hypersphere with radius R, the 2D slice at distance d from the center
        # has a radius of sqrt(R^2 - d^2)
        hypersphere_radius = 0.5
        distance_from_center = 0.0  # For the middle slice, distance is 0
        circle_radius = np.sqrt(max(0, hypersphere_radius**2 - distance_from_center**2))
        
        circle = plt.Circle((0, 0), circle_radius, fill=False, color='black', linewidth=2)
        ax.add_artist(circle)
        
        ax.set_title(f'Dimensions {dim1+1} vs {dim2+1}')
        ax.set_xlabel(f'Dimension {dim1+1}')
        ax.set_ylabel(f'Dimension {dim2+1}')
        ax.grid(True)
        ax.set_aspect('equal')
    
    # Add a color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cs, cax=cbar_ax)
    
    # Add overall title
    fig.suptitle(f'Hypersphere Classification - Epoch {epoch}, Loss: {loss:.4f}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f'hypersphere_2d_slices_{epoch:04d}.png', dpi=dpi, bbox_inches='tight')
    plt.close('all')
    
    # Clear any remaining figures and free memory
    plt.clf()
    plt.close('all')

def save_convergence_plot(losses, epochs, dpi=100):
    """Save a plot showing the convergence of loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.title('Convergence Plot')
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=dpi)
    plt.close('all')
    
    # Clear any remaining figures and free memory
    plt.clf()
    plt.close('all')

@hydra.main(version_base=None, config_path="config", config_name="hypersphere_conv")
def main(cfg: DictConfig):
    # Log some useful information
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory : {HydraConfig.get().run.dir}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create model
    model = HypersphereConvNet(
        num_points=cfg.model.num_points,
        position_range=tuple(cfg.model.position_range),
        weight_init=cfg.model.weight_init
    ).to(device)
    
    # Create optimizer
    if cfg.training.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.training.learning_rate)
    elif cfg.training.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    
    # Create loss function
    loss_function = nn.MSELoss()
    
    # Generate grid data for training and visualization
    grid_points, grid_reshaped = generate_grid_data(
        grid_size=cfg.data.grid_size,
        range_min=cfg.data.range_min,
        range_max=cfg.data.range_max
    )
    grid_points = grid_points.to(device)
    
    # Generate target data (hypersphere)
    targets = generate_hypersphere_data(
        grid_points, 
        radius=cfg.data.hypersphere.radius,
        center=cfg.data.hypersphere.center if hasattr(cfg.data.hypersphere, 'center') else None
    )
    targets = targets.to(device)
    
    # We'll collect filenames instead of keeping images in memory
    progress_images = []
    weights_conv_images = []
    weights_mlp_images = []
    
    # Training loop
    losses = []
    epochs = []
    
    for epoch in range(cfg.training.num_epochs):
        # Reshape grid points to [batch_size, 1, 2, 2] for the convolutional layer
        inputs_reshaped = grid_points.view(-1, 1, 2, 2)
        
        # Forward pass
        predictions = model(inputs_reshaped)
        loss = loss_function(predictions, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        epochs.append(epoch)
        
        # Adaptive point movement
        if epoch % cfg.training.adapt_interval == 0:
            if cfg.training.adapt == "move":
                # Move points in the convolutional layer
                # Note: EfficientAdaptivePiecewiseConv2d.move_smoothest returns a boolean
                # Set threshold to 0.0 to force movement even when errors are all close to zero
                conv_moved = model.conv.move_smoothest(
                    weighted=cfg.training.weighted, 
                    threshold=0.0  # Force movement by setting threshold to 0.0
                )
                logger.info(f'Conv layer: Points moved: {conv_moved}')
                
                # Move points in the linear layer
                # Note: AdaptivePiecewiseLinear.move_smoothest returns a tuple (moved_pairs, total_pairs)
                linear_moved, linear_total = model.linear.move_smoothest(
                    weighted=cfg.training.weighted, 
                    threshold=cfg.training.move_threshold
                )
                logger.info(f'Linear: Moved {linear_moved}/{linear_total} pairs ({linear_moved/max(1, linear_total)*100:.2f}%)')
                
                # Recreate optimizer after moving points
                if cfg.training.optimizer == "lion":
                    optimizer = Lion(model.parameters(), lr=cfg.training.learning_rate)
                elif cfg.training.optimizer == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
            elif cfg.training.adapt == "global_error":
                # Calculate error
                with torch.no_grad():
                    error = torch.abs(predictions - targets)
                    new_value = largest_error(error, inputs_reshaped)
                if new_value is not None:
                    logger.debug(f'New value: {new_value}')
                    model.remove_add(new_value)
                    # Recreate optimizer
                    if cfg.training.optimizer == "lion":
                        optimizer = Lion(model.parameters(), lr=cfg.training.learning_rate)
                    elif cfg.training.optimizer == "adam":
                        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
                    else:
                        optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
        
        # Save plots at specified intervals
        if epoch % cfg.visualization.plot_interval == 0 or epoch == cfg.training.num_epochs - 1:
            logger.info(f'Epoch {epoch}/{cfg.training.num_epochs}, Loss: {loss.item():.6f}')
            
            # Save 2D slice plots with lower DPI to save memory
            save_2d_slice_plots(model, grid_points, cfg.data.grid_size, epoch, loss.item(), dpi=cfg.visualization.dpi)
            progress_images.append(f'hypersphere_2d_slices_{epoch:04d}.png')
            
            # Save weights plots with lower DPI to save memory (initialize weights on first epoch for better visualization)
            save_piecewise_plots(model, epoch, dpi=cfg.visualization.dpi, initialize_weights=(epoch == 0))
            weights_conv_images.append(f'conv_weights_epoch_{epoch:04d}.png')
            weights_mlp_images.append(f'mlp_weights_epoch_{epoch:04d}.png')
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
    
    # Save convergence plot
    save_convergence_plot(losses, epochs, dpi=cfg.visualization.dpi)
    
    # Create GIFs from collected filenames
    logger.info("Creating GIFs from saved images...")
    
    # Create GIFs one by one to avoid memory issues
    with imageio.get_writer('hypersphere_progress.gif', mode='I', duration=cfg.visualization.gif_duration) as writer:
        for img_file in progress_images:
            try:
                image = imageio.imread(img_file)
                writer.append_data(image)
                # Free memory immediately
                del image
                import gc
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
    
    with imageio.get_writer('conv_weights.gif', mode='I', duration=cfg.visualization.gif_duration) as writer:
        for img_file in weights_conv_images:
            try:
                image = imageio.imread(img_file)
                writer.append_data(image)
                # Free memory immediately
                del image
                import gc
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
    
    with imageio.get_writer('mlp_weights.gif', mode='I', duration=cfg.visualization.gif_duration) as writer:
        for img_file in weights_mlp_images:
            try:
                image = imageio.imread(img_file)
                writer.append_data(image)
                # Free memory immediately
                del image
                import gc
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from non_uniform_piecewise_layers import AdaptivePiecewiseMLP
from lion_pytorch import Lion
import imageio
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def generate_circle_data(x, y, position='lower_left'):
    """Generate a circle at either the lower left or upper right position."""
    if position == 'lower_left':
        center_x, center_y = -0.5, -0.5
    else:  # upper_right
        center_x, center_y = 0.5, 0.5
    
    # Calculate distance from each point to circle center
    distances = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Points inside circle (radius=0.3) get 0.5, outside get -0.5
    outputs = torch.where(distances <= 0.3, 0.5, -0.5)
    return outputs.to(device)

def save_progress_plot(model, inputs, outputs, epoch, loss, position, grid_size):
    """Save a plot showing the current state of the approximation."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert inputs to numpy for plotting
    x_np = inputs[:, 0].reshape(grid_size, grid_size).cpu().numpy()
    y_np = inputs[:, 1].reshape(grid_size, grid_size).cpu().numpy()
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(inputs).reshape(grid_size, grid_size).cpu().numpy()
    
    # Plot the predictions using contour with fixed levels
    levels = np.linspace(-0.75, 0.75, 21)  # 21 fixed levels between -0.75 and 0.75
    cs = ax.contourf(x_np, y_np, predictions, levels=levels, cmap='coolwarm', extend='both')
    ax.contour(x_np, y_np, predictions, levels=[0], colors='k', linestyles='dashed')  # Decision boundary
    plt.colorbar(cs)
    
    # Draw the actual circle
    if position == 'lower_left':
        circle = plt.Circle((-0.5, -0.5), 0.3, fill=False, color='black', linewidth=2)
    else:
        circle = plt.Circle((0.5, 0.5), 0.3, fill=False, color='black', linewidth=2)
    ax.add_artist(circle)
    
    ax.set_title(f'Circle Classification - Epoch {epoch}\n'
                f'Position: {position}, Loss: {loss:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')  # Make sure circles look circular
    
    plt.tight_layout()
    plt.savefig(f'circle_{epoch:03d}.png')
    plt.close()

def save_piecewise_plots(model, epoch):
    """Save plots showing the piecewise approximations for each layer"""
    num_layers = len(model.layers)
    
    # Create figure with a subplot for each layer
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 5*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    # For each layer
    for layer_idx, (layer, ax) in enumerate(zip(model.layers, axes)):
        positions = layer.positions.data.cpu()
        values = layer.values.data.cpu()
        
        # Plot each input-output mapping (limited to first 10)
        max_lines = 10
        for in_dim in range(min(positions.shape[0], max_lines)):  # For each input dimension
            for out_dim in range(min(positions.shape[1], max_lines)):  # For each output dimension
                pos = positions[in_dim, out_dim].numpy()
                val = values[in_dim, out_dim].numpy()
                ax.plot(pos, val, 'o-', label=f'in_{in_dim}_out_{out_dim}')
        
        ax.set_title(f'Layer {layer_idx+1} Piecewise Approximations')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(f'weights_epoch_{epoch:04d}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_convergence_plot(losses, epochs):
    """Save a plot showing the convergence of loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, losses, 'b-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.grid(True)
    ax.set_title('Convergence Plot')
    plt.tight_layout()
    plt.savefig('convergence.png')
    plt.close()

def generate_optimizer(parameters, learning_rate):
    return Lion(parameters, lr=learning_rate)

@hydra.main(version_base=None, config_path="config", config_name="dynamic_circle")
def main(cfg: DictConfig):
    # Log some useful information
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory : {HydraConfig.get().run.dir}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create synthetic data
    x = torch.linspace(cfg.data.x_range[0], cfg.data.x_range[1], cfg.data.grid_size)
    y = torch.linspace(cfg.data.y_range[0], cfg.data.y_range[1], cfg.data.grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    inputs = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    
    # Create model and optimizer
    model = AdaptivePiecewiseMLP(
        width=cfg.model.width,
        num_points=cfg.model.num_points,
        position_range=tuple(cfg.model.position_range)
    ).to(device)
    
    # Create GIF writers
    progress_writer = imageio.get_writer('progress.gif', mode='I', duration=cfg.visualization.gif_duration)
    weights_writer = imageio.get_writer('weights.gif', mode='I', duration=cfg.visualization.gif_duration)
    
    optimizer = generate_optimizer(model.parameters(), cfg.training.learning_rate)
    loss_function = nn.MSELoss()
    losses = []
    epochs = []
    
    # Training loop
    for epoch in range(cfg.training.num_epochs):
        # Generate target data based on current epoch
        if epoch < cfg.training.switch_epoch:
            outputs = generate_circle_data(xx.flatten(), yy.flatten(), 'lower_left')
            position = 'lower_left'
        else:
            outputs = generate_circle_data(xx.flatten(), yy.flatten(), 'upper_right')
            position = 'upper_right'
        
        batched_out = outputs.unsqueeze(1)
        
        # Forward pass
        predictions = model(inputs)
        loss = loss_function(predictions, batched_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        epochs.append(epoch)
        
        error = torch.abs(predictions-batched_out)
        new_value = model.largest_error(error, inputs)
        if new_value is not None:
            success = model.remove_add(new_value)
            optimizer = generate_optimizer(model.parameters(), cfg.training.learning_rate)
        
        # Save progress plot at specified intervals
        if epoch % cfg.visualization.plot_interval == 0:
            # Save progress plot and add to GIF
            save_progress_plot(model, inputs, outputs, epoch, loss.item(), position, cfg.data.grid_size)
            progress_writer.append_data(imageio.imread(f'circle_{epoch:03d}.png'))
            
            # Save weights plot and add to GIF
            save_piecewise_plots(model, epoch)
            weights_writer.append_data(imageio.imread(f'weights_epoch_{epoch:04d}.png'))
            
            logger.info(f'Epoch {epoch}/{cfg.training.num_epochs}, Loss: {loss.item():.4f}')
    
    # Save convergence plot
    save_convergence_plot(losses, epochs)
    
    # Close the writers
    progress_writer.close()
    weights_writer.close()
    
    logger.info("GIFs created successfully!")

if __name__ == "__main__":
    main()

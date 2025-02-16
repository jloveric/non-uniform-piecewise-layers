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

def generate_square_wave(x, position='left', width=0.2, amplitude=0.75):
    """Generate a square wave at either the left or right position."""
    if position == 'left':
        center = -0.5
    else:  # right
        center = 0.5
    
    mask = torch.abs(x - center) < width/2
    y = torch.zeros_like(x)
    y[mask] = amplitude
    return y

def save_progress_plot(model, x, y, epoch, loss, position):
    """Save a plot showing the current state of the approximation and absolute error."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[2, 1])
    
    # Plot the results
    with torch.no_grad():
        y_pred = model(x)

    # Convert to numpy for plotting
    x_np = x.numpy()
    y_np = y.numpy()
    y_pred_np = y_pred.numpy()
    
    # Top subplot: Function approximation
    ax1.plot(x_np, y_np, 'b-', label='True Function', alpha=0.5)
    ax1.plot(x_np, y_pred_np, 'r--', label='MLP Approximation')

    # Plot the control points from the first layer
    positions = model.layers[0].positions.data[0, 0].numpy()
    values = model.layers[0].values.data[0, 0].numpy()
    ax1.scatter(positions, values, c='g', s=100, label='Control Points (First Layer)')

    ax1.set_title(f'Square Wave Approximation - Epoch {epoch}\n'
                f'Position: {position}, Loss: {loss:.4f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    
    # Bottom subplot: Absolute error
    abs_error = np.abs(y_np - y_pred_np)
    ax2.plot(x_np, abs_error, 'k-', label='Absolute Error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Absolute Error')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'square_wave_{epoch:03d}.png')
    plt.close()

def generate_optimizer(parameters, learning_rate):
    return Lion(parameters, lr=learning_rate)

@hydra.main(version_base=None, config_path="config", config_name="dynamic_square_wave")
def main(cfg: DictConfig):
    # Log some useful information
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory : {HydraConfig.get().run.dir}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create synthetic data
    x = torch.linspace(cfg.data.x_range[0], cfg.data.x_range[1], cfg.data.num_points).reshape(-1, 1)
    
    # Create model and optimizer
    model = AdaptivePiecewiseMLP(
        width=cfg.model.width,
        num_points=cfg.model.num_points,
        position_range=tuple(cfg.model.position_range)
    )
    
    optimizer = generate_optimizer(model.parameters(), cfg.training.learning_rate)
    
    # Prepare for creating a GIF
    images = []
    
    # Training loop
    for epoch in range(cfg.training.num_epochs):
        # Generate target data based on epoch
        position = 'left' if epoch < cfg.training.switch_epoch else 'right'
        y = generate_square_wave(
            x, 
            position=position,
            width=cfg.data.wave.width,
            amplitude=cfg.data.wave.amplitude
        )
        
        # Forward pass
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Call remove_add after each epoch
        error = torch.abs(y_pred-y)
        new_value = model.largest_error(error, x)
        if new_value is not None:
            logger.debug(f'New value: {new_value}')
            model.remove_add(new_value)
            optimizer = generate_optimizer(model.parameters(), cfg.training.learning_rate)
        
        # Save progress plot at specified intervals
        if epoch % cfg.visualization.plot_interval == 0:
            save_progress_plot(model, x, y, epoch, loss.item(), position)
            images.append(imageio.imread(f'square_wave_{epoch:03d}.png'))
            logger.info(f'Epoch {epoch}/{cfg.training.num_epochs}, Loss: {loss.item():.6f}, Position: {position}')
    
    # Save the images as a GIF
    imageio.mimsave('dynamic_square_wave.gif', images, duration=cfg.visualization.gif_duration)
    logger.info("GIF 'dynamic_square_wave.gif' created successfully!")

if __name__ == "__main__":
    main()

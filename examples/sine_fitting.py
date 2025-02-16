import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from non_uniform_piecewise_layers import NonUniformPiecewiseLinear
from non_uniform_piecewise_layers import AdaptivePiecewiseLinear
from lion_pytorch import Lion
from torch.autograd.functional import jacobian
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging
import imageio.v2 as imageio

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def save_progress_plot(model, x, y, epoch, loss, strategy):
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
    ax1.plot(x_np, y_pred_np, 'r--', label='Piecewise Linear Approximation')

    # Plot the control points
    positions = model.piecewise.positions.data[0, 0].numpy()
    values = model.piecewise.values.data[0, 0].numpy()
    ax1.scatter(positions, values, c='g', s=100, label='Control Points')

    ax1.set_title(f'Function Approximation - Epoch {epoch}\n'
                f'Points: {len(positions)}, Loss: {loss:.4f}, Strategy: {strategy}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    
    # Bottom subplot: Absolute error
    abs_error = np.abs(y_np - y_pred_np)
    ax2.plot(x_np, abs_error, 'k-', label='Absolute Error')
    ax2.scatter(positions, np.zeros_like(positions), c='g', s=100, 
               label='Control Points', zorder=3)  # zorder to ensure points are on top
    ax2.set_title('Absolute Error')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error|')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()

    plt.savefig(os.path.join(HydraConfig.get().run.dir, f'approximation_epoch_{epoch:04d}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create a simple model with our non-uniform piecewise linear layer
class SineApproximator(nn.Module):
    def __init__(self, num_points, model_type="adaptive"):
        super().__init__()

        if model_type == "adaptive":
            self.piecewise = AdaptivePiecewiseLinear(
                num_inputs=1,
                num_outputs=1,
                num_points=num_points
            )
        else:  # non_uniform
            self.piecewise = NonUniformPiecewiseLinear(
                num_inputs=1,
                num_outputs=1,
                num_points=num_points
            )
    
    def forward(self, x):
        return self.piecewise(x)

    def compute_abs_grads(self, x):
        return self.piecewise.compute_abs_grads(x)

    def largest_error(self, error, x):
        return self.piecewise.largest_error(error, x)

    def insert_points(self, x):
        return self.piecewise.insert_points(x)
    
    def insert_nearby_point(self, x):
        return self.piecewise.insert_nearby_point(x)

def generate_optimizer(parameters, learning_rate):
    return Lion(parameters, lr=learning_rate)

def save_final_plots(model, x, y, losses, num_points_history):
    """Save the final plots showing training progress and results."""
    plt.figure(figsize=(15, 15))

    # Plot training loss
    plt.subplot(3, 1, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)

    # Plot number of points over time
    plt.subplot(3, 1, 2)
    plt.plot(num_points_history)
    plt.title('Number of Control Points')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Points')
    plt.grid(True)

    # Plot the final results
    plt.subplot(3, 1, 3)
    with torch.no_grad():
        y_pred = model(x)

    # Convert to numpy for plotting
    x_np = x.numpy()
    y_np = y.numpy()
    y_pred_np = y_pred.numpy()

    plt.plot(x_np, y_np, 'b-', label='True Function')
    plt.plot(x_np, y_pred_np, 'r--', label='Piecewise Linear Approximation')

    # Plot the learned points
    positions = model.piecewise.positions.data[0, 0].numpy()
    values = model.piecewise.values.data[0, 0].numpy()
    plt.scatter(positions, values, c='g', label='Control Points')

    plt.title('Final Function Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(HydraConfig.get().run.dir, 'final_approximation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return positions, values

@hydra.main(version_base=None, config_path="config", config_name="sine_fitting")
def main(cfg: DictConfig):
    # Log some useful information
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory : {HydraConfig.get().run.dir}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Ensure output directory exists
    os.makedirs(HydraConfig.get().run.dir, exist_ok=True)

    # Create synthetic sine wave data
    x = torch.linspace(cfg.data.x_range[0], cfg.data.x_range[1], cfg.data.num_points).reshape(-1, 1)
    y = torch.cos(1/(torch.abs(x)+0.05))

    # Create model and optimizer
    model = SineApproximator(cfg.model.initial_points, cfg.model.type)
    optimizer = generate_optimizer(model.parameters(), cfg.training.learning_rate)
    criterion = nn.MSELoss()

    # Keep track of frames for GIF
    gif_frame_epochs = []

    # Save initial state
    save_progress_plot(model, x, y, 0, float('inf'), 'initial')
    gif_frame_epochs.append(0)

    # Training loop
    losses = []
    num_points_history = []
    last_point_added_epoch = -cfg.training.min_epochs_between_points  # Allow adding point at start if needed
    best_loss = float('inf')

    for epoch in range(cfg.training.num_epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Store the loss
        current_loss = loss.item()
        losses.append(current_loss)
        best_loss = min(best_loss, current_loss)
        
        # Check if we should add a new point
        if (epoch > cfg.training.plateau.window and  # Need enough history
            epoch - last_point_added_epoch >= cfg.training.min_epochs_between_points and  # Minimum waiting period
            model.piecewise.num_points < cfg.model.max_points):  # Haven't reached max points
            
            # Check if loss has plateaued
            window_start_loss = losses[epoch - cfg.training.plateau.window]
            relative_improvement = (window_start_loss - current_loss) / window_start_loss
            #print('relative improvement', relative_improvement)
            # Only add point if we're at the best loss we've seen and improvement is minimal
            if relative_improvement < cfg.training.plateau.threshold or \
               (epoch - last_point_added_epoch >= cfg.training.max_epochs_between_points):
                
                error = torch.pow(torch.abs(output-y), cfg.training.error_exponent)
                new_value = model.largest_error(error, x)
                logger.debug(f'New value: {new_value}')
                
                success = False
                if new_value is not None:
                    success = model.insert_nearby_point(new_value)
                
                if success:
                    strategy = 0
                    last_point_added_epoch = epoch
                    logger.info(f'Epoch {epoch}: Added point using strategy {strategy}. '
                              f'Now using {model.piecewise.num_points} points. '
                              f'Loss: {current_loss:.6f}')
                    # Create new optimizer since parameters have changed
                    optimizer = generate_optimizer(model.parameters(), cfg.training.learning_rate)
                    # Save plot after adding new point
                    save_progress_plot(model, x, y, epoch, loss.item(), strategy)
                    gif_frame_epochs.append(epoch)
        
        optimizer.step()
        
        # Enforce monotonicity of the positions
        if isinstance(model.piecewise, NonUniformPiecewiseLinear):
            model.piecewise.enforce_monotonic()
        
        # Clamp positions to allowed range after optimizer step
        with torch.no_grad():
            model.piecewise.positions.data.clamp_(model.piecewise.position_min, model.piecewise.position_max)
        
        num_points_history.append(model.piecewise.num_points)
        
        if (epoch + 1) % cfg.visualization.plot_interval == 0:
            logger.info(f'Epoch [{epoch+1}/{cfg.training.num_epochs}], Loss: {loss.item():.4f}, '
                       f'Points: {model.piecewise.num_points}')
            save_progress_plot(model, x, y, epoch+1, loss.item(), 'progress')
            gif_frame_epochs.append(epoch + 1)

    # Save final plots and get final positions/values
    positions, values = save_final_plots(model, x, y, losses, num_points_history)

    # Create GIF using streaming approach
    logger.info("Creating GIF...")
    with imageio.get_writer(os.path.join(HydraConfig.get().run.dir, 'training_progress.gif'), mode='I', duration=0.5) as writer:
        for epoch in gif_frame_epochs:
            image_path = os.path.join(HydraConfig.get().run.dir, f'approximation_epoch_{epoch:04d}.png')
            image = imageio.imread(image_path)
            writer.append_data(image)
            # Delete the PNG file after adding it to the GIF to save space
            os.remove(image_path)
    logger.info("GIF created successfully!")

    # Print final positions and values
    logger.info("\nFinal control points:")
    for pos, val in zip(positions, values):
        logger.info(f"x: {pos:6.3f}, y: {val:6.3f}")

    # Print statistics
    logger.info(f"\nFinal number of points: {model.piecewise.num_points}")
    logger.info(f"Final loss: {losses[-1]:.6f}")

if __name__ == "__main__":
    main()

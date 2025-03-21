import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from PIL import Image
import imageio
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import image
from lion_pytorch import Lion

from non_uniform_piecewise_layers import AdaptivePiecewiseMLP
from non_uniform_piecewise_layers.rotation_layer import fixed_rotation_layer
from non_uniform_piecewise_layers.utils import largest_error
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def generate_mesh_grid(width, height, device="cpu", normalize=True):
    """
    Generate a mesh grid of positions for an image.
    
    Args:
        width: Width of the image
        height: Height of the image
        device: Device to place the tensors on
        normalize: Whether to normalize coordinates to [-1, 1]
        
    Returns:
        Tensor of shape [width*height, 2] containing (x, y) coordinates
    """
    # Create coordinate grids
    if normalize:
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_coords = torch.linspace(-1, 1, height, device=device)
    else:
        x_coords = torch.arange(width, device=device)
        y_coords = torch.arange(height, device=device)
    
    # Create mesh grid
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Reshape to [width*height, 2]
    positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
    
    return positions


def image_to_dataset(filename, device="cpu"):
    """
    Read in an image file and return the flattened position input,
    flattened output and torch array of the original image.
    
    Args:
        filename: Image filename
        device: Device to place the tensors on
        
    Returns:
        Tuple of (flattened image [width*height, 3], positions [width*height, 2], original image)
    """
    # Read the image
    img = image.imread(filename)
    
    # Convert to torch tensor
    torch_image = torch.from_numpy(np.array(img))
    
    # Generate position coordinates
    positions = generate_mesh_grid(
        width=torch_image.shape[1],  # Width is the second dimension in the image
        height=torch_image.shape[0],  # Height is the first dimension
        device=device,
        normalize=True
    )
    
    # Normalize image values to [-1, 1]
    if torch_image.dtype == torch.uint8:
        torch_image_flat = torch_image.reshape(-1, 3).float() * 2.0 / 255.0 - 1.0
    else:
        # Assume it's already normalized if not uint8
        torch_image_flat = torch_image.reshape(-1, 3)
    
    return torch_image_flat, positions, torch_image


class ImplicitImageNetwork(nn.Module):
    def __init__(self, 
                 input_dim=2, 
                 output_dim=3, 
                 hidden_layers=[64, 64, 64], 
                 rotations=4, 
                 num_points=5, 
                 position_range=(-1, 1),
                 anti_periodic=True):
        """
        Neural network for implicit image representation.
        First applies a rotation layer to the input coordinates, then passes through an adaptive MLP.
        
        Args:
            input_dim: Input dimension (typically 2 for x,y coordinates)
            output_dim: Output dimension (typically 3 for RGB values)
            hidden_layers: List of hidden layer sizes for the MLP
            rotations: Number of rotations to apply in the rotation layer
            num_points: Number of points for each piecewise function
            position_range: Range of positions for the piecewise functions
            anti_periodic: Whether to use anti-periodic boundary conditions
        """
        super().__init__()
        
        # Create the rotation layer
        self.rotation_layer, rotation_output_dim = fixed_rotation_layer(
            n=input_dim, 
            rotations=rotations, 
            normalize=True
        )
        
        # Create the adaptive MLP
        mlp_widths = [rotation_output_dim] + hidden_layers + [output_dim]
        self.mlp = AdaptivePiecewiseMLP(
            width=mlp_widths,
            num_points=num_points,
            position_range=position_range,
            anti_periodic=anti_periodic
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Apply rotation layer
        x = self.rotation_layer(x)
        
        # Apply MLP
        x = self.mlp(x)
        
        return x
    
    def move_smoothest(self):
        """
        Move the smoothest point in the network to improve fitting.
        """
        self.mlp.move_smoothest()
    
    def global_error(self, error, x):
        """
        Find the input x value that corresponds to the largest error and add a point there.
        
        Args:
            error: Error tensor of shape (batch_size, output_dim)
            x: Input tensor of shape (batch_size, input_dim)
        """
        new_value = largest_error(error, x)
        if new_value is not None:
            logger.debug(f'New value: {new_value}')
            self.mlp.remove_add(new_value)


def save_progress_image(model, inputs, original_image, epoch, loss, output_dir):
    """
    Save a plot showing the current state of the approximation and the original image.
    
    Args:
        model: The neural network model
        inputs: Input coordinates
        original_image: Original image tensor
        epoch: Current epoch
        loss: Current loss value
        output_dir: Directory to save the image
    """
    with torch.no_grad():
        predictions = model(inputs)
    
    # Reshape predictions to image dimensions
    height, width = original_image.shape[:2]
    pred_image = predictions.reshape(height, width, 3)
    
    # Convert from [-1, 1] to [0, 1] range
    pred_image = (pred_image + 1.0) / 2.0
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the predicted image
    ax1.imshow(pred_image.detach().numpy())
    ax1.set_title(f'Predicted Image - Epoch {epoch}')
    ax1.axis('off')
    
    # Plot the original image
    ax2.imshow(original_image)
    ax2.set_title(f'Original Image - Loss: {loss:.6f}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/image_progress_{epoch:04d}.png')
    plt.close()


def generate_optimizer(parameters, learning_rate, name="lion"):
    
    if name.lower() == "lion":
        return Lion(parameters, lr=learning_rate)
    elif name.lower() == "adam":
        return optim.Adam(parameters, lr=learning_rate)
    elif name.lower() == "sgd":
        return optim.SGD(parameters, lr=learning_rate)
    else:
        # Default to Lion
        logger.warning(f"Unknown optimizer '{name}', using Lion as default")
        return Lion(parameters, lr=learning_rate)


@hydra.main(version_base=None, config_path="config", config_name="implicit_images")
def main(cfg: DictConfig):
    # Log some useful information
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Output directory: {HydraConfig.get().run.dir}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Get original working directory
    orig_cwd = hydra.utils.get_original_cwd()
    
    # Load the image
    image_path = os.path.join(orig_cwd, cfg.image_path)
    logger.info(f"Loading image from: {image_path}")
    
    # Convert image to dataset
    image_data, position_data, original_image = image_to_dataset(
        filename=image_path
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(position_data, image_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True
    )
    
    # Create model
    model = ImplicitImageNetwork(
        input_dim=position_data.shape[1],
        output_dim=image_data.shape[1],
        hidden_layers=cfg.model.hidden_layers,
        rotations=cfg.model.rotations,
        num_points=cfg.model.num_points,
        position_range=tuple(cfg.model.position_range),
        anti_periodic=cfg.model.anti_periodic
    )
    
    # Create optimizer
    optimizer = generate_optimizer(
        parameters=model.parameters(),
        learning_rate=cfg.training.learning_rate,
        name=cfg.training.optimizer
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Prepare for creating a GIF
    images = []
    
    # Training loop
    for epoch in range(cfg.training.num_epochs):
        total_loss = 0.0
        
        # Train with batches
        for batch_inputs, batch_targets in dataloader:
            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_inputs.size(0)
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataset)
        
        # Log progress
        logger.info(f'Epoch {epoch+1}/{cfg.training.num_epochs}, Loss: {avg_loss:.6f}')
        
        # Apply adaptation strategies
        if epoch % cfg.training.adapt_every == 0:
            # Get full predictions for adaptation
            with torch.no_grad():
                full_predictions = model(position_data)
                error = torch.abs(full_predictions - image_data)
                
            if cfg.training.adapt_strategy == "global_error":
                model.global_error(error, position_data)
                # Recreate optimizer after modifying the model
                optimizer = generate_optimizer(
                    parameters=model.parameters(),
                    learning_rate=cfg.training.learning_rate,
                    name=cfg.training.optimizer
                )
            elif cfg.training.adapt_strategy == "move_smoothest":
                model.move_smoothest()
                # Recreate optimizer after modifying the model
                optimizer = generate_optimizer(
                    parameters=model.parameters(),
                    learning_rate=cfg.training.learning_rate,
                    name=cfg.training.optimizer
                )
        
        # Save progress visualization
        if epoch % cfg.visualization.save_every == 0 or epoch == cfg.training.num_epochs - 1:
            save_progress_image(model, position_data, original_image, epoch, avg_loss, os.getcwd())
            images.append(imageio.imread(f'{os.getcwd()}/image_progress_{epoch:04d}.png'))
    
    # Save the final model
    torch.save(model.state_dict(), f'{os.getcwd()}/implicit_image_model.pt')
    
    # Create a GIF of the training progress
    if len(images) > 1:
        imageio.mimsave(f'{os.getcwd()}/training_progress.gif', images, duration=cfg.visualization.gif_duration)
        logger.info(f"GIF of training progress saved to {os.getcwd()}/training_progress.gif")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

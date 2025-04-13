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
import trimesh
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from lion_pytorch import Lion
from torch.utils.tensorboard import SummaryWriter
import requests
from io import BytesIO
from tqdm import tqdm
import mcubes
from skimage import measure
import open3d as o3d

from non_uniform_piecewise_layers import AdaptivePiecewiseMLP
from non_uniform_piecewise_layers.rotation_layer import fixed_rotation_layer
from non_uniform_piecewise_layers.utils import largest_error

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def download_model(url, save_path=None):
    """
    Download a 3D model from a URL.
    
    Args:
        url: URL to download from
        save_path: Path to save the downloaded model
        
    Returns:
        Path to the downloaded model
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"Model already exists at {save_path}, skipping download")
        return save_path
    
    logger.info(f"Downloading model from {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path
    else:
        return BytesIO(response.content)


def load_mesh(file_path):
    """
    Load a 3D mesh from a file.
    
    Args:
        file_path: Path to the mesh file
        
    Returns:
        Trimesh object
    """
    logger.info(f"Loading mesh from {file_path}")
    mesh = trimesh.load(file_path)
    return mesh


def generate_point_cloud(mesh, num_points=10000, signed_distance=True, normalize=True):
    """
    Generate a point cloud from a mesh with signed distance values.
    
    Args:
        mesh: Trimesh mesh
        num_points: Number of points to sample
        signed_distance: Whether to compute signed distance values
        normalize: Whether to normalize coordinates to [-1, 1]
        
    Returns:
        Tuple of (points, sdf_values)
    """
    # Sample points on the surface
    surface_points_readonly, _ = trimesh.sample.sample_surface(mesh, num_points // 2)
    surface_points = surface_points_readonly.copy()

    # Sample points in the volume
    # Create a slightly larger bounding box
    bounds = mesh.bounding_box.bounds
    min_bound, max_bound = bounds.copy()
    padding = (max_bound - min_bound) * 0.1
    min_bound -= padding
    max_bound += padding
    
    # Sample random points in the volume
    volume_points = np.random.uniform(min_bound, max_bound, size=(num_points // 2, 3))
    
    # Combine points
    points = np.vstack([surface_points, volume_points])
    
    # Compute signed distance values
    if signed_distance:
        # Positive outside, negative inside
        sdf_values = np.zeros(points.shape[0])
        
        # Surface points have distance close to 0
        sdf_values[:num_points // 2] = 0.0 #np.random.normal(0, 0.01, size=num_points // 2)
        
        # Volume points need distance computation
        for i in range(num_points // 2, num_points):
            point = points[i]
            closest_point, distance, _ = trimesh.proximity.closest_point(mesh, [point])
            # Check if point is inside or outside
            inside = mesh.contains([point])[0]
            sdf_values[i] = -distance if inside else distance
    else:
        # Just use 1 for surface, 0 for non-surface
        sdf_values = np.zeros(points.shape[0])
        sdf_values[:num_points // 2] = 1.0
    
    # Normalize coordinates to [-1, 1]
    if normalize:
        center = (min_bound + max_bound) / 2
        scale = np.max(max_bound - min_bound) / 2
        points = (points - center) / scale
    
    # Convert to torch tensors
    points_tensor = torch.from_numpy(points).float()
    sdf_values_tensor = torch.from_numpy(sdf_values).float().unsqueeze(1)
    
    return points_tensor, sdf_values_tensor


class Implicit3DNetwork(nn.Module):
    """
    Neural network for implicit 3D representation.
    Uses a rotation layer followed by an adaptive MLP.
    """
    def __init__(
        self,
        input_dim=3,
        output_dim=1,
        hidden_layers=[64, 64, 64],
        rotations=4,
        num_points=5,
        position_range=(-1, 1),
        anti_periodic=True,
        position_init='random'
    ):
        """
        Initialize the network.
        
        Args:
            input_dim: Dimension of input (typically 3 for x,y,z coordinates)
            output_dim: Output dimension (typically 1 for SDF values)
            hidden_layers: List of hidden layer sizes
            rotations: Number of rotations in the rotation layer
            num_points: Number of points for each piecewise function
            position_range: Range of positions for the piecewise functions
            anti_periodic: Whether to use anti-periodic boundary conditions
        """
        super(Implicit3DNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
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
            anti_periodic=anti_periodic,
            position_init=position_init,
            normalization="maxabs"
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
    
    def global_error(self, error, x):
        """
        Find the input x value that corresponds to the largest error and add a point there.
        
        Args:
            error: Error tensor of shape (batch_size, output_dim)
            x: Input tensor of shape (batch_size, input_dim)
        """
        # Make sure inputs are on the same device as the model
        device = next(self.parameters()).device
        if error.device != device:
            error = error.to(device)
        if x.device != device:
            x = x.to(device)
            
        # Apply rotation to get the rotated positions
        rotated_x = self.rotation_layer(x)
        
        # Find the largest error and add a point
        new_value = largest_error(error, rotated_x)
        if new_value is not None:
            logger.debug(f'New value: {new_value}')
            self.mlp.remove_add(new_value)
    
    def move_smoothest(self):
        """
        Move the smoothest point in the network to improve fitting.
        """
        self.mlp.move_smoothest()


def batch_predict(model, inputs, batch_size=1024):
    """
    Run predictions in batches to avoid memory issues.
    
    Args:
        model: The model to use for predictions
        inputs: Input tensor
        batch_size: Batch size for predictions
        
    Returns:
        Tensor of predictions
    """
    model.eval()
    predictions = []
    num_samples = inputs.size(0)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_predictions = model(batch_inputs)
            predictions.append(batch_predictions)
    
    return torch.cat(predictions, dim=0)


def extract_mesh(model, resolution=64, threshold=0.0, device='cpu'):
    """
    Extract a mesh from the implicit function using marching cubes.
    
    Args:
        model: The neural network model
        resolution: Grid resolution for marching cubes
        threshold: Isosurface threshold
        device: Device to run the model on
        
    Returns:
        Tuple of (vertices, faces)
    """
    # Create a grid of points
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Reshape to a list of points
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    points_tensor = torch.from_numpy(points).float().to(device)
    
    # Predict SDF values
    sdf_values = batch_predict(model, points_tensor).cpu().numpy().reshape(resolution, resolution, resolution)
    
    # Debug information about SDF values
    logger.info(f"SDF values - min: {np.min(sdf_values)}, max: {np.max(sdf_values)}, mean: {np.mean(sdf_values)}")
    logger.info(f"Number of values below threshold {threshold}: {np.sum(sdf_values < threshold)}")
    logger.info(f"Number of values above threshold {threshold}: {np.sum(sdf_values > threshold)}")
    
    # Try different thresholds if needed
    """
    if np.sum(sdf_values < threshold) == 0 or np.sum(sdf_values > threshold) == 0:
        logger.warning(f"No isosurface found at threshold {threshold}. Trying to find a better threshold...")
        # Find a threshold that would create a surface
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        for p in percentiles:
            potential_threshold = np.percentile(sdf_values, p)
            below_count = np.sum(sdf_values < potential_threshold)
            above_count = np.sum(sdf_values > potential_threshold)
            logger.info(f"Percentile {p}%: threshold={potential_threshold}, below={below_count}, above={above_count}")
        
        # Use median as fallback threshold if original threshold doesn't work
        if np.sum(sdf_values < threshold) == 0 or np.sum(sdf_values > threshold) == 0:
            new_threshold = np.median(sdf_values)
            logger.warning(f"Using median as threshold: {new_threshold}")
            threshold = new_threshold
    """
    
    # Extract mesh using marching cubes
    try:
        vertices, faces = mcubes.marching_cubes(sdf_values, threshold)
        logger.info(f"Mesh extracted: {len(vertices)} vertices, {len(faces)} faces")
    except Exception as e:
        logger.error(f"Error in marching cubes: {str(e)}")
        return np.array([]), np.array([])  # Return empty arrays on error
    
    # Check if mesh is empty
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Generated mesh is empty!")
        return np.array([]), np.array([])
    
    # Rescale vertices to [-1, 1]
    vertices = vertices / (resolution - 1) * 2 - 1
    
    return vertices, faces


def save_mesh(vertices, faces, filename):
    """
    Save a mesh to an OBJ file.
    
    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        filename: Output filename
    """
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning(f"Cannot save empty mesh to {filename}")
        # Create an empty file with a comment explaining why it's empty
        with open(filename, 'w') as f:
            f.write("# Empty mesh - no isosurface found\n")
        return
    
    logger.info(f"Saving mesh with {len(vertices)} vertices and {len(faces)} faces to {filename}")
    mcubes.export_obj(vertices, faces, filename)


def add_mesh_to_tensorboard(writer, vertices, faces, epoch, tag="mesh"):
    """
    Add a 3D mesh to TensorBoard using PyTorch's native TensorBoard mesh visualization.
    
    Args:
        writer: TensorBoard SummaryWriter
        vertices: Mesh vertices as numpy array (should be on CPU)
        faces: Mesh faces as numpy array (should be on CPU)
        epoch: Current epoch
        tag: Tag for the mesh in TensorBoard
    """
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        logger.warning(f"Cannot add empty or invalid mesh '{tag}' to TensorBoard")
        return
    
    # --- SOLUTION: Ensure we work with a CPU Open3D TriangleMesh --- 
    # Explicitly create a standard CPU mesh and copy data.
    # This avoids potential issues with implicit CUDA mesh creation.
    logger.info(f"Creating CPU mesh for '{tag}'...")
    
    # Ensure input arrays are numpy arrays on CPU
    if isinstance(vertices, torch.Tensor):
        vertices_np = vertices.detach().cpu().numpy()
    else:
        vertices_np = np.asarray(vertices)
        
    if isinstance(faces, torch.Tensor):
        faces_np = faces.detach().cpu().numpy()
    else:
        faces_np = np.asarray(faces)
        
    # Fix inside-out rendering by reversing face orientations
    # Swap the order of indices in each face (e.g., [0,1,2] becomes [0,2,1])
    logger.info(f"Reversing face orientations for '{tag}' to fix inside-out rendering...")
    reversed_faces_np = faces_np.copy()
    # Swap the second and third vertex indices for each face
    reversed_faces_np[:, [1, 2]] = reversed_faces_np[:, [2, 1]]
    # Use the reversed faces
    faces_np = reversed_faces_np
    
    # Skip the Open3D mesh creation and orientation entirely
    # Just prepare the data for TensorBoard directly
    
    # Create default colors based on vertex positions for visualization
    # Normalize vertex positions to [0,1] range for coloring
    min_vals = np.min(vertices_np, axis=0)
    max_vals = np.max(vertices_np, axis=0)
    range_vals = max_vals - min_vals
    # Avoid division by zero
    range_vals[range_vals == 0] = 1.0
    
    # Use normalized XYZ coordinates as RGB colors
    colors = (vertices_np - min_vals) / range_vals
    # Scale to [0, 255] for TensorBoard
    colors = (colors * 255).astype(np.uint8)
    
    # Convert to PyTorch tensors and add batch dimension
    vertices_tensor = torch.tensor(vertices_np).float().unsqueeze(0)  # [1, N, 3]
    faces_tensor = torch.tensor(faces_np).int().unsqueeze(0)          # [1, F, 3]
    colors_tensor = torch.tensor(colors).byte().unsqueeze(0)          # [1, N, 3]
    
    # Add to TensorBoard
    try:
        writer.add_mesh(
            tag=tag,
            vertices=vertices_tensor,
            faces=faces_tensor,
            colors=colors_tensor,
            global_step=epoch
        )
        logger.info(f"Mesh '{tag}' added to TensorBoard.")
    except Exception as e:
        logger.error(f"Failed to add mesh '{tag}' to TensorBoard: {e}")


def save_progress_image(model, points, sdf_values, epoch, loss, output_dir, batch_size=512, writer=None):
    """
    Save a plot showing the current state of the approximation.
    
    Args:
        model: The neural network model
        points: Input coordinates
        sdf_values: Ground truth SDF values
        epoch: Current epoch
        loss: Current loss value
        output_dir: Directory to save the image
        batch_size: Batch size for prediction to avoid memory issues
        writer: TensorBoard SummaryWriter for logging
    """
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Get model predictions
    predictions = batch_predict(model, points, batch_size=batch_size)
    
    # Convert tensors to numpy for plotting
    points_np = points.cpu().numpy()
    sdf_values_np = sdf_values.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    # Plot ground truth
    sc1 = axs[0].scatter(points_np[:, 0], points_np[:, 1], c=sdf_values_np, cmap='viridis', s=1)
    axs[0].set_title(f'Ground Truth')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    fig.colorbar(sc1, ax=axs[0])
    
    # Plot predictions
    sc2 = axs[1].scatter(points_np[:, 0], points_np[:, 1], c=predictions_np, cmap='viridis', s=1)
    axs[1].set_title(f'Prediction (Epoch {epoch}, Loss {loss:.6f})')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    fig.colorbar(sc2, ax=axs[1])
    
    # Save the figure
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'progress_epoch_{epoch:04d}.png'), dpi=150)
    plt.close(fig)
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar('Loss/train', loss, epoch)


def generate_optimizer(parameters, learning_rate, name="lion"):
    """
    Generate an optimizer for training.
    
    Args:
        parameters: Model parameters to optimize
        learning_rate: Learning rate for the optimizer
        name: Optimizer name (lion or adam)
        
    Returns:
        Optimizer instance
    """
    if name.lower() == "lion":
        return Lion(parameters, lr=learning_rate, weight_decay=0)
    else:
        return optim.Adam(parameters, lr=learning_rate, weight_decay=0)


@hydra.main(config_path="config", config_name="implicit_3d")
def main(cfg: DictConfig):
    """
    Main function for training an implicit 3D representation.
    
    Args:
        cfg: Hydra configuration
    """
    # Get output directory from Hydra
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Download and load 3D model
    model_path = download_model(cfg.model_url, os.path.join(output_dir, "model.obj"))
    mesh = load_mesh(model_path)
    
    # Generate point cloud with SDF values
    points, sdf_values = generate_point_cloud(
        mesh, 
        num_points=cfg.num_mesh_points, 
        signed_distance=True, 
        normalize=True
    )
    
    # Move data to device
    points = points.to(device)
    sdf_values = sdf_values.to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(points, sdf_values)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True
    )
    
    # Create model
    model = Implicit3DNetwork(
        input_dim=3,
        output_dim=1,
        hidden_layers=cfg.hidden_layers,
        rotations=cfg.rotations,
        num_points=cfg.num_points,
        position_range=cfg.position_range,
        anti_periodic=cfg.anti_periodic,
        position_init=cfg.position_init
    ).to(device)
    
    # Create optimizer
    optimizer = generate_optimizer(
        model.parameters(), 
        learning_rate=cfg.learning_rate,
        name=cfg.optimizer
    )
    
    # Create loss function
    loss_fn = nn.MSELoss()
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        
        # Process batches
        for batch_idx, (batch_points, batch_sdf) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # Forward pass
            predictions = model(batch_points)
            loss = loss_fn(predictions, batch_sdf)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            
            # Adaptive point management (every N batches)
            if cfg.adaptive and batch_idx % cfg.adaptive_frequency == 0:
                with torch.no_grad():
                    # Calculate error
                    # error = torch.abs(predictions - batch_sdf)
                    
                    # Find point with largest error and add a point there
                    # model.global_error(error, batch_points)
                    
                    # Move smoothest point
                    if cfg.move_smoothest and epoch > cfg.move_smoothest_after:
                        optimizer = generate_optimizer(
                            parameters=model.parameters(),
                            learning_rate=cfg.learning_rate,
                            name=cfg.optimizer
                        )
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Save progress visualization
        if epoch % cfg.save_frequency == 0:
            save_progress_image(
                model, 
                points, 
                sdf_values, 
                epoch, 
                avg_loss, 
                os.path.join(output_dir, "progress"),
                batch_size=cfg.batch_size,
                writer=writer
            )
            
            # Extract and save mesh
            if cfg.extract_mesh:
                vertices, faces = extract_mesh(
                    model, 
                    resolution=cfg.mesh_resolution, 
                    threshold=cfg.mesh_threshold,
                    device=device
                )
                save_mesh(
                    vertices, 
                    faces, 
                    os.path.join(output_dir, f"mesh_epoch_{epoch:04d}.obj")
                )
                
                # Add mesh to TensorBoard
                add_mesh_to_tensorboard(
                    writer,
                    vertices,
                    faces,
                    epoch,
                    tag="3d_mesh"  # Use consistent tag for slider effect
                )
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        
        # Save checkpoint
        if epoch % cfg.checkpoint_frequency == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch:04d}.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    
    # Extract final mesh
    if cfg.extract_mesh:
        vertices, faces = extract_mesh(
            model, 
            resolution=cfg.mesh_resolution, 
            threshold=cfg.mesh_threshold,
            device=device
        )
        save_mesh(vertices, faces, os.path.join(output_dir, "final_mesh.obj"))
        
        # Add final mesh to TensorBoard
        add_mesh_to_tensorboard(
            writer,
            vertices,
            faces,
            cfg.epochs,
            tag="3d_mesh"  # Use consistent tag for slider effect
        )
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

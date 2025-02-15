import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from non_uniform_piecewise_layers import AdaptivePiecewiseMLP
from lion_pytorch import Lion
import imageio

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for plots
os.makedirs('examples/dynamic_circle_plots', exist_ok=True)

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
    return outputs

def save_progress_plot(model, inputs, outputs, epoch, loss, position, grid_size):
    """Save a plot showing the current state of the approximation."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert inputs to numpy for plotting
    x_np = inputs[:, 0].reshape(grid_size, grid_size).numpy()
    y_np = inputs[:, 1].reshape(grid_size, grid_size).numpy()
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(inputs).reshape(grid_size, grid_size).numpy()
    
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
    plt.savefig(f'examples/dynamic_circle_plots/circle_{epoch:03d}.png')
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
        positions = layer.positions.data
        values = layer.values.data
        
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
    filename = f'examples/dynamic_circle_plots/weights_epoch_{epoch:04d}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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
    plt.savefig('examples/dynamic_circle_plots/convergence.png')
    plt.close()

# Create synthetic data
grid_size = 50
x = torch.linspace(-1, 1, grid_size)
y = torch.linspace(-1, 1, grid_size)
xx, yy = torch.meshgrid(x, y, indexing='ij')
inputs = torch.stack([xx.flatten(), yy.flatten()], dim=1)

# Training parameters
num_points = 10  # Initial number of points in piecewise function
hidden_width=5
num_epochs = 400  # Total number of epochs
switch_epoch = 200  # Epoch at which to switch the circle position
learning_rate = 0.001

# Create model and optimizer
model = AdaptivePiecewiseMLP(
    width=[2, hidden_width,hidden_width, 1],  # Input dim: 2, Hidden layers: 32, Output dim: 1
    num_points=num_points,
    position_range=(-1, 1)
)

# Create GIF writers
progress_writer = imageio.get_writer('examples/dynamic_circle_plots/progress.gif', mode='I', duration=0.5)
weights_writer = imageio.get_writer('examples/dynamic_circle_plots/weights.gif', mode='I', duration=0.5)

def generate_optimizer(parameters, learning_rate):
    return Lion(parameters, lr=learning_rate)

optimizer = generate_optimizer(model.parameters(), learning_rate)
loss_function = nn.MSELoss()
losses = []
epochs = []

# Training loop
for epoch in range(num_epochs):
    # Generate target data based on current epoch
    
    if epoch < switch_epoch:
        outputs = generate_circle_data(xx.flatten(), yy.flatten(),'lower_left')
        position = 'lower_left'
    else:
        outputs = generate_circle_data(xx.flatten(), yy.flatten(),'upper_right')
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
        success=model.remove_add(new_value)
        optimizer=generate_optimizer(model.parameters(),learning_rate)

    # Save progress plot every 10 epochs
    if epoch % 10 == 0:
        # Save progress plot and add to GIF
        progress_path = f'examples/dynamic_circle_plots/circle_{epoch:03d}.png'
        save_progress_plot(model, inputs, outputs, epoch, loss.item(), position, grid_size)
        progress_writer.append_data(imageio.imread(progress_path))
        
        # Save weights plot and add to GIF
        weights_path = f'examples/dynamic_circle_plots/weights_epoch_{epoch:04d}.png'
        save_piecewise_plots(model, epoch)
        weights_writer.append_data(imageio.imread(weights_path))
        
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}')

# Save convergence plot
save_convergence_plot(losses, epochs)

# Close the writers
progress_writer.close()
weights_writer.close()

print("GIFs created successfully!")

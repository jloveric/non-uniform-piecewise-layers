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
os.makedirs('examples/dynamic_square_wave_plots', exist_ok=True)

def generate_square_wave(x, position='left'):
    """Generate a square wave at either the left or right position."""
    if position == 'left':
        center = -0.5
    else:  # right
        center = 0.5
    
    width = 0.2  # width of the square pulse
    mask = torch.abs(x - center) < width/2
    y = torch.zeros_like(x)
    y[mask] = 0.75
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
    fig.savefig(f'examples/dynamic_square_wave_plots/square_wave_{epoch}.png')
    images.append(imageio.imread(f'examples/dynamic_square_wave_plots/square_wave_{epoch}.png'))
    plt.close()

# Create synthetic data
x = torch.linspace(-1, 1, 1000).reshape(-1, 1)

# Training parameters
num_points = 10  # Initial number of points in piecewise function
num_epochs = 400  # Total number of epochs
switch_epoch = 100  # Epoch at which to switch the square wave position
learning_rate = 0.01

# Create model and optimizer
model = AdaptivePiecewiseMLP(
    width=[1, 1],  # Input dim: 1, Hidden layers: 10, Output dim: 1
    num_points=num_points,
    position_range=(-1, 1)
)

def generate_optimizer(parameters, learning_rate) :
    #return torch.optim.Adam(parameters,1e-3)
    #return torch.optim.SGD(parameters, lr=1e-2)
    return Lion(parameters, lr=learning_rate)

optimizer = generate_optimizer(model.parameters(), learning_rate)

# Prepare for creating a GIF
images = []

# Training loop
for epoch in range(num_epochs):
    # Generate target data based on epoch
    position = 'left' if epoch < switch_epoch else 'right'
    y = generate_square_wave(x, position)
    
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
    model.remove_add(new_value)
    optimizer=generate_optimizer(model.parameters(),learning_rate)
    
    # Save progress plot every 10 epochs
    if epoch % 10 == 0:
        save_progress_plot(model, x, y, epoch, loss.item(), position)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}, Position: {position}')

# Save the images as a GIF
imageio.mimsave('dynamic_square_wave.gif', images, duration=0.1)  # Adjust duration as needed

print("GIF 'dynamic_square_wave.gif' created successfully!")

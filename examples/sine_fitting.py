import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from non_uniform_piecewise_layers import NonUniformPiecewiseLinear
from lion_pytorch import Lion

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for plots
os.makedirs('examples/progress_plots', exist_ok=True)

def save_progress_plot(model, x, y, epoch, loss, strategy):
    """Save a plot showing the current state of the approximation."""
    plt.figure(figsize=(12, 8))
    
    # Plot the results
    with torch.no_grad():
        y_pred = model(x)

    # Convert to numpy for plotting
    x_np = x.numpy()
    y_np = y.numpy()
    y_pred_np = y_pred.numpy()

    plt.plot(x_np, y_np, 'b-', label='True Function', alpha=0.5)
    plt.plot(x_np, y_pred_np, 'r--', label='Piecewise Linear Approximation')

    # Plot the control points
    positions = model.piecewise.positions.data[0, 0].numpy()
    values = model.piecewise.values.data[0, 0].numpy()
    plt.scatter(positions, values, c='g', s=100, label='Control Points')

    plt.title(f'Function Approximation - Epoch {epoch}\n'
             f'Points: {len(positions)}, Loss: {loss:.4f}, Strategy: {strategy}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'examples/progress_plots/approximation_epoch_{epoch:04d}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Create synthetic sine wave data
x = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y = torch.cos(1/torch.abs(x)+0.25)

# Create a simple model with our non-uniform piecewise linear layer
class SineApproximator(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.piecewise = NonUniformPiecewiseLinear(
            num_inputs=1,
            num_outputs=1,
            num_points=num_points
        )
    
    def forward(self, x):
        return self.piecewise(x)

# Training parameters
initial_points = 2  # Number of points in piecewise function
max_points = 50    # Maximum number of points to add
points_add_frequency = 500  # Add a point every N epochs
model = SineApproximator(initial_points)
criterion = nn.MSELoss()
optimizer = Lion(model.parameters(), lr=1e-3)
num_epochs = 10000

# Save initial state
save_progress_plot(model, x, y, 0, float('inf'), 'initial')

# Training loop
losses = []
num_points_history = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    model.piecewise.zero_abs_grad_accumulation()  # Zero out absolute gradient accumulation
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Add a new point periodically if we haven't reached max_points
    if epoch > 0 and epoch % points_add_frequency == 0 and model.piecewise.num_points < max_points:
        # Try each split strategy in turn
        strategy = 2 #(epoch // points_add_frequency) % 3
        success = model.piecewise.add_point_at_max_error(split_strategy=strategy)
        print('split_strategy', strategy)
        if success:
            print(f'Epoch {epoch}: Added point using strategy {strategy}. '
                  f'Now using {model.piecewise.num_points} points')
            # Create new optimizer since parameters have changed
            optimizer = Lion(model.parameters(), lr=1e-3)
            # Save plot after adding new point
            save_progress_plot(model, x, y, epoch, loss.item(), strategy)
    
    optimizer.step()
    
    # Enforce monotonicity of the positions
    model.piecewise.enforce_monotonic()
    
    losses.append(loss.item())
    num_points_history.append(model.piecewise.num_points)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
              f'Points: {model.piecewise.num_points}')

# Final plots
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
plt.savefig('examples/final_approximation.png', dpi=300, bbox_inches='tight')
plt.close()

# Print final positions and values
print("\nFinal control points:")
for pos, val in zip(positions, values):
    print(f"x: {pos:6.3f}, y: {val:6.3f}")

# Print statistics
print(f"\nFinal number of points: {model.piecewise.num_points}")
print(f"Final loss: {losses[-1]:.6f}")

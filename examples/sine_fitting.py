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


        #self.piecewise = NonUniformPiecewiseLinear(
        #    num_inputs=1,
        #    num_outputs=1,
        #    num_points=num_points
        #)
        self.piecewise = AdaptivePiecewiseLinear(
            num_inputs=1,
            num_outputs=1,
            num_points=num_points
        )
    
    def forward(self, x):
        return self.piecewise(x)

    def compute_abs_grads(self, x):
        return self.piecewise.compute_abs_grads(x)

def generate_optimizer(parameters) :
    #return torch.optim.Adam(parameters,1e-3)
    #return torch.optim.SGD(parameters, lr=1e-2)
    return Lion(parameters, lr=1e-3)
    

# Training parameters
initial_points = 3  # Number of points in piecewise function
max_points = 50    # Maximum number of points to add
min_epochs_between_points = 500  # Minimum epochs to wait between adding points
max_epochs_between_points = 10000
plateau_window = 200  # Window size to check for loss plateau
plateau_threshold = 0.001  # Relative improvement threshold to detect plateau
model = SineApproximator(initial_points)
criterion = nn.MSELoss()
optimizer = generate_optimizer(model.parameters())
num_epochs = 100000

# Save initial state
save_progress_plot(model, x, y, 0, float('inf'), 'initial')

# Training loop
losses = []
num_points_history = []
last_point_added_epoch = -min_epochs_between_points  # Allow adding point at start if needed
best_loss = float('inf')

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Store the loss
    current_loss = loss.item()
    losses.append(current_loss)
    best_loss = min(best_loss, current_loss)
    
    # Check if we should add a new point
    if (epoch > plateau_window and  # Need enough history
        epoch - last_point_added_epoch >= min_epochs_between_points and  # Minimum waiting period
        model.piecewise.num_points < max_points) :  # Haven't reached max points
        
        # Check if loss has plateaued by comparing current loss with loss from plateau_window epochs ago
        window_start_loss = losses[epoch - plateau_window]
        relative_improvement = (window_start_loss - current_loss) / window_start_loss
        
        # Only add point if we're at the best loss we've seen and improvement is minimal
        if relative_improvement < plateau_threshold and current_loss <= best_loss or (epoch-last_point_added_epoch>=max_epochs_between_points):
            # Loss has plateaued at best value, try to add a point
            abs_grad = model.compute_abs_grads(x)
            strategy = 2
            success = model.piecewise.add_point_at_max_error(abs_grad=abs_grad, split_strategy=strategy)
            
            if success:
                last_point_added_epoch = epoch
                print(f'Epoch {epoch}: Added point using strategy {strategy}. '
                      f'Now using {model.piecewise.num_points} points. '
                      f'Current loss: {current_loss:.6f} (best loss)')
                # Create new optimizer since parameters have changed
                optimizer = generate_optimizer(model.parameters())
                # Save plot after adding new point
                save_progress_plot(model, x, y, epoch, loss.item(), strategy)
    
    optimizer.step()
    
    # Enforce monotonicity of the positions
    if isinstance(model, NonUniformPiecewiseLinear) :
        model.piecewise.enforce_monotonic()
    
    # Clamp positions to allowed range after optimizer step
    with torch.no_grad():
        model.piecewise.positions.data.clamp_(model.piecewise.position_min, model.piecewise.position_max)
    
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

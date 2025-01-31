import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from non_uniform_piecewise_layers import NonUniformPiecewiseLinear
from lion_pytorch import Lion

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create synthetic sine wave data
x = torch.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
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
num_points = 50  # Number of points in piecewise function
model = SineApproximator(num_points)
criterion = nn.MSELoss()
optimizer = Lion(model.parameters(), lr=1e-3)
num_epochs = 10000

# Training loop
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    
    
    losses.append(loss.item())
    # Enforce monotonicity of the positions
    model.piecewise.enforce_monotonic()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting
plt.figure(figsize=(15, 10))

# Plot training loss
plt.subplot(2, 1, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.grid(True)

# Plot the results
plt.subplot(2, 1, 2)
with torch.no_grad():
    y_pred = model(x)

# Convert to numpy for plotting
x_np = x.numpy()
y_np = y.numpy()
y_pred_np = y_pred.numpy()

plt.plot(x_np, y_np, 'b-', label='True Sine')
plt.plot(x_np, y_pred_np, 'r--', label='Piecewise Linear Approximation')

# Plot the learned points
positions = model.piecewise.positions.data[0, 0].numpy()
values = model.piecewise.values.data[0, 0].numpy()
plt.scatter(positions, values, c='g', label='Control Points')

plt.title('Sine Wave Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('examples/sine_approximation.png', dpi=300, bbox_inches='tight')
plt.close()

# Print final positions and values
print("\nLearned control points:")
for pos, val in zip(positions, values):
    print(f"x: {pos:6.3f}, y: {val:6.3f}")

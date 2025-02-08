import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from non_uniform_piecewise_layers import AdaptivePiecewiseMLP
from lion_pytorch import Lion

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for plots
os.makedirs('examples/shape_plots', exist_ok=True)

class Shape:
    """Base class for shapes"""
    def is_inside(self, x, y):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError

class Circle(Shape):
    """Circle centered at origin with given radius"""
    def __init__(self, radius=0.5):
        self.radius = radius
    
    def is_inside(self, x, y):
        return x**2 + y**2 <= self.radius**2
    
    def get_name(self):
        return f"circle_r{self.radius}"

class Square(Shape):
    """Square centered at origin with given side length"""
    def __init__(self, side_length=1.0):
        self.side_length = side_length
        self.half_length = side_length / 2
    
    def is_inside(self, x, y):
        return (abs(x) <= self.half_length) & (abs(y) <= self.half_length)
    
    def get_name(self):
        return f"square_s{self.side_length}"

class Triangle(Shape):
    """Equilateral triangle centered at origin with given side length"""
    def __init__(self, side_length=1.0):
        self.side_length = side_length
        self.height = side_length * np.sqrt(3) / 2
        self.half_length = side_length / 2
        
    def is_inside(self, x, y):
        # Shift triangle so its base is at y=0
        y = y + self.height/3  # Center of mass at origin
        
        # Check if point is inside triangle
        return (y <= self.height) & \
               (y >= -2 * x / np.sqrt(3) - self.height/3) & \
               (y >= 2 * x / np.sqrt(3) - self.height/3)
    
    def get_name(self):
        return f"triangle_s{self.side_length}"

def generate_training_data(shape, num_points=1000, range_min=-1, range_max=1):
    """Generate training data for shape classification"""
    # Generate random points
    x = torch.rand(num_points, 2) * (range_max - range_min) + range_min
    
    # Compute labels (inside/outside shape)
    labels = torch.tensor([shape.is_inside(x_i[0].item(), x_i[1].item()) 
                          for x_i in x], dtype=torch.float32).reshape(-1, 1)
    
    return x, labels

def save_progress_plot(model, shape, epoch, loss, num_points=100):
    """Save a plot showing the current state of the classification"""
    # Create a grid of points
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Convert to torch tensor
    grid_points = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)
    
    # Get model predictions
    with torch.no_grad():
        Z = model(grid_points).reshape(num_points, num_points).numpy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot decision boundary
    contour = ax.contour(X, Y, Z, levels=[0.5], colors='k')
    
    # Plot filled contours for prediction probabilities
    cf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu', alpha=0.7)
    
    # Plot true shape boundary
    theta = np.linspace(0, 2*np.pi, 100)
    if isinstance(shape, Circle):
        ax.plot(shape.radius * np.cos(theta), shape.radius * np.sin(theta), 
                'r--', label='True Shape')
    elif isinstance(shape, Square):
        square_x = np.array([-1, 1, 1, -1, -1]) * shape.half_length
        square_y = np.array([-1, -1, 1, 1, -1]) * shape.half_length
        ax.plot(square_x, square_y, 'r--', label='True Shape')
    elif isinstance(shape, Triangle):
        triangle_x = np.array([0, shape.half_length, -shape.half_length, 0]) 
        triangle_y = np.array([shape.height*2/3, -shape.height/3, -shape.height/3, shape.height*2/3])
        ax.plot(triangle_x, triangle_y, 'r--', label='True Shape')
    
    # Add colorbar
    plt.colorbar(cf, label='Inside Probability')
    
    ax.set_title(f'Shape Classification - Epoch {epoch}\nLoss: {loss:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.savefig(f'examples/shape_plots/{shape.get_name()}_epoch_{epoch:04d}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def save_piecewise_plots(model, shape, epoch):
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
        
        # Plot each input-output mapping
        for in_dim in range(positions.shape[0]):  # For each input dimension
            for out_dim in range(positions.shape[1]):  # For each output dimension
                pos = positions[in_dim, out_dim].numpy()
                val = values[in_dim, out_dim].numpy()
                ax.plot(pos, val, 'o-')  # Removed label
        
        ax.set_title(f'Layer {layer_idx+1} Piecewise Approximations')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(f'examples/shape_plots/{shape.get_name()}_piecewise_epoch_{epoch:04d}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def save_convergence_plot(shape, losses, points_per_layer):
    """Save a plot showing loss and points per layer over epochs"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot loss
    epochs = range(len(losses))
    ax1.plot(epochs, losses, 'b-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Convergence - {shape.get_name()}')
    ax1.grid(True)
    
    # Plot points per layer
    for i in range(len(points_per_layer[0])):
        layer_points = [points[i] for points in points_per_layer]
        ax2.plot(epochs, layer_points, label=f'Layer {i+1}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Number of Points')
    ax2.set_title('Points per Layer')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'examples/shape_plots/{shape.get_name()}_convergence.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_optimizer(parameters) :
    #return torch.optim.Adam(parameters,1e-3)
    #return torch.optim.SGD(parameters, lr=1e-2)
    return Lion(parameters, lr=1e-3)

def train_shape_classifier(shape, max_epochs=100, hidden_width=8):
    """Train a model to classify points as inside/outside a shape"""
    # Create model
    model = AdaptivePiecewiseMLP(
        width=[2, hidden_width, hidden_width, 1],  # 2D input -> hidden -> 1D output
        num_points=3  # Start with minimal points
    )
    
    # Generate training data
    x_train, y_train = generate_training_data(shape, num_points=1000)
    
    # Create optimizer and loss function
    optimizer = generate_optimizer(model.parameters())
    criterion = nn.MSELoss()  # Binary cross entropy with logits
    
    # Lists to store convergence data
    losses = []
    points_per_layer = []
    
    # Training loop
    for epoch in range(max_epochs):
        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Store convergence data
        losses.append(loss.item())
        points_per_layer.append([layer.positions.shape[-1] for layer in model.layers])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Try to insert points every 10 epochs
        if epoch % 10 == 0:
            # Find points with largest error
            with torch.no_grad():
                error = torch.abs(y_pred - y_train)
            x_error = model.largest_error(error, x_train)
            print('x_error', x_error)
            if x_error is not None:
                model.insert_nearby_point(x_error)
                optimizer = generate_optimizer(model.parameters())
        
        # Save progress plot every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            save_progress_plot(model, shape, epoch, loss.item())
    
    # Save final convergence plot and piecewise plots
    save_convergence_plot(shape, losses, points_per_layer)
    save_piecewise_plots(model, shape, max_epochs-1)
    
    return model

if __name__ == "__main__":
    # Train on different shapes
    shapes = [
        Circle(radius=0.5),
        Square(side_length=1.0),
        Triangle(side_length=1.0)
    ]
    
    for shape in shapes:
        print(f"\nTraining on {shape.get_name()}")
        model = train_shape_classifier(shape)
        print(f"Finished training on {shape.get_name()}")

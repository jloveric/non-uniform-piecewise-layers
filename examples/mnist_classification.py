import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from non_uniform_piecewise_layers import AdaptivePiecewiseConv2d
from lion_pytorch import Lion
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class AdaptiveConvNet(nn.Module):
    def __init__(self, num_points=3):
        super().__init__()
        # First convolutional layer: 1 input channel, 16 output channels
        self.conv1 = AdaptivePiecewiseConv2d(1, 4, kernel_size=3, num_points=num_points)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer: 16 input channels, 32 output channels
        self.conv2 = AdaptivePiecewiseConv2d(4, 8, kernel_size=3, num_points=num_points)
        self.pool2 = nn.MaxPool2d(2)
        
        # Calculate the size of flattened features
        # Input: 28x28 -> Conv1: 26x26 -> Pool1: 13x13
        # Conv2: 11x11 -> Pool2: 5x5 -> Flattened: 5*5*32
        self.fc1 = nn.Linear(5*5*8, 10)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.reshape(-1, 5*5*8)
        x = self.fc1(x)
        return x

    def adapt_layers(self, x, errors):
        """Adapt the convolutional layers based on prediction errors
        
        Args:
            x: Input batch
            errors: Prediction errors for the batch
        """
        # Check if we've reached the maximum number of nodes (20 per layer)
        if (self.conv1.piecewise.positions.shape[-1] >= 20 or 
            self.conv2.piecewise.positions.shape[-1] >= 20):
            return

        # Find input corresponding to largest error
        x_error = self.largest_error(errors, x)
        if x_error is not None:
            # Forward pass to get intermediate activations for the high-error input
            with torch.no_grad():
                # For conv1, we want a single 3x3 window where the error is highest
                # Unfold the input for conv1
                x1_unfolded = F.unfold(
                    x_error, 
                    kernel_size=self.conv1.kernel_size,
                    padding=self.conv1.padding,
                    stride=self.conv1.stride
                )
                # Reshape to match piecewise layer input shape
                x1_unfolded = x1_unfolded.transpose(1, 2).contiguous()
                # Find the window with highest error (using L1 norm)
                window_errors = torch.abs(x1_unfolded[0]).sum(dim=1)
                max_window_idx = torch.argmax(window_errors)
                x1_point = x1_unfolded[0][max_window_idx]
                
                if self.conv1.piecewise.positions.shape[-1] < 20:
                    self.conv1.insert_nearby_point(x1_point)
                
                # Get intermediate activation for conv2
                x1 = self.conv1(x_error)
                x1 = self.pool1(x1)
                
                # For conv2, the dimensions are different since we're working with feature maps
                x2_unfolded = F.unfold(
                    x1,
                    kernel_size=self.conv2.kernel_size,
                    padding=self.conv2.padding,
                    stride=self.conv2.stride
                )
                x2_unfolded = x2_unfolded.transpose(1, 2).contiguous()
                # Find window with highest error
                window_errors = torch.abs(x2_unfolded[0]).sum(dim=1)
                max_window_idx = torch.argmax(window_errors)
                x2_point = x2_unfolded[0][max_window_idx]
                
                if self.conv2.piecewise.positions.shape[-1] < 20:
                    self.conv2.insert_nearby_point(x2_point)

    def largest_error(self, errors, x):
        """Find the input corresponding to the largest error"""
        max_error_idx = torch.argmax(errors)
        # Ensure we return a batch dimension
        return x[max_error_idx:max_error_idx+1]

def generate_optimizer(parameters):
    """Generate the optimizer for training"""
    return Lion(parameters, lr=1e-2)

def train(model, train_loader, test_loader, epochs=10, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = generate_optimizer(model.parameters())
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Adapt layers based on error
            if batch_idx % 100 == 0:
                _, predicted = torch.max(output.data, 1)
                errors = (predicted != target).float()
                model.adapt_layers(data, errors)
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                print(f'Conv1 points: {model.conv1.piecewise.num_points}, Conv2 points: {model.conv2.piecewise.num_points}')
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies

def plot_results(train_losses, test_accuracies, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mnist_training_results.png'))
    plt.close()

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Create and train model
    model = AdaptiveConvNet(num_points=3).to(device)
    train_losses, test_accuracies = train(model, train_loader, test_loader, epochs=10, device=device)
    
    # Plot and save results
    plot_results(train_losses, test_accuracies, output_dir)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'mnist_model.pth'))

if __name__ == '__main__':
    main()

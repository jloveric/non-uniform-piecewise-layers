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
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F
from non_uniform_piecewise_layers.utils import largest_error
from torch.utils.tensorboard import SummaryWriter
import logging
import tqdm

logger = logging.getLogger(__name__)

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

    def move_smoothest(self, weighted:bool=True):
        success = True
        with torch.no_grad():
            success = self.conv1.move_smoothest(weighted=weighted)
            success = success & self.conv2.move_smoothest(weighted=weighted)

def generate_optimizer(parameters, learning_rate):
    """Generate the optimizer for training"""
    return Lion(parameters, lr=learning_rate)

def train(model, train_loader, test_loader, epochs, device, learning_rate, max_points, adapt_frequency, writer=None, log_interval=100, move_nodes:bool=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = generate_optimizer(model.parameters(), learning_rate)
    
    train_losses = []
    test_accuracies = []
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
            #data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Log to TensorBoard
            if writer is not None and batch_idx % log_interval == 0:
                writer.add_scalar('training/batch_loss', loss.item(), global_step)
                writer.add_scalar('model/conv1_points', model.conv1.piecewise.positions.shape[-1], global_step)
                writer.add_scalar('model/conv2_points', model.conv2.piecewise.positions.shape[-1], global_step)
                global_step += 1
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                logger.info(f'Conv1 points: {model.conv1.piecewise.positions.shape[-1]}, Conv2 points: {model.conv2.piecewise.positions.shape[-1]}')
        
            if move_nodes:
                model.move_smoothest(weighted=True)
                optimizer = generate_optimizer(model.parameters(), learning_rate)


        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                #data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar('training/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('evaluation/accuracy', accuracy, epoch)
        
        logger.info(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies

def plot_results(train_losses, test_accuracies, save_dir):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
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

@hydra.main(version_base=None, config_path="config", config_name="mnist_classification")
def main(cfg: DictConfig):
    """Train an adaptive convolutional network on MNIST"""
    # Log Hydra configuration information
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Output directory: {HydraConfig.get().run.dir}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Get the project root directory (for data storage)
    project_root = hydra.utils.get_original_cwd()
    
    # Extract configuration parameters
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    device = cfg.device
    max_points = cfg.max_points
    adapt_frequency = cfg.adapt_frequency
    num_points = cfg.num_points
    training_fraction = cfg.training_fraction
    move_nodes=cfg.move_nodes
    
    # Set up TensorBoard writer
    writer = None
    if cfg.tensorboard.enabled:
        writer = SummaryWriter()
        logger.info(f"TensorBoard logs will be saved to {writer.log_dir}")
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    print(f"Using device: {device}")

    # Create data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Store MNIST data in the project root directory to avoid re-downloading
    data_dir = os.path.join(project_root, 'data')
    logger.info(f"Using MNIST data directory: {data_dir}")
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_dataset.data = train_dataset.data.to(device)
    train_dataset.targets = train_dataset.targets.to(device)

    test_dataset.data = test_dataset.data.to(device)
    test_dataset.targets = test_dataset.targets.to(device)
    # Use only a fraction of the training data if specified
    if training_fraction < 1.0:
        # Calculate the number of samples to use
        num_train_samples = int(len(train_dataset) * training_fraction)
        # Create a subset of the training data
        indices = torch.randperm(len(train_dataset))[:num_train_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        logger.info(f"Using {num_train_samples} training samples ({training_fraction:.2%} of the full dataset)")
    else:
        logger.info(f"Using full training dataset with {len(train_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = AdaptiveConvNet(num_points=num_points).to(device)
    
    # Train the model
    train_losses, test_accuracies = train(
        model, 
        train_loader, 
        test_loader, 
        epochs=epochs,
        device=device,
        learning_rate=learning_rate,
        max_points=max_points,
        adapt_frequency=adapt_frequency,
        writer=writer,
        log_interval=cfg.tensorboard.log_interval,
        move_nodes=move_nodes
    )
    
    # Plot results
    plot_results(train_losses, test_accuracies, '.')
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed successfully")

if __name__ == '__main__':
    main()

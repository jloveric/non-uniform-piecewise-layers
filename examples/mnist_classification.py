import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from non_uniform_piecewise_layers import AdaptivePiecewiseConv2d
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear
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
        # First convolutional layer: 1 input channel, 4 output channels
        self.conv1 = AdaptivePiecewiseConv2d(1, 4, kernel_size=3, num_points=num_points)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer: 4 input channels, 8 output channels
        self.conv2 = AdaptivePiecewiseConv2d(4, 8, kernel_size=3, num_points=num_points)
        self.pool2 = nn.MaxPool2d(2)
        
        # Calculate the size of flattened features
        # Input: 28x28 -> Conv1: 26x26 -> Pool1: 13x13
        # Conv2: 11x11 -> Pool2: 5x5 -> Flattened: 5*5*8
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
        return success

class HybridConvNet(nn.Module):
    def __init__(self, num_points=3):
        super().__init__()
        # First convolutional layer: 1 input channel, 4 output channels
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer: 4 input channels, 8 output channels
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        
        # Calculate the size of flattened features
        # Input: 28x28 -> Conv1: 26x26 -> Pool1: 13x13
        # Conv2: 11x11 -> Pool2: 5x5 -> Flattened: 5*5*8
        self.fc1 = AdaptivePiecewiseLinear(5*5*8, 10, num_points=num_points)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.reshape(-1, 5*5*8)
        x = self.fc1(x)
        return x

    def move_smoothest(self, weighted:bool=True):
        success = True
        with torch.no_grad():
            success = self.fc1.move_smoothest(weighted=weighted)
        return success

class StandardConvNet(nn.Module):
    def __init__(self, num_points=None):  # num_points is ignored but kept for consistent interface
        super().__init__()
        # First convolutional layer: 1 input channel, 4 output channels
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer: 4 input channels, 8 output channels
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        
        # Calculate the size of flattened features
        # Input: 28x28 -> Conv1: 26x26 -> Pool1: 13x13
        # Conv2: 11x11 -> Pool2: 5x5 -> Flattened: 5*5*8
        self.fc1 = nn.Linear(5*5*8, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.reshape(-1, 5*5*8)
        x = self.fc1(x)
        return x

    def move_smoothest(self, weighted:bool=True):
        # This is a no-op for the standard network, but kept for consistent interface
        return True

def generate_optimizer(parameters, learning_rate):
    """Generate the optimizer for training"""
    return Lion(parameters, lr=learning_rate)

def train(model, train_loader, test_loader, epochs, device, learning_rate, max_points, adapt_frequency, writer=None, log_interval=100, move_nodes:bool=True, val_loader=None, test_interval=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = generate_optimizer(model.parameters(), learning_rate)
    
    train_losses = []
    test_accuracies = []
    train_accuracies = []  # Will now store batch accuracies during training
    val_accuracies = []
    global_step = 0
    training_step = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate batch training accuracy
            _, predicted = torch.max(output.data, 1)
            batch_total = target.size(0)
            batch_correct = (predicted == target).sum().item()
            batch_accuracy = 100 * batch_correct / batch_total
            
            # Accumulate batch statistics for epoch accuracy
            epoch_total += batch_total
            epoch_correct += batch_correct
            
            # Log to TensorBoard
            if writer is not None and batch_idx % log_interval == 0:
                writer.add_scalar('training/batch_loss', loss.item(), global_step)
                writer.add_scalar('training/batch_accuracy', batch_accuracy, global_step)
                global_step += 1
            
            if move_nodes and (training_step%adapt_frequency)==0:
                model.move_smoothest(weighted=True)
                optimizer = generate_optimizer(model.parameters(), learning_rate)
                #print('adapting')

            training_step+=1

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Calculate epoch training accuracy from accumulated batch statistics
        epoch_train_accuracy = 100 * epoch_correct / epoch_total
        train_accuracies.append(epoch_train_accuracy)
        
        # Evaluate on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        if val_loader is not None:
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            val_accuracies.append(val_accuracy)
        
        # Evaluate on test set only every test_interval epochs and on the last epoch
        if (epoch + 1) % test_interval == 0 or epoch == epochs - 1:
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
            
            test_accuracy = 100 * test_correct / test_total
            test_accuracies.append(test_accuracy)
        else:
            # If we're not evaluating on test set this epoch, use the previous value or 0
            test_accuracy = test_accuracies[-1] if test_accuracies else 0
            test_accuracies.append(test_accuracy)
        
        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar('training/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('training/epoch_accuracy', epoch_train_accuracy, epoch)
            writer.add_scalar('evaluation/test_accuracy', test_accuracy, epoch)
            if val_loader is not None:
                writer.add_scalar('evaluation/val_accuracy', val_accuracy, epoch)
        
        # Log progress
        log_message = f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%'
        if val_loader is not None:
            log_message += f', Val Accuracy: {val_accuracy:.2f}%'
        if (epoch + 1) % test_interval == 0 or epoch == epochs - 1:
            log_message += f', Test Accuracy: {test_accuracy:.2f}%'
        logger.info(log_message)
    
    # Calculate final training accuracy on the entire training set (only at the end)
    if epochs > 0:
        model.eval()
        final_train_correct = 0
        final_train_total = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                final_train_total += target.size(0)
                final_train_correct += (predicted == target).sum().item()
        
        final_train_accuracy = 100 * final_train_correct / final_train_total
        logger.info(f'Final Training Accuracy: {final_train_accuracy:.2f}%')
    
    return train_losses, test_accuracies, train_accuracies, val_accuracies if val_loader is not None else None, final_train_accuracy if epochs > 0 else 0

def plot_results(train_losses, test_accuracies, save_dir, val_accuracies=None):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    if val_accuracies is not None:
        plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
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
    move_nodes = cfg.move_nodes
    model_type = cfg.model_type
    test_interval = cfg.test_interval
    
    torch.manual_seed(cfg.seed)

    # Set up TensorBoard writer
    writer = None
    if cfg.tensorboard.enabled:
        writer = SummaryWriter()
        logger.info(f"TensorBoard logs will be saved to {writer.log_dir}")
        
        # Prepare hyperparameters dictionary for TensorBoard
        # Convert all values to strings as TensorBoard expects
        hparams = {
            'epochs': str(epochs),
            'batch_size': str(batch_size),
            'learning_rate': str(learning_rate),
            'max_points': str(max_points),
            'adapt_frequency': str(adapt_frequency),
            'num_points': str(num_points),
            'training_fraction': str(training_fraction),
            'move_nodes': str(move_nodes),
            'model_type': model_type
        }
        # We'll add metrics after training completes
        
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
    
    
    # Total number of training samples
    total_train_samples = len(train_dataset)
    
    # Number of validation samples (10k)
    val_samples = 10000
    
    # Create indices for the train/validation split
    indices = torch.arange(total_train_samples)
    train_indices = indices[val_samples:]
    val_indices = indices[:val_samples]
    
    # Create train and validation datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    logger.info(f"Split training data: {len(train_subset)} training samples, {len(val_subset)} validation samples")
    
    # Apply training fraction if specified (to the training subset, not validation)
    if training_fraction < 1.0:
        # Calculate the number of samples to use
        num_train_samples = int(len(train_subset) * training_fraction)
        # Create a subset of the training data
        train_indices_subset = torch.randperm(len(train_subset))[:num_train_samples]
        train_subset = torch.utils.data.Subset(train_subset, train_indices_subset)
        logger.info(f"Using {num_train_samples} training samples ({training_fraction:.2%} of the training subset)")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model based on model_type
    if model_type == 'adaptive':
        model = AdaptiveConvNet(num_points=num_points).to(device)
        logger.info(f"AdaptiveConvNet initialized with {num_points} points per piecewise function")
    elif model_type == 'hybrid':
        model = HybridConvNet(num_points=num_points).to(device)
        logger.info(f"HybridConvNet initialized with {num_points} points in the linear layer")
    elif model_type == 'standard':
        model = StandardConvNet().to(device)
        logger.info("StandardConvNet initialized with regular Conv2d and Linear layers")
    else:
        logger.error(f"Unknown model type: {model_type}. Using AdaptiveConvNet as default.")
        model = AdaptiveConvNet(num_points=num_points).to(device)
    
    # Train the model
    train_losses, test_accuracies, train_accuracies, val_accuracies, final_train_accuracy = train(
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
        move_nodes=move_nodes,
        val_loader=val_loader,
        test_interval=test_interval
    )
    
    # Plot results
    plot_results(train_losses, test_accuracies, '.', val_accuracies)
    
    # Log final metrics with hyperparameters to TensorBoard
    if writer is not None:
        # Get the final metrics
        final_test_accuracy = test_accuracies[-1] if test_accuracies else 0
        final_val_accuracy = val_accuracies[-1] if val_accuracies else 0
        final_loss = train_losses[-1] if train_losses else 0
        
        # Create metric dictionaries for TensorBoard
        metric_dict = {
            'hparam/test_accuracy': final_test_accuracy,
            'hparam/train_accuracy': final_train_accuracy,
            'hparam/val_accuracy': final_val_accuracy,
            'hparam/loss': final_loss
        }
        
        # Log metrics individually first (required for hparams visualization)
        for metric_name, metric_value in metric_dict.items():
            writer.add_scalar(metric_name, metric_value)
        
        # Log hyperparameters and metrics together using the same writer
        writer.add_hparams(hparams, metric_dict)
        writer.close()
        logger.info("TensorBoard writer closed successfully")

if __name__ == '__main__':
    main()

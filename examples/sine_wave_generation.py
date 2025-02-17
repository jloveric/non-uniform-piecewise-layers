import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from non_uniform_piecewise_layers.adaptive_piecewise_mingru import MinGRUStack, AdaptivePiecewiseLinear
from lion_pytorch import Lion
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# Set project root
os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent.absolute())

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class TimeSeriesMinGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_points=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # MinGRU stack for processing time series
        self.rnn = MinGRUStack(
            input_dim=input_size,  # Take raw input
            state_dim=hidden_size,
            out_features=input_size,  # Output back to input size
            layers=num_layers,
            num_points=num_points
        )
    
    def forward(self, x, h=None):
        # x shape: (batch, seq_len, input_size)
        output, states = self.rnn(x, h)
        return output, states

    def generate(self, start_sequence, prediction_length=1000):
        self.eval()
        with torch.no_grad():
            
            hidden_states = None

            # Get last value from start sequence
            current = start_sequence[:, -1:, :]  # Shape: [batch=1, time=1, feature=1]
            predictions = []
            
            for _ in range(prediction_length):
                # Forward pass with just the current value
                output, hidden_states = self(current, hidden_states)
                next_value = output[:, -1:, :]  # Shape: [batch=1, time=1, feature=1]
                predictions.append(next_value.squeeze().item())
                # Update current with the predicted value
                current = next_value
            
            return predictions

class SineWaveDataset(torch.utils.data.Dataset):
    def __init__(self, seq_length=100, num_samples=10000, frequencies=[1.0], amplitudes=[1.0], 
                 sample_rate=100, noise_level=0.0):
        self.seq_length = seq_length
        
        # Generate time points
        t = np.linspace(0, num_samples/sample_rate, num_samples)
        
        # Generate sine wave with multiple frequencies
        signal = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)
            
        # Add noise if specified
        if noise_level > 0:
            signal += np.random.normal(0, noise_level, signal.shape)
            
        # Convert to tensor
        self.data = torch.FloatTensor(signal).to(device)
        self.data_size = len(self.data) - seq_length - 1
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length].unsqueeze(-1)
        target = self.data[idx + 1:idx + self.seq_length + 1].unsqueeze(-1)
        return sequence, target

def train_epoch(model, data_loader, criterion, optimizer, writer=None, epoch=None):
    model.train()
    total_loss = 0
    
    for batch_idx, (sequence, target) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(sequence)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if writer is not None and batch_idx % 100 == 0:
            writer.add_scalar('Loss/train_step', loss.item(), 
                            epoch * len(data_loader) + batch_idx)
    
    avg_loss = total_loss / len(data_loader)
    if writer is not None:
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    
    return avg_loss

def plot_prediction_vs_truth(predictions, ground_truth, save_path=None):
    """Create a plot comparing model predictions with ground truth data."""
    fig = plt.figure(figsize=(12, 6))
    t = np.arange(len(predictions))
    
    plt.plot(t, ground_truth, 'b-', label='Ground Truth', alpha=0.5)
    plt.plot(t, predictions, 'r--', label='Prediction', alpha=0.8)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Sine Wave Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig

@hydra.main(version_base=None, config_path="../config", config_name="sine_config")
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create save directory using Hydra's output dir
    save_dir = Path(HydraConfig.get().runtime.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=str(save_dir))
    
    # Create dataset
    dataset = SineWaveDataset(
        seq_length=cfg.data.seq_length,
        num_samples=cfg.data.num_samples,
        frequencies=cfg.data.frequencies,
        amplitudes=cfg.data.amplitudes,
        sample_rate=cfg.data.sample_rate,
        noise_level=cfg.data.noise_level
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = TimeSeriesMinGRU(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_points=cfg.model.num_points
    ).to(device)
    
    # Initialize optimizer and loss
    optimizer = Lion(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(cfg.training.num_epochs):
        avg_loss = train_epoch(model, data_loader, criterion, optimizer, 
                             writer=writer, epoch=epoch)
        
        logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        
        # Generate and plot predictions periodically
        if epoch % cfg.training.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Get a sample sequence and ensure correct dimensions
                sequence, _ = dataset[0]
                sequence = sequence.unsqueeze(0)  # Add batch dimension
                
                # Generate predictions
                predictions = model.generate(sequence, prediction_length=200)
                
                # Get corresponding ground truth
                start_idx = 0  # Starting index in dataset
                ground_truth = dataset.data[start_idx:start_idx + 200].cpu().numpy()
                
                # Create and save plot
                fig = plot_prediction_vs_truth(
                    predictions, 
                    ground_truth,
                    save_path=save_dir / f'prediction_epoch_{epoch}.png'
                )
                
                # Add plot to tensorboard
                writer.add_figure('Predictions/comparison', fig, epoch)
                plt.close()
                
                # Log first prediction value
                writer.add_scalar('Evaluation/prediction_first_value', 
                                predictions[0], epoch)
    
    writer.close()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()

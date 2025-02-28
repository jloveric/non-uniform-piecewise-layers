import torch
import torch.nn as nn
import torch.optim as optim
import requests
from pathlib import Path
import numpy as np
from non_uniform_piecewise_layers.adaptive_piecewise_mingru import MinGRUStack
from lion_pytorch import Lion
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# Set project root for data storage
os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent.absolute())

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class CharLevelMinGRU(nn.Module):
    def __init__(self, n_chars, hidden_size=256, num_layers=2, num_points=10, position_init="random"):
        super().__init__()
        self.n_chars = n_chars
        # Embedding layer to convert character indices to vectors
        self.embedding = nn.Embedding(n_chars, hidden_size)
        # MinGRU stack
        self.rnn = MinGRUStack(
            input_dim=hidden_size,
            state_dim=hidden_size,
            out_features=n_chars,
            layers=num_layers,
            num_points=num_points,
            position_init=position_init
        )
        # Output layer
        #self.fc = nn.Linear(hidden_size, n_chars)
    
    def forward(self, x, h=None):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, hidden_size)
        hidden, states = self.rnn(x, h)  # (batch, seq_len, hidden_size)
        #output = self.fc(hidden)  # (batch, seq_len, n_chars)
        return hidden, states #output, states

    def generate(self, start_char, max_length=1000, temperature=0.8):
        self.eval()
        with torch.no_grad():
            current = torch.tensor([[start_char]], dtype=torch.long, device=device)
            output_chars = []
            hidden_states = None  # Will be initialized as list of states in forward pass
            
            for _ in range(max_length):
                # Forward pass
                logits, hidden_states = self(current, hidden_states)
                if temperature == 0:
                    # For temperature 0, just take the argmax
                    next_char = torch.argmax(logits[0, -1]).unsqueeze(0)
                else:
                    # Apply temperature and sample
                    probs = (logits[0, -1] / temperature).softmax(dim=-1)
                    next_char = torch.multinomial(probs, 1)
                output_chars.append(next_char.item())
                current = next_char.unsqueeze(0)
            
            return output_chars

    def remove_add(self, x, h=None):
        x = self.embedding(x)
        return self.rnn.remove_add(x,h)

    def move_smoothest(self):
        return self.rnn.move_smoothest()

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_length=100, max_length=None):
        self.text = text[:max_length] if max_length is not None else text
        self.seq_length = seq_length
        # Use ASCII characters (0-127) instead of computing set from text
        self.chars = [chr(i) for i in range(128)]
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data_size = len(self.text) - seq_length - 1
        
        # Convert text to indices once, using default value for unknown chars
        self.text_indices = torch.tensor([self.char_to_idx.get(ch, 0) for ch in self.text], dtype=torch.long, device=device)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # Get sequence and target
        sequence = self.text_indices[idx:idx + self.seq_length]
        target = self.text_indices[idx + 1:idx + self.seq_length + 1]
        return sequence, target
    
    @property
    def vocab_size(self):
        return len(self.chars)

def load_tiny_shakespeare(url, cache_dir):
    """Download and load the Tiny Shakespeare dataset"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "tinyshakespeare.txt"
    
    if not cache_file.exists():
        print(f"Downloading Tiny Shakespeare dataset to {cache_file}...")
        response = requests.get(url)
        response.raise_for_status()
        cache_file.write_text(response.text)
    else:
        print(f"Using cached dataset from {cache_file}")
    
    return cache_file.read_text()

class PlateauDetector:
    def __init__(self, patience=5, min_delta=1e-4, mode='min'):
        """
        Initialize plateau detector
        Args:
            patience: Number of epochs to wait before triggering plateau detection
            min_delta: Minimum change in metric to be considered as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.history = []
        
    def update(self, value):
        """
        Update the plateau detector with a new value
        Returns True if plateau is detected
        """
        self.history.append(value)
        
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
            
        if improved:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
            
    def reset(self):
        """Reset the plateau detector"""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0

def train_epoch(model, data_loader, criterion, optimizer, writer=None, epoch=None, remove_add_every_n_batches=10, plateau_mode=False, plateau_adjustments=3, error_tracking_batches=5,adapt="move"):
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    batch_since_last_remove_add = 0
    max_error = None
    max_error_input = None
    
    # For tracking errors across multiple batches
    error_tracking = []
    
    # In plateau mode, calculate when to do the remove_add operations
    if plateau_mode:
        total_batches = len(data_loader)
        # Spread the adjustments evenly across the epoch
        adjustment_points = [total_batches // (plateau_adjustments + 1) * (i + 1) for i in range(plateau_adjustments)]
    
    for i, (sequences, targets) in enumerate(tqdm(data_loader, desc="Training")):
        sequences = sequences.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output, h = model(sequences)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = output.view(-1, output.size(-1)).argmax(dim=1)
        correct = (predictions == targets.view(-1)).float().mean()
        total_accuracy += correct.item()
        num_batches += 1
        
        batch_since_last_remove_add += 1
        
        if adapt == "largest_error":
            # Calculate error for this batch
            with torch.no_grad():
                B, T, V = output.shape
                error = output.view(-1, output.size(-1)) - torch.nn.functional.one_hot(targets.view(-1), num_classes=output.size(-1)).float()
                error = torch.sum(torch.abs(error.view(B, T, V)), dim=2)
                batch_max_error, flat_idx = error.view(-1).max(dim=0)
                B_idx, T_idx = torch.unravel_index(flat_idx, error.shape)
                
                # Store error information for this batch
                x_error = sequences[B_idx, T_idx].unsqueeze(0).unsqueeze(0)
                if T_idx > 0:
                    h_error = [this_h[B_idx, T_idx-1].unsqueeze(0) for this_h in h]
                else:
                    h_error = 0
                    
                error_tracking.append({
                    'error': batch_max_error,
                    'x_error': x_error,
                    'h_error': h_error,
                    'batch_idx': i
                })
                
                # Keep only the last error_tracking_batches
                if len(error_tracking) > error_tracking_batches:
                    error_tracking.pop(0)
            
        # Determine if we should do remove_add based on mode
        should_remove_add = (
            (plateau_mode and i in adjustment_points) or
            (not plateau_mode and batch_since_last_remove_add >= remove_add_every_n_batches)
        )
        if adapt=="global_error":
            # Call remove_add if conditions are met
            if should_remove_add and error_tracking:
                # Find the maximum error across all tracked batches
                max_error_batch = max(error_tracking, key=lambda x: x['error'])
                max_error_input = (max_error_batch['x_error'], max_error_batch['h_error'])
                
                success = model.remove_add(*max_error_input)
                if success:
                    print(f'Moved points! Using error from batch {max_error_batch["batch_idx"]}')
                    optimizer = Lion(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                
                # Clear error tracking after remove_add
                error_tracking = []
                batch_since_last_remove_add = 0
        elif adapt=="move":
            if batch_since_last_remove_add >= remove_add_every_n_batches:
                model.move_smoothest()
                optimizer = Lion(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                batch_since_last_remove_add = 0
        elif adapt==None:
            pass
        else:
            raise ValueError(f"Adaptation {adapt} not recognized")
            
        # Log batch loss and accuracy to tensorboard
        if writer is not None and epoch is not None:
            writer.add_scalar('Loss/batch', loss.item(), epoch * len(data_loader) + i)
            writer.add_scalar('Accuracy/batch', correct.item(), epoch * len(data_loader) + i)
            
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / num_batches
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/epoch', avg_accuracy, epoch)
    
    return avg_loss, avg_accuracy, optimizer

@hydra.main(version_base=None, config_path="config", config_name="shakespeare_generation")
def main(cfg: DictConfig):
    logger.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    logger.info(f"Current working directory : {os.getcwd()}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Get Hydra's output directory for this run
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Create tensorboard writer in Hydra's output directory
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    # Load data
    text = load_tiny_shakespeare(cfg.data.url, cfg.data.cache_dir)
    print(f"Total text length: {len(text)}")
    dataset = ShakespeareDataset(text, seq_length=cfg.data.seq_length, max_length=cfg.data.max_length)
    print(f"Using text length: {len(dataset.text)}")
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0
    )

    print('Building base model')
    # Initialize model
    model = CharLevelMinGRU(
        n_chars=dataset.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_points=cfg.model.num_points,
        position_init=cfg.model.position_init
    ).to(device)  # Move model to GPU
    print('Finished building model')
    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=cfg.training.learning_rate)
    
    # Log model architecture
    sample_input = torch.zeros((1, cfg.data.seq_length), dtype=torch.long, device=device)
    writer.add_graph(model, (sample_input,))
    
    # Create checkpoint directory inside Hydra's output directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_loss = float('inf')
    # Store original remove_add_every_n_batches value
    original_remove_add_every_n_batches = cfg.training.remove_add_every_n_batches
    
    # Initialize plateau detectors for both loss and accuracy
    loss_plateau_detector = PlateauDetector(
        patience=cfg.training.plateau_patience,
        min_delta=cfg.training.plateau_min_delta,
        mode='min'
    )
    
    plateau_mode = False  # Track if we're in plateau recovery mode
    
    for epoch in range(cfg.training.num_epochs):
        loss, accuracy, optimizer = train_epoch(
            model, data_loader, criterion, optimizer, writer, epoch, 
            remove_add_every_n_batches=cfg.training.remove_add_every_n_batches,
            plateau_mode=plateau_mode,
            plateau_adjustments=cfg.training.plateau_adjustments,
            error_tracking_batches=cfg.training.error_tracking_batches
        )
        
        # Check for plateaus
        loss_plateaued = loss_plateau_detector.update(loss)
        
        if loss_plateaued:
            logger.info(f"Plateau detected at epoch {epoch}. Entering plateau recovery mode.")
            plateau_mode = True
            loss_plateau_detector.reset()
        else:
            if plateau_mode:
                logger.info(f"Exiting plateau recovery mode at epoch {epoch}.")
            plateau_mode = False
        
        # Generate sample text with different temperatures
        if epoch % cfg.training.sample_every == 0:
            model.eval()
            start_char = dataset.text[0]
            start_idx = dataset.char_to_idx[start_char]
            
            # Generate with different temperatures
            temperatures = [0.0, 0.5, 1.0]
            for temp in temperatures:
                generated_chars = model.generate(
                    start_idx,
                    max_length=200,
                    temperature=temp
                )
                generated_text = ''.join([dataset.idx_to_char[idx] for idx in generated_chars])
                writer.add_text(f'Generated Text (temp={temp})', generated_text, epoch)
            
            model.train()
        
        # Save checkpoint if best loss
        if loss < best_loss:
            best_loss = loss
            checkpoint_path = os.path.join(checkpoint_dir, f"model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': OmegaConf.to_container(cfg)
            }, checkpoint_path)
        
        # Save periodic checkpoint
        if epoch % cfg.training.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': OmegaConf.to_container(cfg)
            }, checkpoint_path)
    
    writer.close()

if __name__ == "__main__":
    main()

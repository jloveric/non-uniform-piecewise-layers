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

logger = logging.getLogger(__name__)

# Set project root for data storage
os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent.absolute())

class CharLevelMinGRU(nn.Module):
    def __init__(self, n_chars, hidden_size=256, num_layers=2, num_points=10):
        super().__init__()
        self.n_chars = n_chars
        # Embedding layer to convert character indices to vectors
        self.embedding = nn.Embedding(n_chars, hidden_size)
        # MinGRU stack
        self.rnn = MinGRUStack(
            input_dim=hidden_size,
            state_dim=hidden_size,
            out_features=hidden_size,
            layers=num_layers,
            num_points=num_points
        )
        # Output layer
        self.fc = nn.Linear(hidden_size, n_chars)
    
    def forward(self, x, h=None):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, hidden_size)
        hidden, states = self.rnn(x, h)  # (batch, seq_len, hidden_size)
        output = self.fc(hidden)  # (batch, seq_len, n_chars)
        return output, states

    def generate(self, start_char, max_length=1000, temperature=0.8):
        self.eval()
        with torch.no_grad():
            current = torch.tensor([[start_char]], dtype=torch.long)
            output_chars = []
            hidden_states = None  # Will be initialized as list of states in forward pass
            
            for _ in range(max_length):
                # Forward pass
                logits, hidden_states = self(current, hidden_states)
                # Apply temperature
                probs = (logits[0, -1] / temperature).softmax(dim=-1)
                # Sample next character
                next_char = torch.multinomial(probs, 1)
                output_chars.append(next_char.item())
                current = next_char.unsqueeze(0)
            
            return output_chars

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
        self.text_indices = torch.tensor([self.char_to_idx.get(ch, 0) for ch in self.text], dtype=torch.long)
    
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

def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for sequences, targets in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        output, _ = model(sequences)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

@hydra.main(version_base=None, config_path="config", config_name="shakespeare_generation")
def main(cfg: DictConfig):
    logger.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    logger.info(f"Current working directory : {os.getcwd()}")
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
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
        num_points=cfg.model.num_points
    )
    print('Finished building model')
    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        loss = train_epoch(model, data_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # Generate sample text
        if (epoch + 1) % 1 == 0:
            start_char = dataset.char_to_idx[text[0]]
            generated = model.generate(
                start_char, 
                max_length=cfg.generation.max_length,
                temperature=cfg.generation.temperature
            )
            generated_text = ''.join([dataset.idx_to_char[idx] for idx in generated])
            print("\nGenerated text:")
            print(generated_text)
            print()

if __name__ == "__main__":
    main()

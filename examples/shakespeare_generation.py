import torch
import torch.nn as nn
import torch.optim as optim
import requests
from pathlib import Path
import numpy as np
from non_uniform_piecewise_layers.adaptive_piecewise_mingru import MinGRUStack
from lion_pytorch import Lion

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
            h = None
            
            for _ in range(max_length):
                # Forward pass
                logits, h = self(current, h)
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
        # Get sequence starting at random position
        sequence = self.text_indices[idx:idx + self.seq_length]
        target = self.text_indices[idx + 1:idx + self.seq_length + 1]
        return sequence, target
    
    @property
    def vocab_size(self):
        return len(self.chars)

def load_tiny_shakespeare():
    """Download and load the Tiny Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    shakespeare_path = Path("tiny_shakespeare.txt")
    
    if not shakespeare_path.exists():
        print("Downloading Tiny Shakespeare dataset...")
        response = requests.get(url)
        shakespeare_path.write_text(response.text)
    
    return shakespeare_path.read_text()

def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x_batch, y_batch in data_loader:
        optimizer.zero_grad()
        output, _ = model(x_batch)
        loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    # Hyperparameters
    hidden_size = 32
    num_layers = 2
    batch_size = 4
    seq_length = 100
    num_epochs = 10
    learning_rate = 0.001
    max_length = 10000  # Using first 10K characters for quick testing
    num_points=3

    # Load data
    text = load_tiny_shakespeare()
    print(f"Total text length: {len(text)}")
    dataset = ShakespeareDataset(text, seq_length=seq_length, max_length=max_length)
    print(f"Using text length: {len(dataset.text)}")
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print('Building base model')
    # Initialize model
    model = CharLevelMinGRU(
        n_chars=dataset.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_points=num_points
    )
    print('Finished building model')
    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        loss = train_epoch(model, data_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # Generate sample text
        if (epoch + 1) % 1 == 0:
            start_char = dataset.char_to_idx[text[0]]
            generated = model.generate(start_char, max_length=200)
            generated_text = ''.join([dataset.idx_to_char[idx] for idx in generated])
            print("\nGenerated text:")
            print(generated_text)
            print()

if __name__ == "__main__":
    main()

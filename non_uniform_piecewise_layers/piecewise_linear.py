import torch
import torch.nn as nn

class NonUniformPiecewiseLinear(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_points: int):
        """
        Initialize a non-uniform piecewise linear layer.
        
        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output features
            num_points (int): Number of points per piecewise function
        """
        super().__init__()
        
        # Initialize the x positions (must be monotonically increasing)
        self.positions = nn.Parameter(
            torch.linspace(-1, 1, num_points).repeat(num_inputs, num_outputs, 1)
        )
        
        # Initialize the y values
        self.values = nn.Parameter(
            torch.randn(num_inputs, num_outputs, num_points) * 0.1
        )
        
        print('shape.position', self.positions.shape)
        print('shape.values', self.values.shape)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_points = num_points

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs)
        """
        batch_size = x.shape[0]
        
        # Expand x for broadcasting: (batch_size, num_inputs, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Expand dimensions for broadcasting
        # x_expanded: (batch_size, num_inputs, 1)
        # positions: (num_inputs, num_outputs, num_points)
        # We want: (batch_size, num_inputs, num_outputs, num_points)
        x_broad = x_expanded.unsqueeze(2)  # (batch_size, num_inputs, 1, 1)
        pos_broad = self.positions.unsqueeze(0)  # (1, num_inputs, num_outputs, num_points)
        
        # Find which interval each x value falls into
        # mask shape: (batch_size, num_inputs, num_outputs, num_points-1)
        mask = (x_broad >= pos_broad[..., :-1]) & (x_broad < pos_broad[..., 1:])
        
        # Prepare positions and values for vectorized computation
        # Shape: (1, num_inputs, num_outputs, num_points-1)
        x0 = self.positions[..., :-1].unsqueeze(0)
        x1 = self.positions[..., 1:].unsqueeze(0)
        y0 = self.values[..., :-1].unsqueeze(0)
        y1 = self.values[..., 1:].unsqueeze(0)
        
        # Compute slopes for all segments at once
        # Shape: (1, num_inputs, num_outputs, num_points-1)
        slopes = (y1 - y0) / (x1 - x0)
        
        # Compute all interpolated values at once
        # x_broad shape: (batch_size, num_inputs, 1, 1)
        # Shape: (batch_size, num_inputs, num_outputs, num_points-1)
        interpolated = y0 + (x_broad - x0) * slopes
        
        # Apply mask and sum over the segments dimension
        # Shape after mask: (batch_size, num_inputs, num_outputs)
        output = (interpolated * mask).sum(dim=-1)
        
        # Handle edge cases
        left_mask = (x_broad < pos_broad[..., 0:1]).squeeze(-1)  # (batch_size, num_inputs, num_outputs)
        right_mask = (x_broad >= pos_broad[..., -1:]).squeeze(-1)  # (batch_size, num_inputs, num_outputs)
        
        # Add edge values where x is outside the intervals
        output = output + (self.values[..., 0].unsqueeze(0) * left_mask)
        output = output + (self.values[..., -1].unsqueeze(0) * right_mask)
        
        # Sum over the input dimension to get final output
        # Shape: (batch_size, num_outputs)
        output = output.sum(dim=1)
        
        return output

    def enforce_monotonic(self):
        """
        Enforce monotonically increasing positions by sorting them.
        Also sorts the corresponding values to maintain position-value mapping.
        Call this during training if necessary.
        """
        with torch.no_grad():
            # Get the sorting indices for each input-output pair
            # positions shape: (num_inputs, num_outputs, num_points)
            sorted_indices = torch.argsort(self.positions, dim=-1)
            
            # Use advanced indexing to sort both positions and values
            # We need to create proper indices for the batch dimensions
            i = torch.arange(self.positions.size(0)).unsqueeze(1).unsqueeze(2)
            j = torch.arange(self.positions.size(1)).unsqueeze(0).unsqueeze(2)
            
            # Sort positions and values using the same indices
            self.positions.data = self.positions[i, j, sorted_indices]
            self.values.data = self.values[i, j, sorted_indices]

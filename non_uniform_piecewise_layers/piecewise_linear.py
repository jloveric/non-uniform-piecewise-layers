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

    def add_point_at_max_error(self, split_strategy: int = 0):
        """
        Add a new control point where the absolute gradient is largest, indicating
        where the error is most sensitive to changes.
        
        Args:
            split_strategy (int): How to split the interval:
                0: Add point halfway to left neighbor
                1: Add point halfway to right neighbor
                2: Move existing point left 1/3 and add new point 1/3 right
        
        Returns:
            bool: True if point was successfully added, False otherwise
        
        Note:
            This method should be called after a forward and backward pass,
            when gradients have been accumulated.
        """
        if self.values.grad is None:
            raise ValueError("No gradients available. Run backward() first.")
            
        with torch.no_grad():
            # Use absolute gradients of values as error estimate
            abs_grads = torch.abs(self.values.grad)  # (num_inputs, num_outputs, num_points)
            
            # Find the point with maximum gradient
            max_grad_flat = torch.argmax(abs_grads.view(-1))
            
            # Convert flat index to 3D indices using torch operations
            num_outputs = self.values.size(1)
            num_points = self.values.size(2)
            
            # First get point_idx (remainder when divided by num_points)
            point_idx = torch.remainder(max_grad_flat, num_points)
            
            # Then get output_idx from the remaining division
            temp_idx = torch.div(max_grad_flat, num_points, rounding_mode='floor')
            output_idx = torch.remainder(temp_idx, num_outputs)
            
            # Finally get input_idx
            input_idx = torch.div(temp_idx, num_outputs, rounding_mode='floor')
            
            # Get current positions and values
            old_positions = self.positions[input_idx, output_idx]
            old_values = self.values[input_idx, output_idx]
            
            # Create new tensors with space for one more point
            new_positions = torch.zeros(self.num_inputs, self.num_outputs, self.num_points + 1,
                                     device=self.positions.device)
            new_values = torch.zeros(self.num_inputs, self.num_outputs, self.num_points + 1,
                                   device=self.values.device)
            
            # Copy existing points
            new_positions[:, :, :self.num_points] = self.positions
            new_values[:, :, :self.num_points] = self.values
            
            # Calculate new point position based on strategy
            if split_strategy == 0 and point_idx > 0:
                # Add point halfway to left neighbor
                left_pos = old_positions[point_idx - 1]
                curr_pos = old_positions[point_idx]
                new_pos = (left_pos + curr_pos) / 2
                insert_idx = point_idx
            elif split_strategy == 1 and point_idx < self.num_points - 1:
                # Add point halfway to right neighbor
                curr_pos = old_positions[point_idx]
                right_pos = old_positions[point_idx + 1]
                new_pos = (curr_pos + right_pos) / 2
                insert_idx = point_idx + 1
            else:  # split_strategy == 2
                if point_idx >= self.num_points - 1:
                    return False
                    
                # Move current point left 1/3 and add new point 1/3 right
                left_pos = old_positions[point_idx]
                right_pos = old_positions[point_idx + 1]
                interval_size = right_pos - left_pos
                
                # Update current point position (1/3 to the left)
                new_positions[input_idx, output_idx, point_idx] = left_pos + interval_size / 3
                
                # New point position (1/3 to the right)
                new_pos = right_pos - interval_size / 3
                insert_idx = point_idx + 1
            
            # Linearly interpolate to get the value at the new position
            if insert_idx > 0:
                left_pos = old_positions[insert_idx - 1]
                left_val = old_values[insert_idx - 1]
                right_pos = old_positions[insert_idx]
                right_val = old_values[insert_idx]
                
                # Linear interpolation
                t = (new_pos - left_pos) / (right_pos - left_pos)
                new_value = left_val + t * (right_val - left_val)
            else:
                # If inserting at the start, use the value of the first point
                new_value = old_values[0]
            
            # Move all points after insert_idx one position to the right
            if insert_idx < self.num_points:
                new_positions[input_idx, output_idx, insert_idx+1:] = old_positions[insert_idx:]
                new_values[input_idx, output_idx, insert_idx+1:] = old_values[insert_idx:]
            
            # Insert the new point
            new_positions[input_idx, output_idx, insert_idx] = new_pos
            new_values[input_idx, output_idx, insert_idx] = new_value
            
            # Update the layer parameters
            self.positions = nn.Parameter(new_positions)
            self.values = nn.Parameter(new_values)
            self.num_points += 1
            
            return True

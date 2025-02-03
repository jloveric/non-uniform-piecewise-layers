import torch
import torch.nn as nn

class NonUniformPiecewiseLinear(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_points: int, position_range=(-1, 1)):
        """
        Initialize a non-uniform piecewise linear layer.
        
        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output features
            num_points (int): Number of points per piecewise function
            position_range (tuple): Tuple of (min, max) for allowed position range. Default is (-1, 1)
        """
        super().__init__()
        
        self.position_min, self.position_max = position_range
        
        # Initialize the x positions (must be monotonically increasing)
        self.positions = nn.Parameter(
            torch.linspace(self.position_min, self.position_max, num_points).repeat(num_inputs, num_outputs, 1)
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

        # Register buffer for absolute gradient accumulation
        #self.register_buffer('abs_grad_accumulation', None)
        
        # Register the hook for gradient accumulation
        """
        def accumulate_abs_grad_hook(grad):
            #print('grad.shape', grad.shape)
            if self.abs_grad_accumulation is None:
                self.abs_grad_accumulation = torch.zeros_like(grad)
            self.abs_grad_accumulation = torch.abs(grad)
            return grad
        
        self.values.register_hook(accumulate_abs_grad_hook)
        """

    def zero_abs_grad_accumulation(self):
        """Zero out the accumulated absolute gradients."""
        if self.abs_grad_accumulation is not None:
            self.abs_grad_accumulation.zero_()

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
            
            # Clamp positions to allowed range
            self.positions.data.clamp_(self.position_min, self.position_max)

    def compute_abs_grads(self, x):
        """
        Super slow computation so you only want to compute this periodically
        """
        output = self(x)
        #print('output.shape', output.shape,output.shape[0])
        grads = [torch.autograd.grad(output[element],self.parameters(), retain_graph=True) for element in range(output.shape[0])]
        #print('grads[0]',grads[0])
        abs_grad=[torch.flatten(torch.abs(torch.cat(grad)),start_dim=1).sum(dim=0) for grad in grads]
        #print('abs_grad.shape', abs_grad[0].shape)
        abs_grad = torch.stack(abs_grad).sum(dim=0)
        return abs_grad

    def add_point_at_max_error(self, abs_grad, split_strategy: int = 0):
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
            
        with torch.no_grad():
            # Use accumulated absolute gradients as error estimate
            abs_grads = abs_grad  # (num_inputs, num_outputs, num_points)
            
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

            if split_strategy==2:  # split_strategy == 2
                print(f"Strategy 2: point_idx={point_idx}, num_points={self.num_points}")
                
                if point_idx >= self.num_points - 1: 
                    split_strategy=0
                elif point_idx <= 0:
                    split_strategy=1
                
                else:
                    # Get the boundaries (points to left and right of max error point)
                    left_boundary = old_positions[point_idx - 1]
                    right_boundary = old_positions[point_idx + 1]
                    interval_size = right_boundary - left_boundary
                    
                    print(f"Strategy 2: Boundaries: left={left_boundary}, right={right_boundary}")
                    
                    # Calculate positions for the two points (evenly spaced)
                    first_third = left_boundary + interval_size / 3
                    second_third = left_boundary + 2 * interval_size / 3
                    
                    # Clamp the new positions to allowed range
                    first_third = torch.clamp(first_third, self.position_min, self.position_max)
                    second_third = torch.clamp(second_third, self.position_min, self.position_max)
                    
                    print(f"Strategy 2: New positions: first={first_third}, second={second_third}")
                    
                    # Get the original values for interpolation
                    left_val = old_values[point_idx - 1]
                    right_val = old_values[point_idx + 1]
                    
                    # Linear interpolation for both points
                    t1 = (first_third - left_boundary) / (right_boundary - left_boundary)
                    t2 = (second_third - left_boundary) / (right_boundary - left_boundary)
                    first_val = left_val + t1 * (right_val - left_val)
                    second_val = left_val + t2 * (right_val - left_val)
                    
                    # Update the existing point's position and value
                    new_positions[input_idx, output_idx, point_idx] = first_third
                    new_values[input_idx, output_idx, point_idx] = first_val
                    
                    # Set up for inserting the second point
                    new_pos = second_third
                    new_value = second_val
                    insert_idx = point_idx + 1
            
            
            print('point_idx', point_idx,'split_strategy', split_strategy)
            # Calculate new point position based on strategy
            if split_strategy == 0 and point_idx > 0:
                # Add point halfway to left neighbor
                left_pos = old_positions[point_idx - 1]
                curr_pos = old_positions[point_idx]
                new_pos = (left_pos + curr_pos) / 2
                new_pos = torch.clamp(new_pos, self.position_min, self.position_max)
                insert_idx = point_idx
            elif split_strategy == 1 and point_idx < self.num_points-1:
                # Add point halfway to right neighbor
                curr_pos = old_positions[point_idx]
                right_pos = old_positions[point_idx + 1]
                new_pos = (curr_pos + right_pos) / 2
                new_pos = torch.clamp(new_pos, self.position_min, self.position_max)
                insert_idx = point_idx + 1
            
            
            # Linearly interpolate to get the value at the new position
            if insert_idx > 0 and split_strategy != 2:  
                left_pos = old_positions[insert_idx - 1]
                left_val = old_values[insert_idx - 1]
                right_pos = old_positions[insert_idx]
                right_val = old_values[insert_idx]
                
                # Linear interpolation
                t = (new_pos - left_pos) / (right_pos - left_pos)
                new_value = left_val + t * (right_val - left_val)
            elif insert_idx == 0 and split_strategy != 2:
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
            
            print(f"Successfully added point. New num_points: {self.num_points}")
            return True

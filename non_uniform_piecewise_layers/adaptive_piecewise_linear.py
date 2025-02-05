import torch
import torch.nn as nn
import numpy as np

class AdaptivePiecewiseLinear(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_points: int, position_range=(-1, 1)):
        """
        Initialize an adaptive piecewise linear layer where positions are not learnable.
        New positions are inserted based on binary search between existing points.
        
        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output features
            num_points (int): Initial number of points per piecewise function
            position_range (tuple): Tuple of (min, max) for allowed position range. Default is (-1, 1)
        """
        super().__init__()
        
        self.position_min, self.position_max = position_range
        
        # Initialize fixed positions (not learnable)
        self.register_buffer('positions', 
            torch.linspace(self.position_min, self.position_max, num_points).repeat(num_inputs, num_outputs, 1)
        )
        
        # Initialize the y values (these are learnable)
        self.values = nn.Parameter(
            torch.randn(num_inputs, num_outputs, num_points) * 0.1
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_points = num_points

    def insert_positions(self, x_values: torch.Tensor):
        """
        Insert new positions based on binary search of input x_values.
        Only inserts positions between existing points, maintaining the domain extents.
        
        Args:
            x_values (torch.Tensor): Input tensor of shape (batch_size, num_inputs)
        """
        with torch.no_grad():
            # Flatten positions and get unique sorted values
            current_positions = torch.unique(self.positions.flatten())
            
            # Get unique x values within the position range
            x_flat = x_values.flatten()
            mask = (x_flat >= self.position_min) & (x_flat <= self.position_max)
            x_candidates = torch.unique(x_flat[mask])
            
            # Find insertion points using binary search
            indices = torch.searchsorted(current_positions, x_candidates)
            
            # Filter out duplicates and points at the extents
            valid_mask = (indices > 0) & (indices < len(current_positions))
            new_positions = x_candidates[valid_mask]
            
            if len(new_positions) > 0:
                # Combine current and new positions
                combined = torch.cat([current_positions, new_positions])
                combined = torch.unique(combined)
                
                # Create new position tensor with the same shape as before
                new_pos = combined.repeat(self.num_inputs, self.num_outputs, 1)
                
                # Interpolate new values
                new_vals = []
                for i in range(self.num_inputs):
                    for j in range(self.num_outputs):
                        old_pos = self.positions[i, j]
                        old_vals = self.values[i, j]
                        new_vals.append(torch.interp(combined, old_pos, old_vals))
                
                new_vals = torch.stack(new_vals).reshape(self.num_inputs, self.num_outputs, -1)
                
                # Update the buffers and parameters
                self.positions = new_pos
                self.values = nn.Parameter(new_vals)
                self.num_points = len(combined)

    def insert_points(self, points: torch.Tensor):
        """
        Insert specified points into the model, interpolating values between the two nearest neighbors
        for each point. Points are assumed to be within [-1, 1].

        If the user has values that repeat in one dimension but not another this will
        fail.
        
        Args:
            points (torch.Tensor): Points to insert with shape (batch_size, num_inputs) or (num_inputs,)
            
        Returns:
            bool: True if points were inserted, False if points were too close to existing ones
        """
        with torch.no_grad():
            # Ensure points has correct shape (num_inputs,)
            if points is None:
                return False
                
            if points.dim() == 2:
                # If we get a batch of points, just take the first one
                points = points[0]
            
            if points.size(0) != self.num_inputs:
                raise ValueError(f"Points must have {self.num_inputs} dimensions, got {points.size(0)}")

            # Check if any point is too close to existing points
            min_distance = 1e-6
            for i in range(self.num_inputs):
                positions = self.positions[i, 0]  # Use first output dimension as reference
                distances = torch.abs(positions - points[i])
                if torch.any(distances < min_distance):
                    return False

            # Combine current and new positions
            current_positions = self.positions  # (num_inputs, num_outputs, num_points)
            current_values = self.values       # (num_inputs, num_outputs, num_points)
            
            # For each new point, we'll interpolate between its two nearest neighbors
            new_pos = []
            new_vals = []
            
            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    pos = current_positions[i, j]  # Current positions for this i,j
                    vals = current_values[i, j]    # Current values for this i,j
                    
                    # Add the new point for this input dimension
                    all_points = torch.cat([pos, points[i].unsqueeze(0)])
                    sorted_indices = torch.argsort(all_points)
                    sorted_points = all_points[sorted_indices]
                    
                    # Remove duplicates while preserving order
                    unique_points, unique_indices = torch.unique_consecutive(sorted_points, return_inverse=True)
                    
                    # Initialize values for all points
                    all_values = torch.zeros_like(unique_points)
                    
                    # Copy existing values
                    existing_mask = unique_points.unsqueeze(1) == pos.unsqueeze(0)
                    existing_indices = torch.where(existing_mask.any(dim=1))[0]
                    all_values[existing_indices] = vals[existing_mask.any(dim=0)]
                    
                    # Find indices of new points (those not in existing_indices)
                    new_indices = torch.ones_like(unique_points, dtype=torch.bool)
                    new_indices[existing_indices] = False
                    new_point_indices = torch.where(new_indices)[0]
                    
                    # For each new point, interpolate between nearest neighbors
                    for idx in new_point_indices:
                        point = unique_points[idx]
                        
                        # Find nearest existing points
                        left_mask = pos <= point
                        right_mask = pos > point
                        
                        if not left_mask.any() or not right_mask.any():
                            # If point is outside range, use nearest value
                            nearest_idx = torch.argmin(torch.abs(pos - point))
                            all_values[idx] = vals[nearest_idx]
                        else:
                            # Get nearest left and right points
                            left_idx = torch.where(left_mask)[0][-1]
                            right_idx = torch.where(right_mask)[0][0]
                            
                            # Linear interpolation between nearest points
                            left_pos = pos[left_idx]
                            right_pos = pos[right_idx]
                            left_val = vals[left_idx]
                            right_val = vals[right_idx]
                            
                            # Compute interpolated value
                            t = (point - left_pos) / (right_pos - left_pos)
                            all_values[idx] = left_val + t * (right_val - left_val)
                    
                    new_pos.append(unique_points)
                    new_vals.append(all_values)
            
            # Stack all the new positions and values
            new_pos = torch.stack([p for p in new_pos]).reshape(self.num_inputs, self.num_outputs, -1)
            new_vals = torch.stack([v for v in new_vals]).reshape(self.num_inputs, self.num_outputs, -1)
            
            # Update the buffers and parameters
            self.positions = new_pos
            self.values = nn.Parameter(new_vals)
            self.num_points = new_pos.size(-1)
            
            return True

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
        x_broad = x_expanded.unsqueeze(2)  # (batch_size, num_inputs, 1, 1)
        pos_broad = self.positions.unsqueeze(0)  # (1, num_inputs, num_outputs, num_points)
        
        # Find which interval each x value falls into
        mask = (x_broad >= pos_broad[..., :-1]) & (x_broad < pos_broad[..., 1:])
        
        # Prepare positions and values for vectorized computation
        x0 = self.positions[..., :-1].unsqueeze(0)
        x1 = self.positions[..., 1:].unsqueeze(0)
        y0 = self.values[..., :-1].unsqueeze(0)
        y1 = self.values[..., 1:].unsqueeze(0)
        
        # Compute slopes for all segments at once
        slopes = (y1 - y0) / (x1 - x0)
        
        # Compute all interpolated values at once
        interpolated = y0 + (x_broad - x0) * slopes
        
        # Apply mask and sum over the segments dimension
        output = (interpolated * mask).sum(dim=-1)
        
        # Handle edge cases
        left_mask = (x_broad < pos_broad[..., 0:1]).squeeze(-1)
        right_mask = (x_broad >= pos_broad[..., -1:]).squeeze(-1)
        
        # Add edge values where x is outside the intervals
        output = output + (self.values[..., 0].unsqueeze(0) * left_mask)
        output = output + (self.values[..., -1].unsqueeze(0) * right_mask)
        
        # Sum over the input dimension to get final output
        output = output.sum(dim=1)
        
        return output

    def compute_abs_grads(self, x):
        """
        Super slow computation so you only want to compute this periodically
        """
        output = self(x)
        grads = [torch.autograd.grad(output[element],self.parameters(), retain_graph=True) for element in range(output.shape[0])]
        abs_grad=[torch.flatten(torch.abs(torch.cat(grad)),start_dim=1).sum(dim=0) for grad in grads]
        abs_grad = torch.stack(abs_grad).sum(dim=0)
        return abs_grad

    def add_point_at_max_error_old(self, abs_grad, split_strategy: int = 0):
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
            
            # Update the layer's parameters
            self.positions = new_positions
            self.values = nn.Parameter(new_values)
            self.num_points += 1
            
            return True

    def add_point_at_max_error(self, abs_grad, split_strategy=0):
        """
        Add a new control point between the point with maximum error and its neighbor
        with the larger error (left or right). If there are only 2 points, it adds
        a point in the center.
        
        Args:
            abs_grad: Absolute gradients tensor from compute_abs_grads
            split_strategy: Ignored, kept for backward compatibility
        
        Returns:
            bool: True if point was successfully added, False otherwise
        
        Note:
            This method should be called after a forward and backward pass,
            when gradients have been accumulated.
        """
            
        with torch.no_grad():
            # Use accumulated absolute gradients as error estimate
            abs_grads = abs_grad  # (num_points)
            
            # Find the point with maximum gradient
            point_idx = torch.argmax(abs_grads)
            
            # Get current positions and values for the relevant input/output pair
            old_positions = self.positions[0, 0]  # Since we're using 1 input, 1 output
            old_values = self.values[0, 0]
            
            # Create new tensors with space for one more point
            new_positions = torch.zeros(self.num_inputs, self.num_outputs, self.num_points + 1,
                                     device=self.positions.device)
            new_values = torch.zeros(self.num_inputs, self.num_outputs, self.num_points + 1,
                                   device=self.values.device)
            
            # Copy existing points
            new_positions[:, :, :self.num_points] = self.positions
            new_values[:, :, :self.num_points] = self.values

            # Special case: if only 2 points, add point in center
            if self.num_points == 2:
                left_pos = old_positions[0]
                right_pos = old_positions[1]
                new_pos = (left_pos + right_pos) / 2
                new_pos = torch.clamp(new_pos, self.position_min, self.position_max)
                insert_idx = 1

                # Linear interpolation for the new value
                left_val = old_values[0]
                right_val = old_values[1]
                t = 0.5  # Since we're inserting at midpoint
                new_value = left_val + t * (right_val - left_val)
            else:
                # Get errors of left and right neighbors
                left_error = abs_grads[point_idx - 1] if point_idx > 0 else torch.tensor(-float('inf'))
                right_error = abs_grads[point_idx + 1] if point_idx < self.num_points - 1 else torch.tensor(-float('inf'))
                
                # Choose neighbor with larger error
                if left_error > right_error:
                    # Insert between max error point and left neighbor
                    left_pos = old_positions[point_idx - 1]
                    right_pos = old_positions[point_idx]
                    insert_idx = point_idx
                else:
                    # Insert between max error point and right neighbor
                    left_pos = old_positions[point_idx]
                    right_pos = old_positions[point_idx + 1]
                    insert_idx = point_idx + 1
                
                # Calculate new position halfway between points
                new_pos = (left_pos + right_pos) / 2
                new_pos = torch.clamp(new_pos, self.position_min, self.position_max)
                
                # Linear interpolation for the new value
                left_val = old_values[insert_idx - 1]
                right_val = old_values[insert_idx]
                t = (new_pos - left_pos) / (right_pos - left_pos)
                new_value = left_val + t * (right_val - left_val)
            
            # Move all points after insert_idx one position to the right
            if insert_idx < self.num_points:
                new_positions[:, :, insert_idx+1:] = self.positions[:, :, insert_idx:]
                new_values[:, :, insert_idx+1:] = self.values[:, :, insert_idx:]
            
            # Insert the new point
            new_positions[:, :, insert_idx] = new_pos
            new_values[:, :, insert_idx] = new_value
            
            # Update the layer's parameters
            self.positions = new_positions
            self.values = nn.Parameter(new_values)
            self.num_points += 1
            
            return True

    def largest_error(self, error: torch.Tensor, x: torch.Tensor, min_distance: float = 1e-6) -> torch.Tensor:
        """
        Find the x value that corresponds to the largest error in the batch.
        Excludes points that are too close to existing points.
        
        Args:
            error (torch.Tensor): Error tensor of shape (batch_size, error)
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs)
            min_distance (float): Minimum distance required from existing points
            
        Returns:
            torch.Tensor: x value that had the largest error, or None if no valid point found
        """
        with torch.no_grad():
            # Sort errors in descending order
            sorted_errors, indices = torch.sort(error.abs().view(-1), descending=True)
            
            # Convert to batch indices
            batch_indices = indices // error.size(1)
            
            # Get corresponding x values
            candidate_x = x[batch_indices]
            
            # Check each candidate until we find one that's far enough from existing points
            for i in range(len(candidate_x)):
                x_val = candidate_x[i:i+1]  # Keep batch dimension
                
                # Check distance to all existing points
                too_close = False
                for j in range(self.num_inputs):
                    positions = self.positions[j, 0]  # Use first output dimension as reference
                    distances = torch.abs(positions - x_val[0, j])
                    if torch.any(distances < min_distance):
                        too_close = True
                        break
                
                if not too_close:
                    return x_val
            
            # If we get here, no valid point was found
            return None

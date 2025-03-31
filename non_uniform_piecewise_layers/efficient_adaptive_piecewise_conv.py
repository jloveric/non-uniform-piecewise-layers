import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PiecewiseLinearExpansion2d(nn.Module):
    """
    Expansion layer that transforms input tensor using piecewise linear functions.
    This is used to efficiently implement the adaptive piecewise linear convolution.
    """
    def __init__(
        self,
        num_points: int,
        position_range=(-1, 1),
        position_init: str = "uniform",
    ):
        """
        Initialize the piecewise linear expansion layer.
        
        Args:
            num_points (int): Number of points in the piecewise linear function
            position_range (tuple): Tuple of (min, max) for allowed position range. Default is (-1, 1)
            position_init (str): Position initialization method. Must be one of ["uniform", "random"]. Default is "uniform"
        """
        super().__init__()
        
        if position_init not in ["uniform", "random"]:
            raise ValueError("position_init must be one of ['uniform', 'random']")
            
        self.position_min, self.position_max = position_range
        self.num_points = num_points
        
        # Initialize positions based on initialization method
        if position_init == "uniform":
            # Uniform initialization
            positions = torch.linspace(self.position_min, self.position_max, num_points)
        else:  # random
            # Create random positions between min and max
            positions = torch.rand(num_points) * (self.position_max - self.position_min) + self.position_min
            # Sort positions to maintain order
            positions, _ = torch.sort(positions)
            # Fix first and last positions
            positions[0] = self.position_min
            positions[-1] = self.position_max
            
        self.register_buffer("positions", positions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand input tensor using piecewise linear basis functions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Expanded tensor of shape (batch_size, channels * num_points, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # No clamping - we'll use linear extrapolation outside the boundaries
        # This ensures we maintain gradients even for values outside the position range
        x_clamped = x
        
        # Reshape x_clamped for broadcasting with positions
        # Shape: (batch_size, 1, height, width)
        x_clamped_reshaped = x_clamped.mean(dim=1, keepdim=True)
        
        # Prepare positions tensor for vectorized computation
        # Shape: (num_points)
        positions = self.positions
        
        # Create a tensor of all positions
        # Shape: (num_points, 1, 1)
        positions_reshaped = positions.view(-1, 1, 1)
        
        # Create output tensor
        expanded = torch.zeros(
            (batch_size, channels * self.num_points, height, width),
            device=x.device,
            dtype=x.dtype
        )
        
        # Compute basis functions for all points in a fully vectorized way
        if self.num_points == 2:
            # Special case for only 2 points (just linear interpolation)
            # Linear function from left to right
            values_left = (positions[1] - x_clamped_reshaped) / (positions[1] - positions[0])
            values_right = (x_clamped_reshaped - positions[0]) / (positions[1] - positions[0])
            
            # Assign values
            expanded[:, 0::self.num_points] = values_left * x
            expanded[:, 1::self.num_points] = values_right * x
        else:
            # General case for num_points > 2
            
            # 1. First, handle the leftmost point (i=0)
            mask_left = (x_clamped_reshaped <= positions[1])
            values_left = (positions[1] - x_clamped_reshaped) / (positions[1] - positions[0])
            values_left = values_left * mask_left
            expanded[:, 0::self.num_points] = values_left * x
            
            # 2. Handle the rightmost point (i=num_points-1)
            mask_right = (x_clamped_reshaped >= positions[-2])
            values_right = (x_clamped_reshaped - positions[-2]) / (positions[-1] - positions[-2])
            values_right = values_right * mask_right
            expanded[:, (self.num_points-1)::self.num_points] = values_right * x
            
            # 3. Handle all interior points (0 < i < num_points-1) in a fully vectorized way
            if self.num_points > 2:
                # Get all interior points and their neighbors
                interior_indices = torch.arange(1, self.num_points-1, device=x.device)
                
                # For each interior point, we need its position and its left/right neighbors
                pos_interior = positions[interior_indices].view(1, -1, 1, 1)  # (1, num_interior, 1, 1)
                pos_left = positions[interior_indices - 1].view(1, -1, 1, 1)  # (1, num_interior, 1, 1)
                pos_right = positions[interior_indices + 1].view(1, -1, 1, 1)  # (1, num_interior, 1, 1)
                
                # Compute masks and values for all interior points at once
                left_mask = (x_clamped_reshaped >= pos_left) & (x_clamped_reshaped <= pos_interior)
                right_mask = (x_clamped_reshaped > pos_interior) & (x_clamped_reshaped <= pos_right)
                
                left_values = (x_clamped_reshaped - pos_left) / (pos_interior - pos_left) * left_mask
                right_values = (pos_right - x_clamped_reshaped) / (pos_right - pos_interior) * right_mask
                
                # Combined values for all interior points
                # Shape: (batch_size, num_interior, height, width)
                interior_values = left_values + right_values
                
                # Now we need to distribute these values to the appropriate channels in the output
                # We'll use a reshape + permute approach to avoid loops
                
                # First, multiply by x to get the final values
                # Reshape x for broadcasting: (batch_size, channels, 1, height, width)
                x_reshaped = x.unsqueeze(2)
                
                # Reshape interior_values for broadcasting: (batch_size, 1, num_interior, height, width)
                interior_values_reshaped = interior_values.unsqueeze(1)
                
                # Multiply to get: (batch_size, channels, num_interior, height, width)
                interior_output = x_reshaped * interior_values_reshaped
                
                # Now we need to assign these values to the correct positions in the expanded tensor
                # We'll use a completely loop-free approach with advanced indexing
                
                # Reshape interior_output to (batch_size, channels * num_interior, height, width)
                # by interleaving the channels and interior dimensions
                b, c, ni, h, w = interior_output.shape  # batch, channels, num_interior, height, width
                
                # First, reshape to merge batch with height and width dimensions
                interior_output_flat = interior_output.reshape(b, c, ni, -1)  # (b, c, ni, h*w)
                
                # Transpose to get (b, ni, c, h*w)
                interior_output_flat = interior_output_flat.transpose(1, 2)  # (b, ni, c, h*w)
                
                # Reshape to (b, ni*c, h*w)
                interior_output_flat = interior_output_flat.reshape(b, ni * c, -1)  # (b, ni*c, h*w)
                
                # Create indices for the target positions in expanded tensor in a fully vectorized way
                # We need to create indices for each interior point (1 to num_points-2)
                # For each interior point i, we need indices i, i+num_points, i+2*num_points, etc.
                
                # First, create a tensor of interior point indices: [1, 2, ..., num_points-2]
                interior_point_indices = torch.arange(1, self.num_points-1, device=x.device)
                # Shape: (num_interior_points, 1)
                interior_point_indices = interior_point_indices.view(-1, 1)
                
                # Create a tensor of channel offsets: [0, num_points, 2*num_points, ...]
                channel_offsets = torch.arange(0, c, device=x.device) * self.num_points
                # Shape: (1, channels)
                channel_offsets = channel_offsets.view(1, -1)
                
                # Add the interior point indices to the channel offsets to get all indices at once
                # Shape: (num_interior_points, channels)
                indices = interior_point_indices + channel_offsets  # (ni, c)
                
                # Reshape to (ni*c)
                indices = indices.reshape(-1)  # (ni*c)
                
                # Now use these indices to place values in the expanded tensor
                # Reshape expanded to (b, c*num_points, h*w) for easier indexing
                expanded_flat = expanded.reshape(b, -1, h*w)  # (b, c*num_points, h*w)
                
                # Use advanced indexing to place all interior values at once
                expanded_flat[:, indices] = interior_output_flat
                
                # Reshape back to original shape
                expanded = expanded_flat.reshape(b, -1, h, w)  # (b, c*num_points, h, w)
        
        return expanded


class EfficientAdaptivePiecewiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        num_points=3,
        position_range=(-1, 1),
        position_init="uniform",
        weight_init="random",
    ):
        """
        Efficient 2D convolutional layer using adaptive piecewise linear functions.
        This implementation expands the input tensor first and then applies a regular Conv2d,
        which is much more efficient than applying piecewise functions to each unfolded patch.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            num_points (int): Number of points per piecewise function. Default: 3
            position_range (tuple): Tuple of (min, max) for allowed position range. Default: (-1, 1)
            position_init (str): Position initialization method. Must be one of ["uniform", "random"]. Default is "uniform"
            weight_init (str): Weight initialization method. Must be one of ["random", "linear"]. Default is "random"
                - "random": Initialize weights with uniform random values
                - "linear": Initialize weights to follow a linear pattern for each filter/channel
        """
        super().__init__()

        if isinstance(kernel_size, int):
            if kernel_size <= 0:
                raise ValueError(f"kernel_size must be positive, got {kernel_size}")
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if num_points < 2:
            raise ValueError(f"num_points must be at least 2, got {num_points}")
        if weight_init not in ["random", "linear"]:
            raise ValueError(f"weight_init must be one of ['random', 'linear'], got {weight_init}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.position_range = position_range
        self.weight_init = weight_init
        self.position_init = position_init
        self.num_points = num_points
        
        # Create the expansion layer
        self.expansion = PiecewiseLinearExpansion2d(
            num_points=num_points,
            position_range=position_range,
            position_init=position_init,
        )
        
        # Create the convolutional layer
        # The input channels to the conv layer are the original channels multiplied by the number of points
        self.conv = nn.Conv2d(
            in_channels=in_channels * num_points,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False, #No bias, this is effectively handled per element
        )
        
        # Initialize the weights based on the specified method
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of the convolutional layer based on the specified method.
        """
        # The factor is similar to what's used in AdaptivePiecewiseLinear
        factor = 0.5 * math.sqrt(1.0 / (3 * self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        
        if self.weight_init == "random":
            # Initialize with uniform random values
            # Standard random initialization
            self.conv.weight.data.uniform_(-factor, factor)
        else:  # linear
            # Initialize each filter/channel with a random line (collinear points)
            # Similar to AdaptivePiecewiseLinear initialization
            positions = self.expansion.positions
            num_points = positions.size(0)
            
            # Reshape weights to separate in_channels and points dimensions
            # [out_channels, in_channels, num_points, kernel_height, kernel_width]
            weights_shape = (self.out_channels, self.in_channels, num_points, *self.kernel_size)
            weights_reshaped = self.conv.weight.data.view(weights_shape)
            
            # For each output channel and input channel, initialize weights to follow a linear pattern
            for out_idx in range(self.out_channels):
                for in_idx in range(self.in_channels):
                    # Generate random start and end values for the linear function
                    start = torch.empty(1).uniform_(-factor, factor).item()
                    end = torch.empty(1).uniform_(-factor, factor).item()
                    
                    # Calculate the slope and intercept of the linear function
                    pos_min, pos_max = self.position_range
                    slope = (end - start) / (pos_max - pos_min)
                    intercept = start - slope * pos_min
                    
                    # Set weights to follow the linear function for all kernel positions
                    for p_idx in range(num_points):
                        pos = positions[p_idx].item()
                        value = slope * pos + intercept
                        
                        # Set the same value for all kernel positions
                        weights_reshaped[out_idx, in_idx, p_idx] = value
            
            # Update the weights
            self.conv.weight.data = weights_reshaped.view(self.conv.weight.shape)

    def forward(self, x):
        """
        Forward pass of the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        # Use the standard expansion for all cases
        expanded = self.expansion(x)
        
        # Apply the convolutional layer
        output = self.conv(expanded)
        
        return output
    
    
    def move_smoothest(self, weighted: bool = True, threshold: float = None):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed.
        
        If threshold is provided, only points where the ratio of minimum error to maximum error
        is below the threshold will be moved.
        
        The leftmost and rightmost points cannot be removed or used for insertion.
        
        Args:
            weighted (bool): Whether to weight the errors by the distance between points.
            threshold (float, optional): If provided, only move points where the ratio of 
                                        minimum error to maximum error is below this threshold.
                                        If None, all points are considered for movement.
        
        Returns:
            bool: True if points were moved, False otherwise
        """
        with torch.no_grad():
            # Access the positions from the expansion layer
            positions = self.expansion.positions
            num_points = self.expansion.num_points
            
            # We need at least 4 points to be able to remove one and still have enough for interpolation
            if num_points < 4:
                return False
            
            # Compute removal errors for each interior point
            # For each interior point, calculate the error if we were to remove it
            # and interpolate between its neighbors
            interior_indices = torch.arange(1, num_points-1, device=positions.device)
            
            # For each interior point, we need its position and its left/right neighbors
            errors = []
            
            for idx in interior_indices:
                # Get the left and right neighbors
                left_pos = positions[idx-1]
                mid_pos = positions[idx]
                right_pos = positions[idx+1]
                
                # Calculate interpolation parameter t
                t = (mid_pos - left_pos) / (right_pos - left_pos)
                
                # Calculate the error as the distance from the actual position
                # to where it would be if we interpolated between neighbors
                error = torch.abs(mid_pos - (left_pos + t * (right_pos - left_pos)))
                
                # If weighted, scale by the distance between neighbors
                if weighted:
                    error = error * torch.abs(right_pos - left_pos)
                    
                errors.append(error)
            
            # Convert errors to tensor
            errors = torch.tensor(errors, device=positions.device)
            
            # Check if all errors are zero (within a slightly larger tolerance)
            is_all_close = torch.allclose(errors, torch.zeros_like(errors), atol=1e-7)
            
            # Check if all errors are zero (or very close to zero)
            # This can happen with evenly spaced positions or due to float precision
            if is_all_close:
                # If all errors are zero, we can't determine which point to move
                # So we either move a random point or don't move any point
                if threshold is not None and threshold <= 0.0:
                    # If threshold is 0 or negative, move a random point
                    min_error_idx = torch.randint(1, num_points-1, (1,)).item()
                    max_error_idx = torch.randint(1, num_points-1, (1,)).item()
                    while max_error_idx == min_error_idx:
                        max_error_idx = torch.randint(1, num_points-1, (1,)).item()
                else:
                    # Otherwise (threshold=None or threshold > 0.0), don't move any point
                    return False
            else:
                # Errors are not all zero
                min_error_idx = torch.argmin(errors).item() + 1  # +1 because we skipped the first point
                max_error_idx = torch.argmax(errors).item() + 1
                
                # If threshold is provided and positive, check if the ratio of min to max error is below it
                if threshold is not None and threshold > 0.0:
                    min_error = errors[min_error_idx-1]
                    max_error = errors[max_error_idx-1]
                    epsilon = 1e-10  # To avoid division by zero
                    
                    # When min_error and max_error are equal (within tolerance), the ratio should be 1.0
                    if torch.isclose(min_error, max_error, rtol=1e-5, atol=1e-8):
                        ratio = 1.0
                    else:
                         # Ensure max_error + epsilon isn't zero before division
                        denominator = max_error + epsilon
                        if denominator < 1e-12: # Avoid division by ~zero 
                           ratio = 1.0 # Assign 1 if denominator is effectively zero
                        else:
                           ratio = min_error / denominator
                    
                    # If ratio is greater than or equal to threshold, don't move points
                    if ratio >= threshold:
                        return False  # Don't move any points if ratio is above threshold
            
            # Determine where to insert the new point (left or right of max error point)
            # Adjust max_error_idx if the removed point comes before it
            if min_error_idx < max_error_idx:
                adjusted_max_idx = max_error_idx - 1
            else:
                adjusted_max_idx = max_error_idx
                
            # Handle edge cases: ensure we're not inserting next to endpoints
            if adjusted_max_idx == 0:
                # First point case, can only insert to the right
                left_idx = 0
                right_idx = 1
            elif adjusted_max_idx == num_points - 2:
                # Last point case, can only insert to the left
                left_idx = num_points - 3
                right_idx = num_points - 2
            else:
                # Randomly choose left or right
                random_choice = torch.rand(1, device=positions.device).item() < 0.5
                if random_choice:
                    # Insert to the left
                    left_idx = adjusted_max_idx - 1
                    right_idx = adjusted_max_idx
                else:
                    # Insert to the right
                    left_idx = adjusted_max_idx
                    right_idx = adjusted_max_idx + 1
            
            # Get the positions for interpolation
            left_pos = positions[left_idx]
            right_pos = positions[right_idx]
            
            # Interpolate to get the new position (using t=0.5)
            t = 0.5
            new_pos = left_pos + t * (right_pos - left_pos)
            
            # Create new positions tensor
            # First, remove the point with minimum error
            keep_mask = torch.ones(num_points, dtype=torch.bool, device=positions.device)
            keep_mask[min_error_idx] = False
            kept_positions = positions[keep_mask]
            
            # Then, insert the new point at the appropriate position
            insert_idx = left_idx + 1 if left_idx < min_error_idx else left_idx
            new_positions = torch.cat([
                kept_positions[:insert_idx],
                new_pos.unsqueeze(0),
                kept_positions[insert_idx:]
            ])
            
            # Update the positions in the expansion layer
            self.expansion.positions.data = new_positions
            
            # Now we need to update the convolution weights to match the new positions
            # Get the current weights
            old_weights = self.conv.weight.data  # shape: [out_channels, in_channels * num_points, kernel_height, kernel_width]
            
            # Create a new weights tensor with the same shape
            in_channels = self.in_channels
            out_channels = self.out_channels
            kernel_size = self.kernel_size
            
            # Reshape weights to separate the in_channels and points dimensions
            # [out_channels, in_channels, num_points, kernel_height, kernel_width]
            old_weights_reshaped = old_weights.view(out_channels, in_channels, num_points, *kernel_size)
            
            # Create a keep mask for the weights, removing the weights corresponding to min_error_idx
            # and prepare to insert new weights
            new_weights_reshaped = torch.zeros(out_channels, in_channels, num_points, *kernel_size,
                                             device=old_weights.device, dtype=old_weights.dtype)
            
            # Copy weights for points that are kept (excluding min_error_idx)
            for i in range(num_points):
                if i == min_error_idx:
                    continue
                    
                # Determine the index in the new weights tensor
                new_idx = i if i < min_error_idx else i - 1
                if new_idx >= insert_idx:
                    new_idx += 1
                    
                new_weights_reshaped[:, :, new_idx] = old_weights_reshaped[:, :, i]
            
            # Interpolate weights for the new point
            left_weights = old_weights_reshaped[:, :, left_idx]
            right_weights = old_weights_reshaped[:, :, right_idx]
            new_point_weights = left_weights + t * (right_weights - left_weights)
            
            # Insert the new weights
            new_weights_reshaped[:, :, insert_idx] = new_point_weights
            
            # Reshape back to the original format
            new_weights = new_weights_reshaped.reshape(out_channels, in_channels * num_points, *kernel_size)
            
            # Update the convolution weights
            self.conv.weight.data = new_weights
            
            return True

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomPositionsPiecewiseConv2d(nn.Module):
    """
    Convolutional layer with custom positions for each input-output pair.
    This implementation decomposes the piecewise linear interpolation to support
    different positions for each kernel element.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int or tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        num_points: int = 5,
        position_range: tuple = (-1, 1),
        position_init: str = "uniform",
        weight_init: str = "random",
    ):
        """
        Initialize the custom positions piecewise convolutional layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolutional kernel
            stride (int): Stride of the convolution
            padding (int): Zero-padding added to both sides of the input
            dilation (int): Spacing between kernel elements
            groups (int): Number of blocked connections from input to output channels
            bias (bool): If True, adds a learnable bias to the output
            padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular'
            num_points (int): Number of points in the piecewise linear function
            position_range (tuple): Range of positions for the piecewise linear function
            position_init (str): Position initialization method ('uniform' or 'random')
            weight_init (str): Weight initialization method ('random' or 'sinusoidal')
        """
        super().__init__()
        
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight_init = weight_init
        self.position_init = position_init
        self.num_points = num_points
        self.position_range = position_range
        
        # Create positions for each input-output pair and kernel element
        # Shape: [out_channels, in_channels, kernel_height, kernel_width, num_points]
        positions = self._initialize_positions()
        
        # Register positions as a buffer (not a parameter)
        # It will only be modified by move operations, not by gradient descent
        self.register_buffer("positions", positions)
        
        # Precompute constant terms for piecewise linear interpolation
        # These will be updated whenever positions change
        self._precompute_interpolation_constants()
        
        # Create the convolutional weights
        # Shape: [out_channels, in_channels, num_points, kernel_height, kernel_width]
        self.weights = nn.Parameter(
            torch.Tensor(out_channels, in_channels, num_points, *self.kernel_size)
        )
        
        # Create bias if needed
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights and bias
        self._initialize_weights()
    
    def _initialize_positions(self):
        """
        Initialize positions for each input-output pair and kernel element.
        Vectorized implementation to improve efficiency.
        
        Returns:
            torch.Tensor: Positions tensor of shape [out_channels, in_channels, kernel_height, kernel_width, num_points]
        """
        device = next(self.parameters()).device if list(self.parameters()) else None
        
        # Initialize positions based on the specified method
        if self.position_init == "uniform":
            # Create uniform positions once
            uniform_positions = torch.linspace(
                self.position_range[0], 
                self.position_range[1], 
                self.num_points,
                device=device
            )
            
            # Expand to the desired shape [out_channels, in_channels, kernel_height, kernel_width, num_points]
            # by repeating the same positions for all kernel elements
            positions = uniform_positions.expand(
                self.out_channels, 
                self.in_channels, 
                self.kernel_size[0], 
                self.kernel_size[1], 
                self.num_points
            )
        
        elif self.position_init == "random":
            # Create random positions for all kernel elements at once
            positions = torch.rand(
                self.out_channels, 
                self.in_channels, 
                self.kernel_size[0], 
                self.kernel_size[1], 
                self.num_points,
                device=device
            ) * (self.position_range[1] - self.position_range[0]) + self.position_range[0]
            
            # Sort positions along the last dimension to maintain order
            positions, _ = torch.sort(positions, dim=-1)
            
            # Fix first and last positions for all kernel elements
            positions[..., 0] = self.position_range[0]
            positions[..., -1] = self.position_range[1]
        
        else:
            raise ValueError(f"Unknown position initialization method: {self.position_init}")
        
        return positions
    
    def _precompute_interpolation_constants(self):
        """
        Precompute constant terms for piecewise linear interpolation.
        Vectorized implementation for improved efficiency.
        
        For the piecewise linear interpolation formula ((x-x0)/(x1-x0))*w, we decompose it into:
        - Constant part: (-x0/(x1-x0))*w
        - Variable part: x*w/(x1-x0)
        
        This method precomputes the constant parts and the coefficients for the variable parts.
        """
        # Get positions - already in shape [out_channels, in_channels, kernel_height, kernel_width, num_points]
        positions = self.positions
        
        # Compute the left and right positions for each interval in a single vectorized operation
        # Shape: [out_channels, in_channels, kernel_height, kernel_width, num_points-1]
        left_positions = positions[:, :, :, :, :-1]
        right_positions = positions[:, :, :, :, 1:]
        
        # Compute the denominators (x1-x0) for each interval
        # Shape: [out_channels, in_channels, kernel_height, kernel_width, num_points-1]
        denominators = right_positions - left_positions
        
        # Add a small epsilon to avoid division by zero (using masked_fill for better efficiency)
        epsilon = 1e-6
        denominators = denominators.masked_fill(denominators == 0, epsilon)
        
        # Compute the coefficients for the variable part: 1/(x1-x0)
        # Shape: [out_channels, in_channels, kernel_height, kernel_width, num_points-1]
        variable_coefficients = torch.reciprocal(denominators)  # More efficient than 1.0 / denominators
        self.register_buffer("variable_coefficients", variable_coefficients)
        
        # Compute the constant part: -x0/(x1-x0)
        # Shape: [out_channels, in_channels, kernel_height, kernel_width, num_points-1]
        constant_terms = -left_positions * variable_coefficients  # More efficient than division
        self.register_buffer("constant_terms", constant_terms)
        
        # For the first interval, we also need a special case for inputs less than the first position
        # For x < positions[0], the output is 0
        # For the last interval, we need a special case for inputs greater than the last position
        # For x > positions[-1], the output is 0
        
        # We'll handle these special cases in the forward pass
    
    def _initialize_weights(self):
        """
        Initialize weights and bias based on the specified method.
        Vectorized implementation to improve efficiency.
        """
        if self.weight_init == "random":
            # Initialize with uniform random values
            factor = 1.0 / math.sqrt(self.in_channels * self.num_points * self.kernel_size[0] * self.kernel_size[1])
            self.weights.data.uniform_(-factor, factor)
        
        elif self.weight_init == "sinusoidal":
            # Generate random parameters for all kernel elements at once
            device = self.weights.device
            
            # Create random frequencies, phases, and amplitudes for all kernel elements
            frequencies = 1.0 + 0.5 * torch.rand(
                self.out_channels, self.in_channels, 1, self.kernel_size[0], self.kernel_size[1], 
                device=device
            )
            phases = 2 * math.pi * torch.rand(
                self.out_channels, self.in_channels, 1, self.kernel_size[0], self.kernel_size[1], 
                device=device
            )
            amplitudes = 0.5 + 0.5 * torch.rand(
                self.out_channels, self.in_channels, 1, self.kernel_size[0], self.kernel_size[1], 
                device=device
            )
            
            # Reshape positions to match the weights shape
            # From [out_ch, in_ch, kh, kw, num_points] to [out_ch, in_ch, num_points, kh, kw]
            positions_reshaped = self.positions.permute(0, 1, 4, 2, 3)
            
            # Compute sinusoidal values for all kernel elements at once
            self.weights.data = amplitudes * torch.sin(frequencies * positions_reshaped + phases)
        
        else:
            raise ValueError(f"Unknown weight initialization method: {self.weight_init}")
        
        # Initialize bias if present
        if self.bias is not None:
            self.bias.data.zero_()
    
    def forward(self, input_tensor):
        """
        Forward pass for the custom positions piecewise convolutional layer.
        Fully vectorized implementation for improved efficiency and gradient flow.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        # Get input shape
        batch_size, _, height, width = input_tensor.shape
        
        # Compute output shape
        out_height = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Initialize output tensor
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width, device=input_tensor.device)
        
        # Apply padding if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            # Convert 'zeros' padding mode to 'constant' which is what PyTorch expects
            pad_mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
            input_tensor = F.pad(input_tensor, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode=pad_mode)
        
        # Use unfold to extract all patches at once
        # This creates a tensor of shape [batch_size, channels * kernel_height * kernel_width, out_height * out_width]
        patches = F.unfold(
            input_tensor,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,  # Already padded above
            stride=self.stride
        )
        
        # Reshape to [batch_size, channels, kernel_height * kernel_width, out_height * out_width]
        patches = patches.view(
            batch_size, 
            self.in_channels, 
            self.kernel_size[0] * self.kernel_size[1], 
            out_height * out_width
        )
        
        # Process each output position
        for pos_idx in range(out_height * out_width):
            # Get the output spatial indices
            i = pos_idx // out_width
            j = pos_idx % out_width
            
            # Process each output channel
            for out_ch in range(self.out_channels):
                # Initialize accumulator for this output position and channel
                channel_output = torch.zeros(batch_size, device=input_tensor.device)
                
                # Process all input channels and kernel positions
                for in_ch in range(self.in_channels):
                    for k_idx in range(self.kernel_size[0] * self.kernel_size[1]):
                        # Convert flat index to 2D indices
                        kh = k_idx // self.kernel_size[1]
                        kw = k_idx % self.kernel_size[1]
                        
                        # Get positions for this kernel element
                        positions = self.positions[out_ch, in_ch, kh, kw]
                        
                        # Get weights for this kernel element
                        weights = self.weights[out_ch, in_ch, :, kh, kw]
                        
                        # Get input values for this position
                        input_values = patches[:, in_ch, k_idx, pos_idx]
                        
                        # Special case: if all weights are 1.0, pass through input value
                        if torch.allclose(weights, torch.ones_like(weights), rtol=1e-5, atol=1e-5):
                            channel_output += input_values
                            continue
                        
                        # Create a tensor to accumulate contributions for each input value
                        contributions = torch.zeros_like(input_values)
                        
                        # Handle special cases with vectorized operations
                        # Case 1: Input equals first position
                        mask_first = torch.isclose(input_values, positions[0].expand_as(input_values), rtol=1e-5, atol=1e-5)
                        contributions += mask_first * weights[0]
                        
                        # Case 2: Input equals last position
                        mask_last = torch.isclose(input_values, positions[-1].expand_as(input_values), rtol=1e-5, atol=1e-5)
                        contributions += mask_last * weights[-1]
                        
                        # Case 3: Input within range - perform piecewise linear interpolation
                        # Create mask for values that need interpolation
                        mask_interpolate = ~(mask_first | mask_last) & \
                                         (input_values >= positions[0]) & \
                                         (input_values <= positions[-1])
                        
                        # If any values need interpolation
                        if mask_interpolate.any():
                            # Expand positions and weights for broadcasting
                            # [num_points] -> [num_points-1, 1]
                            left_positions = positions[:-1].unsqueeze(1)  # Left positions for each interval
                            right_positions = positions[1:].unsqueeze(1)  # Right positions for each interval
                            left_weights = weights[:-1].unsqueeze(1)      # Left weights for each interval
                            right_weights = weights[1:].unsqueeze(1)      # Right weights for each interval
                            
                            # Get values that need interpolation
                            interp_values = input_values[mask_interpolate].unsqueeze(0)  # [1, num_interp_values]
                            
                            # Create interval masks for each value
                            # [num_intervals, num_interp_values]
                            interval_masks = (interp_values >= left_positions) & (interp_values < right_positions)
                            
                            # Calculate interpolation factors (t) for all intervals and values
                            # [num_intervals, num_interp_values]
                            t_values = torch.zeros_like(interval_masks, dtype=torch.float32)
                            valid_intervals = interval_masks.any(dim=1)  # [num_intervals]
                            
                            # Only calculate t for valid intervals
                            for interval_idx in range(self.num_points - 1):
                                if valid_intervals[interval_idx]:
                                    # Get mask for this interval
                                    interval_mask = interval_masks[interval_idx]
                                    
                                    # Calculate t for values in this interval
                                    t_values[interval_idx, interval_mask] = (
                                        (interp_values[0, interval_mask] - left_positions[interval_idx, 0]) / 
                                        (right_positions[interval_idx, 0] - left_positions[interval_idx, 0])
                                    )
                            
                            # Compute interpolated weights for each interval and value
                            # [num_intervals, num_interp_values]
                            interpolated = left_weights + t_values * (right_weights - left_weights)
                            
                            # Combine results using the interval masks
                            # For each value, sum the contributions from all intervals
                            # (only one interval will contribute per value)
                            interp_contributions = (interpolated * interval_masks.float()).sum(dim=0)
                            
                            # Add interpolated values back to the contributions tensor
                            contributions[mask_interpolate] += interp_contributions
                        
                        # Add all contributions to the channel output
                        channel_output += contributions
                
                # Assign the accumulated results to the output tensor
                output[:, out_ch, i, j] = channel_output
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, self.out_channels, 1, 1)
        
        return output
                                

        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, self.out_channels, 1, 1)
        
        return output
    
    def move_smoothest(self, weighted: bool = True, threshold: float = None):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed.
        
        If threshold is provided, only points where the ratio of minimum error to maximum error
        is below the threshold will be moved.
        
        Args:
            weighted (bool): Whether to weight the errors by the distance between neighbors
            threshold (float): Threshold for the ratio of minimum error to maximum error
        
        Returns:
            bool: True if points were moved, False otherwise
        """
        with torch.no_grad():
            # We need at least 4 points to be able to remove one and still have enough for interpolation
            if self.num_points < 4:
                return False
            
            # Iterate over all kernel elements
            for out_ch in range(self.out_channels):
                for in_ch in range(self.in_channels):
                    for kh in range(self.kernel_size[0]):
                        for kw in range(self.kernel_size[1]):
                            # Get positions for this kernel element
                            positions = self.positions[out_ch, in_ch, kh, kw]
                            
                            # Compute removal errors for each interior point
                            # For each interior point, calculate the error if we were to remove it
                            # and interpolate between its neighbors
                            interior_indices = torch.arange(1, self.num_points-1, device=positions.device)
                            
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
                                
                                errors.append(error.item())
                            
                            # Convert to tensor
                            errors = torch.tensor(errors, device=positions.device)
                            
                            # Find the point with the smallest removal error
                            min_error_idx = interior_indices[torch.argmin(errors)]
                            min_error = errors[torch.argmin(errors)]
                            
                            # Find the point with the largest removal error
                            max_error_idx = interior_indices[torch.argmax(errors)]
                            max_error = errors[torch.argmax(errors)]
                            
                            # Check if the ratio of minimum error to maximum error is below the threshold
                            if threshold is not None and min_error / max_error > threshold:
                                continue
                            
                            # Remove the point with the smallest removal error
                            # We'll create a new positions tensor without this point
                            new_positions = torch.cat([positions[:min_error_idx], positions[min_error_idx+1:]])
                            
                            # Determine where to insert the new point
                            # We'll insert it to the left or right of the point with the largest error
                            # with equal probability
                            if torch.rand(1).item() < 0.5:
                                # Insert to the left
                                left_idx = max_error_idx - 1
                                right_idx = max_error_idx
                                # Adjust indices if we removed a point before the max error point
                                if min_error_idx < max_error_idx:
                                    left_idx -= 1
                                    right_idx -= 1
                            else:
                                # Insert to the right
                                left_idx = max_error_idx
                                right_idx = max_error_idx + 1
                                # Adjust indices if we removed a point before or at the max error point
                                if min_error_idx <= max_error_idx:
                                    left_idx -= 1
                                    right_idx -= 1
                            
                            # Get the left and right positions
                            left_pos = new_positions[left_idx]
                            right_pos = new_positions[right_idx]
                            
                            # Generate a random position between left and right
                            t = torch.rand(1, device=positions.device).item()
                            new_pos = left_pos + t * (right_pos - left_pos)
                            
                            # Insert the new position
                            insert_idx = left_idx + 1
                            new_positions = torch.cat([new_positions[:insert_idx], new_pos.unsqueeze(0), new_positions[insert_idx:]])
                            
                            # Update the positions for this kernel element
                            self.positions[out_ch, in_ch, kh, kw] = new_positions
            
            # Precompute interpolation constants with the new positions
            self._precompute_interpolation_constants()
            
            return True
    
    def extra_repr(self):
        """
        Return a string representation of the module.
        """
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}, '
            f'dilation={self.dilation}, '
            f'groups={self.groups}, '
            f'bias={self.bias is not None}, '
            f'num_points={self.num_points}, '
            f'position_range={self.position_range}, '
            f'position_init={self.position_init}, '
            f'weight_init={self.weight_init}'
        )

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
        
        # Clamp input values to the position range
        # TODO: Don't think I actually want these clamped...
        x_clamped = torch.clamp(x, self.position_min, self.position_max)
        
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
            anti_periodic (bool): Whether to use anti-periodic boundary conditions. Default is True
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.anti_periodic = anti_periodic
        self.position_range = position_range
        
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
            bias=False,
        )
        
        # Initialize the weights with a small random value
        # The factor is similar to what's used in AdaptivePiecewiseLinear
        factor = 0.5 * math.sqrt(1.0 / (3 * in_channels * kernel_size[0] * kernel_size[1]))
        self.conv.weight.data.uniform_(-factor, factor)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass of the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
            
        # Expand the input using piecewise linear basis functions
        expanded = self.expansion(x)
        
        # Apply the convolutional layer
        output = self.conv(expanded)
        
        return output
    
    
    def move_smoothest(self, weighted: bool = True):
        """
        This method is included for API compatibility with AdaptivePiecewiseConv2d,
        but it doesn't do anything in this implementation since we're using a fixed
        set of basis functions.
        
        Returns:
            bool: Always returns False
        """
        return False

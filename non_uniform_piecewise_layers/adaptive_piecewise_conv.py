import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaptive_piecewise_linear import AdaptivePiecewiseLinear

class AdaptivePiecewiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_points=3, position_range=(-1, 1)):
        """
        2D convolutional layer using adaptive piecewise linear functions.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            num_points (int): Initial number of points per piecewise function. Default: 3
            position_range (tuple): Tuple of (min, max) for allowed position range. Default: (-1, 1)
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
        
        # Each kernel position gets its own piecewise function
        # Total inputs = in_channels * kernel_height * kernel_width
        # Each output channel gets its own set of functions
        self.piecewise = AdaptivePiecewiseLinear(
            num_inputs=in_channels * kernel_size[0] * kernel_size[1],
            num_outputs=out_channels,
            num_points=num_points,
            position_range=position_range
        )
        
    def forward(self, x):
        """
        Forward pass of the convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = ((height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        out_width = ((width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
        
        # Add padding if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        
        # Extract patches using unfold
        # Shape: (batch_size, in_channels * kernel_height * kernel_width, out_height * out_width)
        patches = F.unfold(x, 
                         kernel_size=self.kernel_size, 
                         stride=self.stride)
        
        # Reshape patches to match piecewise layer input
        # Shape: (batch_size * out_height * out_width, in_channels * kernel_height * kernel_width)
        patches = patches.transpose(1, 2).reshape(-1, self.piecewise.num_inputs)
        
        # Apply piecewise functions
        # Shape: (batch_size * out_height * out_width, out_channels)
        output = self.piecewise(patches)
        
        # Reshape back to convolutional output format
        # Shape: (batch_size, out_channels, out_height, out_width)
        output = output.reshape(batch_size, out_height * out_width, self.out_channels)
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, self.out_channels, out_height, out_width)
        
        return output
    
    def move_smoothest(self, weighted:bool=True):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed for each AdaptivePiecewiseLinear layer in the MinGRU cell.
        
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            # Try moving the smoothest point in each layer
            success = self.piecewise.move_smoothest(weighted=weighted)
            
            return success


    def insert_points(self, x):
        """Insert points based on input x"""
        # Extract patches like in forward pass
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.transpose(1, 2).reshape(-1, self.piecewise.num_inputs)
        return self.piecewise.insert_points(patches)
    
    def insert_nearby_point(self, x_error):
        """Insert nearby points based on input x"""
        # x_error should already be in the right shape
        # (in_channels * kernel_height * kernel_width,)
        
        # For each output channel, add a new point
        for out_channel in range(self.out_channels):
            # Create a new positions tensor with space for one more point
            new_positions = torch.zeros(
                self.piecewise.num_inputs,
                self.piecewise.num_outputs,
                self.piecewise.num_points + 1,
                device=self.piecewise.positions.device
            )
            new_values = torch.zeros(
                self.piecewise.num_inputs,
                self.piecewise.num_outputs,
                self.piecewise.num_points + 1,
                device=self.piecewise.values.device
            )
            
            # Find insertion point
            with torch.no_grad():
                # For each input dimension, find the nearest left and right points
                for i in range(self.piecewise.num_inputs):
                    positions = self.piecewise.positions[i, out_channel]
                    
                    # Find points to the left and right of the target point
                    left_mask = positions <= x_error[i]
                    right_mask = positions > x_error[i]
                    
                    if not left_mask.any() or not right_mask.any():
                        continue
                    
                    # Get nearest left and right points
                    left_idx = torch.where(left_mask)[0][-1]
                    right_idx = torch.where(right_mask)[0][0]
                    
                    # Calculate midpoint
                    left_pos = positions[left_idx]
                    right_pos = positions[right_idx]
                    midpoint = (left_pos + right_pos) / 2
                    
                    # Check if midpoint is too close to existing points
                    min_distance = 1e-2
                    distances = torch.abs(positions - midpoint)
                    if torch.any(distances < min_distance):
                        continue
                    
                    # For each output channel
                    for j in range(self.piecewise.num_outputs):
                        # Copy existing points up to insertion point
                        new_positions[i, j, :left_idx+1] = self.piecewise.positions[i, j, :left_idx+1]
                        new_values[i, j, :left_idx+1] = self.piecewise.values[i, j, :left_idx+1]
                        
                        # Insert new point
                        new_positions[i, j, left_idx+1] = midpoint
                        
                        # Linearly interpolate value
                        left_val = self.piecewise.values[i, j, left_idx]
                        right_val = self.piecewise.values[i, j, right_idx]
                        t = (midpoint - left_pos) / (right_pos - left_pos)
                        new_val = left_val + t * (right_val - left_val)
                        new_values[i, j, left_idx+1] = new_val
                        
                        # Copy remaining points
                        new_positions[i, j, left_idx+2:] = self.piecewise.positions[i, j, left_idx+1:]
                        new_values[i, j, left_idx+2:] = self.piecewise.values[i, j, left_idx+1:]
            
            # Update the layer's parameters
            self.piecewise.positions = new_positions
            self.piecewise.values = nn.Parameter(new_values)
            self.piecewise.num_points += 1
        
        return True
    
    def largest_error(self, error, x):
        """Find input x value that corresponds to largest error"""
        # Extract patches like in forward pass
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.transpose(1, 2).reshape(-1, self.piecewise.num_inputs)
        
        # Reshape error to match patches
        error = error.reshape(-1, self.out_channels)
        
        # Get x_error and ensure it's the right shape and within range
        x_error = self.piecewise.largest_error(error, patches)
        x_error = x_error.squeeze()  # Remove any extra dimensions
        x_error = torch.clamp(x_error, self.piecewise.position_min, self.piecewise.position_max)
        return x_error

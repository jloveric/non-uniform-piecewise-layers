import torch
import torch.nn as nn
import torch.nn.functional as F
from .adaptive_piecewise_linear import AdaptivePiecewiseLinear
#import torch.jit

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
        
        # Pre-calculate some constants to avoid recomputing them
        self.kernel_size_product = self.kernel_size[0] * self.kernel_size[1]
        self.total_inputs = self.in_channels * self.kernel_size_product
        
        # Cache for forward pass dimensions
        self._last_input_shape = None
        self._last_output_dims = None
        
    def forward(self, x):
        """
        Forward pass of the convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape
        
        # Use cached dimensions if input shape hasn't changed
        if self._last_input_shape != (batch_size, in_channels, height, width):
            # Calculate output dimensions
            out_height = ((height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
            out_width = ((width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
            self._last_input_shape = (batch_size, in_channels, height, width)
            self._last_output_dims = (out_height, out_width)
        else:
            out_height, out_width = self._last_output_dims
        
        # Add padding if needed (only once)
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        
        # Extract patches using unfold - this is a bottleneck operation
        # Shape: (batch_size, in_channels * kernel_height * kernel_width, out_height * out_width)
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Optimize the reshape operations by doing them in fewer steps
        # Combine transpose and reshape into a single view operation where possible
        patches = patches.permute(0, 2, 1).contiguous().view(-1, self.total_inputs)
        
        # Apply piecewise functions
        output = self.piecewise(patches)
        
        # Optimize the reshape back operations
        # Combine multiple reshapes and transposes into fewer operations
        output = output.view(batch_size, out_height * out_width, self.out_channels)
        output = output.permute(0, 2, 1).contiguous().view(batch_size, self.out_channels, out_height, out_width)
        
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
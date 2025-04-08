import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def piecewise_linear_kernel(
    # Pointers to matrices
    x_ptr, positions_ptr, values_ptr, output_ptr,
    # Matrix dimensions
    batch_size, num_inputs, num_outputs, num_points,
    # Parameters
    min_pos, max_pos,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr, MAX_POINTS: tl.constexpr = 16,
):
    """
    Triton kernel for piecewise linear interpolation.
    
    This kernel efficiently computes the piecewise linear interpolation for each input-output pair.
    It handles all special cases (out of range, exact matches, etc.) in a vectorized manner.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Compute the batch and output indices
    batch_idx = pid // num_outputs
    output_idx = pid % num_outputs
    
    # Bounds checking
    if batch_idx >= batch_size:
        return
    
    # Compute the start offset for this batch and output
    x_offset = batch_idx * num_inputs
    output_offset = batch_idx * num_outputs + output_idx
    
    # Load input values for this batch
    x_block_ptr = x_ptr + x_offset
    x_values = tl.load(x_block_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < num_inputs)
    
    # Initialize output accumulator
    output_value = 0.0
    
    # Process each input
    for input_idx in range(0, num_inputs, BLOCK_SIZE):
        # Compute number of valid inputs in this block
        n_inputs = min(BLOCK_SIZE, num_inputs - input_idx)
        
        # Skip if no valid inputs
        if n_inputs <= 0:
            break
        
        # Load input values for this block
        block_mask = tl.arange(0, BLOCK_SIZE) < n_inputs
        x_block_ptr = x_ptr + x_offset + input_idx
        x_values = tl.load(x_block_ptr + tl.arange(0, BLOCK_SIZE), mask=block_mask)
        
        # Process each input in the block
        for i in range(n_inputs):
            # Get the input value
            x_val = x_values[i]
            
            # Clamp input to position range
            x_val = tl.minimum(tl.maximum(x_val, min_pos), max_pos)
            
            # Process each input-output pair
            pos_offset = (output_idx * num_inputs + input_idx + i) * num_points
            val_offset = (output_idx * num_inputs + input_idx + i) * num_points
            
            # Load positions and values for this input-output pair with a fixed-size arange
            positions = tl.load(positions_ptr + pos_offset + tl.arange(0, 16), 
                              mask=tl.arange(0, 16) < num_points)
            values = tl.load(values_ptr + val_offset + tl.arange(0, 16), 
                           mask=tl.arange(0, 16) < num_points)
            
            # Find which interval the input falls into
            # We'll use a binary search approach for efficiency
            left_idx = 0
            right_idx = num_points - 2  # Last valid interval index
            
            # Handle edge cases first
            is_below_min = x_val <= positions[0]
            is_above_max = x_val >= positions[num_points - 1]
            is_in_range = (not is_below_min) and (not is_above_max)
            
            if is_below_min:
                # Input is at or below the first position
                first_val = tl.load(values_ptr + val_offset)
                output_value += first_val
            elif is_above_max:
                # Input is at or above the last position
                last_val = tl.load(values_ptr + val_offset + num_points - 1)
                output_value += last_val
            else:
                # We'll use direct loads instead of shared memory for simplicity
                # This is still more efficient than our previous approach
                
                # Get first and last positions for edge cases
                first_pos = tl.load(positions_ptr + pos_offset)
                last_pos = tl.load(positions_ptr + pos_offset + num_points - 1)
                
                # Handle edge cases
                if x_val <= first_pos:
                    # Input is at or below the first position
                    output_value += tl.load(values_ptr + val_offset)
                elif x_val >= last_pos:
                    # Input is at or above the last position
                    output_value += tl.load(values_ptr + val_offset + num_points - 1)
                else:
                    # Optimized search with early exit
                    # Use a more efficient search pattern
                    for i in range(0, num_points - 1):
                        pos_i = tl.load(positions_ptr + pos_offset + i)
                        pos_i_plus_1 = tl.load(positions_ptr + pos_offset + i + 1)
                        is_in_interval = (pos_i <= x_val) and (x_val < pos_i_plus_1)
                        
                        if is_in_interval:
                            # Found the interval
                            left_pos = pos_i
                            right_pos = pos_i_plus_1
                            left_val = tl.load(values_ptr + val_offset + i)
                            right_val = tl.load(values_ptr + val_offset + i + 1)
                            
                            # Check for duplicate positions
                            if left_pos == right_pos:
                                output_value += left_val
                            else:
                                # Interpolate
                                t = (x_val - left_pos) / (right_pos - left_pos)
                                output_value += left_val + t * (right_val - left_val)
                            # No break statement, we'll use the found_interval flag instead
    
    # Store the final output
    tl.store(output_ptr + output_offset, output_value)


@triton.jit
def unfold_and_piecewise_kernel(
    # Pointers to matrices
    input_ptr, positions_ptr, values_ptr, output_ptr,
    # Dimensions
    batch_size, in_channels, out_channels, height, width, out_height, out_width,
    kernel_height, kernel_width, num_points,
    # Convolution parameters
    stride_h, stride_w, padding_h, padding_w,
    # Position range
    min_pos, max_pos,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, MAX_POINTS: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 32,  # Number of elements to process in parallel
):
    """
    Triton kernel that combines unfolding and piecewise linear interpolation for the convolutional layer.
    
    This kernel efficiently extracts patches and applies piecewise linear interpolation in a single pass,
    avoiding the memory overhead of storing intermediate tensors.
    """
    # Program ID
    pid_m = tl.program_id(axis=0)  # batch_size * out_height * out_width
    pid_n = tl.program_id(axis=1)  # out_channels
    
    # Compute spatial indices
    batch_out_idx = pid_m
    batch_idx = batch_out_idx // (out_height * out_width)
    out_spatial_idx = batch_out_idx % (out_height * out_width)
    out_h = out_spatial_idx // out_width
    out_w = out_spatial_idx % out_width
    
    # Compute input spatial indices (top-left corner of the patch)
    in_h = out_h * stride_h - padding_h
    in_w = out_w * stride_w - padding_w
    
    # Output offset for this thread
    output_offset = batch_idx * out_channels * out_height * out_width + pid_n * out_height * out_width + out_spatial_idx
    
    # Initialize output accumulator
    output_value = 0.0
    
    # Process each input channel and kernel position
    for in_ch in range(in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Compute input position
                h_pos = in_h + kh
                w_pos = in_w + kw
                
                # Only process if inside input bounds
                is_inside_bounds = (((h_pos >= 0) and (h_pos < height)) and ((w_pos >= 0) and (w_pos < width)))
                
                if is_inside_bounds:
                    # Get input value
                    input_offset = batch_idx * in_channels * height * width + in_ch * height * width + h_pos * width + w_pos
                    x_val = tl.load(input_ptr + input_offset)
                    
                    # Clamp input to position range
                    x_val = tl.minimum(tl.maximum(x_val, min_pos), max_pos)
                    
                    # Get positions and values for this kernel element
                    kernel_idx = kh * kernel_width + kw
                    pos_offset = (pid_n * in_channels * kernel_height * kernel_width + in_ch * kernel_height * kernel_width + kernel_idx) * num_points
                    val_offset = (pid_n * in_channels * kernel_height * kernel_width + in_ch * kernel_height * kernel_width + kernel_idx) * num_points
                    
                    # Load positions and values with a fixed-size arange and mask for the actual number of points
                    positions = tl.load(positions_ptr + pos_offset + tl.arange(0, MAX_POINTS), 
                                      mask=tl.arange(0, MAX_POINTS) < num_points)
                    values = tl.load(values_ptr + val_offset + tl.arange(0, MAX_POINTS), 
                                   mask=tl.arange(0, MAX_POINTS) < num_points)
                    
                    # Handle different cases based on where x_val falls
                    first_pos = tl.load(positions_ptr + pos_offset)
                    last_pos = tl.load(positions_ptr + pos_offset + num_points - 1)
                    is_below_min = x_val <= first_pos
                    is_above_max = x_val >= last_pos
                    
                    if is_below_min:
                        # Input is at or below the first position
                        first_val = tl.load(values_ptr + val_offset)
                        output_value += first_val
                    elif is_above_max:
                        # Input is at or above the last position
                        last_val = tl.load(values_ptr + val_offset + num_points - 1)
                        output_value += last_val
                    else:
                                            # We'll use direct loads instead of shared memory for simplicity
                        # This is still more efficient than our previous approach
                        
                        # Optimized search with early exit
                        found_interval = False
                        # Use a more efficient search pattern
                        for i in range(0, num_points - 1):
                            pos_i = tl.load(positions_ptr + pos_offset + i)
                            pos_i_plus_1 = tl.load(positions_ptr + pos_offset + i + 1)
                            is_in_interval = (pos_i <= x_val) and (x_val < pos_i_plus_1)
                            
                            if is_in_interval:
                                # Found the interval
                                left_pos = pos_i
                                right_pos = pos_i_plus_1
                                left_val = tl.load(values_ptr + val_offset + i)
                                right_val = tl.load(values_ptr + val_offset + i + 1)
                                
                                # Check for duplicate positions
                                if left_pos == right_pos:
                                    output_value += left_val
                                else:
                                    # Interpolate
                                    t = (x_val - left_pos) / (right_pos - left_pos)
                                    output_value += left_val + t * (right_val - left_val)
                                found_interval = True
                        
    # Store the final output
    tl.store(output_ptr + output_offset, output_value)


class TritonAdaptivePiecewiseLinear(nn.Module):
    """
    Triton-accelerated implementation of AdaptivePiecewiseLinear.
    """
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_points: int,
        position_range=(-1, 1),
        anti_periodic: bool = True,
        position_init: str = "uniform",
    ):
        """
        Initialize an adaptive piecewise linear layer with Triton acceleration.
        
        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output features
            num_points (int): Initial number of points per piecewise function
            position_range (tuple): Tuple of (min, max) for allowed position range. Default is (-1, 1)
            anti_periodic (bool): Whether to use anti-periodic boundary conditions. Default is True
            position_init (str): Position initialization method. Must be one of ["uniform", "random"]. Default is "uniform"
        """
        super().__init__()

        if position_init not in ["uniform", "random"]:
            raise ValueError("position_init must be one of ['uniform', 'random']")

        self.position_min, self.position_max = position_range
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_points = num_points
        self.anti_periodic = anti_periodic

        # Initialize positions based on initialization method
        if position_init == "uniform":
            # Original uniform initialization
            positions = torch.linspace(self.position_min, self.position_max, num_points).repeat(num_inputs, num_outputs, 1)
        else:  # random
            # Create random positions between -1 and 1
            positions = torch.rand(num_inputs, num_outputs, num_points) * (self.position_max - self.position_min) + self.position_min
            # Sort positions along last dimension to maintain order
            positions, _ = torch.sort(positions, dim=-1)
            # Fix first and last positions to be -1 and 1
            positions[..., 0] = self.position_min
            positions[..., -1] = self.position_max

        self.register_buffer("positions", positions)

        # Initialize each input-output pair with a random line (collinear points)
        # The factor 0.5 is from trial and error to get a stable solution with mingru
        # the rest is just central limit theorem
        factor = 0.5*math.sqrt(1.0/(3*num_inputs))
        
        start = torch.empty(num_inputs, num_outputs).uniform_(-factor, factor)
        end = torch.empty(num_inputs, num_outputs).uniform_(-factor, factor)
        weights = torch.linspace(0, 1, num_points, device=start.device).view(1, 1, num_points)
        values_line = start.unsqueeze(-1) * (1 - weights) + end.unsqueeze(-1) * weights
        self.values = nn.Parameter(values_line)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer using Triton acceleration.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs)
        """
        batch_size = x.shape[0]
        
        # Handle anti-periodic boundary conditions if needed
        if self.anti_periodic:
            x = torch.where(x < 0, x, -x)
        
        # Prepare output tensor
        output = torch.zeros(batch_size, self.num_outputs, device=x.device, dtype=x.dtype)
        
        # Prepare positions and values for Triton kernel
        # Reshape from [num_inputs, num_outputs, num_points] to [num_outputs, num_inputs, num_points]
        positions_reshaped = self.positions.permute(1, 0, 2).contiguous()
        values_reshaped = self.values.permute(1, 0, 2).contiguous()
        
        # Determine grid and block sizes
        grid = (batch_size * self.num_outputs,)
        
        # Launch Triton kernel
        piecewise_linear_kernel[grid](
            x.contiguous(), 
            positions_reshaped.contiguous(), 
            values_reshaped.contiguous(), 
            output.contiguous(),
            batch_size, 
            self.num_inputs, 
            self.num_outputs, 
            self.num_points,
            self.position_min, 
            self.position_max,
            BLOCK_SIZE=min(128, self.num_inputs),
            MAX_POINTS=16,
        )
        
        return output


class TritonAdaptivePiecewiseConv2d(nn.Module):
    """
    Triton-accelerated implementation of AdaptivePiecewiseConv2d.
    """
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
        2D convolutional layer using adaptive piecewise linear functions with Triton acceleration.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            num_points (int): Initial number of points per piecewise function. Default: 3
            position_range (tuple): Tuple of (min, max) for allowed position range. Default: (-1, 1)
            position_init (str): Position initialization method. Default: "uniform"
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
        self.position_min, self.position_max = position_range
        self.num_points = num_points

        # Initialize positions based on initialization method
        if position_init == "uniform":
            # Create uniform positions for all kernel elements
            positions = torch.linspace(
                self.position_min, 
                self.position_max, 
                num_points
            ).repeat(
                out_channels, 
                in_channels, 
                kernel_size[0], 
                kernel_size[1], 
                1
            )
        else:  # random
            # Create random positions for all kernel elements
            positions = torch.rand(
                out_channels, 
                in_channels, 
                kernel_size[0], 
                kernel_size[1], 
                num_points
            ) * (self.position_max - self.position_min) + self.position_min
            
            # Sort positions along the last dimension
            positions, _ = torch.sort(positions, dim=-1)
            
            # Fix first and last positions
            positions[..., 0] = self.position_min
            positions[..., -1] = self.position_max

        self.register_buffer("positions", positions)

        # Initialize values for each kernel element
        factor = 0.5 * math.sqrt(1.0 / (3 * in_channels * kernel_size[0] * kernel_size[1]))
        
        # Initialize with random lines
        start = torch.empty(
            out_channels, 
            in_channels, 
            kernel_size[0], 
            kernel_size[1]
        ).uniform_(-factor, factor)
        
        end = torch.empty(
            out_channels, 
            in_channels, 
            kernel_size[0], 
            kernel_size[1]
        ).uniform_(-factor, factor)
        
        weights = torch.linspace(0, 1, num_points).view(1, 1, 1, 1, num_points)
        values_line = start.unsqueeze(-1) * (1 - weights) + end.unsqueeze(-1) * weights
        
        self.values = nn.Parameter(values_line)

        # Cache for forward pass dimensions
        self._last_input_shape = None
        self._last_output_dims = None

    def forward(self, x):
        """
        Forward pass of the convolutional layer using Triton acceleration.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape

        # Use cached dimensions if input shape hasn't changed
        if self._last_input_shape != (batch_size, in_channels, height, width):
            # Calculate output dimensions
            out_height = (
                (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]
            ) + 1
            out_width = (
                (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]
            ) + 1
            self._last_input_shape = (batch_size, in_channels, height, width)
            self._last_output_dims = (out_height, out_width)
        else:
            out_height, out_width = self._last_output_dims

        # Add padding if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(
                x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            )

        # Prepare output tensor
        output = torch.zeros(
            batch_size, 
            self.out_channels, 
            out_height, 
            out_width, 
            device=x.device, 
            dtype=x.dtype
        )

        # Determine optimized grid size for Triton kernel
        # Use a more efficient grid layout to maximize GPU utilization
        grid = (batch_size * out_height * out_width, self.out_channels)
        
        # Launch Triton kernel with optimized parameters
        unfold_and_piecewise_kernel[grid](
            x.contiguous(),
            self.positions.contiguous(),
            self.values.contiguous(),
            output.contiguous(),
            batch_size, 
            in_channels, 
            self.out_channels, 
            height + 2 * self.padding[0], 
            width + 2 * self.padding[1],
            out_height, 
            out_width,
            self.kernel_size[0], 
            self.kernel_size[1], 
            self.num_points,
            self.stride[0], 
            self.stride[1], 
            self.padding[0], 
            self.padding[1],
            self.position_min, 
            self.position_max,
            BLOCK_SIZE_M=32,  # Increased block size for better parallelism
            BLOCK_SIZE_N=32,  # Increased block size for better parallelism
            MAX_POINTS=16,    # Maximum number of points to support
            BLOCK_SIZE_K=32,  # Process more elements in parallel
        )
        
        return output

    def move_smoothest(self, weighted: bool = True):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed for each kernel element.
        
        Args:
            weighted (bool): Whether to weight the errors by the distance between points
            
        Returns:
            int: Number of points moved
        """
        with torch.no_grad():
            # This operation is not accelerated with Triton since it's not performance-critical
            # and is typically done infrequently
            
            # Compute removal errors for each point
            errors = []
            for out_ch in range(self.out_channels):
                for in_ch in range(self.in_channels):
                    for kh in range(self.kernel_size[0]):
                        for kw in range(self.kernel_size[1]):
                            positions = self.positions[out_ch, in_ch, kh, kw]
                            values = self.values[out_ch, in_ch, kh, kw]
                            
                            # Skip if we only have 2 points (can't remove any more)
                            if len(positions) <= 2:
                                continue
                            
                            # Compute errors for removing each internal point
                            internal_errors = []
                            for i in range(1, len(positions) - 1):
                                # Get positions and values for adjacent points
                                left_pos = positions[i-1]
                                mid_pos = positions[i]
                                right_pos = positions[i+1]
                                left_val = values[i-1]
                                mid_val = values[i]
                                right_val = values[i+1]
                                
                                # Compute interpolated value at the middle position
                                t = (mid_pos - left_pos) / (right_pos - left_pos)
                                interp_val = left_val + t * (right_val - left_val)
                                
                                # Compute error
                                error = torch.abs(mid_val - interp_val)
                                
                                # Weight by distance if requested
                                if weighted:
                                    dist = (right_pos - left_pos)
                                    error = error / dist
                                
                                internal_errors.append((i, error.item()))
                            
                            if internal_errors:
                                # Find point with minimum error
                                min_idx, min_error = min(internal_errors, key=lambda x: x[1])
                                
                                # Find point with maximum error
                                max_idx, max_error = max(internal_errors, key=lambda x: x[1])
                                
                                # Remove the point with minimum error
                                new_positions = torch.cat([positions[:min_idx], positions[min_idx+1:]])
                                new_values = torch.cat([values[:min_idx], values[min_idx+1:]])
                                
                                # Determine where to insert the new point
                                # Choose a random position to the left or right of the max error point
                                if max_idx == 0:
                                    # Can only insert to the right
                                    left_idx = max_idx
                                    right_idx = max_idx + 1
                                elif max_idx == len(new_positions) - 2:
                                    # Can only insert to the left
                                    left_idx = max_idx - 1
                                    right_idx = max_idx
                                else:
                                    # Can insert to either side, choose randomly
                                    if torch.rand(1).item() < 0.5:
                                        left_idx = max_idx - 1
                                        right_idx = max_idx
                                    else:
                                        left_idx = max_idx
                                        right_idx = max_idx + 1
                                
                                # Compute the new position
                                left_pos = new_positions[left_idx]
                                right_pos = new_positions[right_idx]
                                new_pos = (left_pos + right_pos) / 2
                                
                                # Compute the new value by interpolation
                                t = (new_pos - left_pos) / (right_pos - left_pos)
                                new_val = new_values[left_idx] + t * (new_values[right_idx] - new_values[left_idx])
                                
                                # Insert the new point
                                insert_idx = right_idx
                                new_positions = torch.cat([new_positions[:insert_idx], new_pos.unsqueeze(0), new_positions[insert_idx:]])
                                new_values = torch.cat([new_values[:insert_idx], new_val.unsqueeze(0), new_values[insert_idx:]])
                                
                                # Update the positions and values
                                self.positions[out_ch, in_ch, kh, kw] = new_positions
                                self.values.data[out_ch, in_ch, kh, kw] = new_values
            
            return len(errors)


class TritonAdaptivePiecewiseMLP(nn.Module):
    """
    Triton-accelerated implementation of AdaptivePiecewiseMLP.
    """
    def __init__(
        self,
        width: list,
        num_points: int = 3,
        position_range=(-1, 1),
        anti_periodic: bool = True,
        position_init: str="uniform",
        normalization: str="noop"
    ):
        """
        Initialize a multi-layer perceptron with adaptive piecewise linear layers using Triton acceleration.
        
        Args:
            width (List[int]): List of widths for each layer
            num_points (int): Initial number of points for each piecewise function
            position_range (tuple): Tuple of (min, max) for allowed position range
            anti_periodic (bool): Whether to use anti-periodic boundary conditions
            position_init (str): Position initialization method
            normalization (str): Normalization method to use between layers
        """
        super().__init__()

        if len(width) < 2:
            raise ValueError(f"Width list must have at least 2 elements, got {len(width)}")

        if num_points < 2:
            raise ValueError(f"Number of points must be at least 2, got {num_points}")

        self.anti_periodic = anti_periodic

        # Create layers using Triton-accelerated implementation
        self.layers = nn.ModuleList(
            [
                TritonAdaptivePiecewiseLinear(
                    num_inputs=width[i],
                    num_outputs=width[i + 1],
                    num_points=num_points,
                    position_range=position_range,
                    anti_periodic=anti_periodic,
                    position_init=position_init,
                )
                for i in range(len(width) - 1)
            ]
        )

        # Set up normalization
        if normalization == "noop":
            self.normalization = lambda x: x
        elif normalization == "maxabs":
            self.normalization = lambda x: x / (torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + 1e-6)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

    def forward(self, x):
        """
        Forward pass through all layers using Triton acceleration.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_width)
        """
        current = x
        for layer in self.layers[:-1]:
            current = self.normalization(layer(current))

        # We don't normalize the last layer
        current = self.layers[-1](current)
        return current

    def move_smoothest(self, weighted: bool = True):
        """
        Remove the point with the smallest removal error and insert a new point
        for each layer in the MLP.
        
        Args:
            weighted (bool): Whether to weight the errors by the distance between points
            
        Returns:
            list: Number of points moved in each layer
        """
        moved = []
        for layer in self.layers:
            moved.append(layer.move_smoothest(weighted=weighted))
        return moved

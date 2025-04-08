import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fast_piecewise_kernel(
    # Pointers to matrices
    patches_ptr, positions_ptr, values_ptr, output_ptr,
    # Dimensions
    batch_size, num_patches, in_channels, out_channels, kernel_h, kernel_w, num_points,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Efficient Triton kernel for piecewise linear interpolation.
    This kernel processes patches in parallel using a simplified algorithm.
    """
    # Get program ID and compute indices
    pid = tl.program_id(0)
    
    # Each thread processes one patch and one output channel
    batch_idx = pid // (out_channels * num_patches)
    remainder = pid % (out_channels * num_patches)
    out_ch = remainder // num_patches
    patch_idx = remainder % num_patches
    
    # Skip if out of bounds
    if batch_idx >= batch_size:
        return
    if out_ch >= out_channels:
        return
    if patch_idx >= num_patches:
        return
    
    # Initialize output value for this patch and output channel
    output_value = 0.0
    
    # Calculate the total number of inputs per patch
    total_inputs = in_channels * kernel_h * kernel_w
    
    # Calculate offsets for this batch and patch
    batch_offset = batch_idx * num_patches * total_inputs
    patch_offset = patch_idx * total_inputs
    base_patch_offset = batch_offset + patch_offset
    
    # Calculate base offset for positions and values for this output channel
    base_pos_offset = out_ch * total_inputs * num_points
    
    # Process each input in a single loop to reduce loop overhead
    for input_idx in range(total_inputs):
        # Get patch value with coalesced memory access
        x_val = tl.load(patches_ptr + base_patch_offset + input_idx)
        
        # Calculate position and value offsets for this input
        pos_offset = base_pos_offset + input_idx * num_points
        
        # Get first and last positions for edge cases
        first_pos = tl.load(positions_ptr + pos_offset)
        last_pos = tl.load(positions_ptr + pos_offset + num_points - 1)
        
        # Handle edge cases
        if x_val <= first_pos:
            # Input is at or below the first position
            output_value += tl.load(values_ptr + pos_offset)
        elif x_val >= last_pos:
            # Input is at or above the last position
            output_value += tl.load(values_ptr + pos_offset + num_points - 1)
        else:
            # Find the interval using linear search without break statements
            # We'll use a mask-based approach which is more Triton-friendly
            interval_idx = 0
            
            # Loop through all possible intervals
            for j in range(num_points - 1):
                pos_i = tl.load(positions_ptr + pos_offset + j)
                pos_i_plus_1 = tl.load(positions_ptr + pos_offset + j + 1)
                
                # Check if x_val is in this interval
                in_interval = (pos_i <= x_val) & (x_val < pos_i_plus_1)
                
                # Update interval_idx if we're in this interval
                # This will overwrite previous values, but the last match is what we want
                interval_idx = tl.where(in_interval, j, interval_idx)
            
            # Get values for the found interval
            val_i = tl.load(values_ptr + pos_offset + interval_idx)
            val_i_plus_1 = tl.load(values_ptr + pos_offset + interval_idx + 1)
            
            # Get positions for the found interval
            pos_i = tl.load(positions_ptr + pos_offset + interval_idx)
            pos_i_plus_1 = tl.load(positions_ptr + pos_offset + interval_idx + 1)
            
            # Check if we have duplicate positions
            duplicate_positions = pos_i == pos_i_plus_1
            
            # Calculate interpolation factor
            t = (x_val - pos_i) / (pos_i_plus_1 - pos_i + 1e-6)  # Add small epsilon for stability
            t = tl.maximum(0.0, tl.minimum(1.0, t))  # Clamp to [0, 1]
            
            # Compute interpolated value
            interpolated = val_i + t * (val_i_plus_1 - val_i)
            
            # Use val_i if positions are duplicate, otherwise use interpolated value
            result = tl.where(duplicate_positions, val_i, interpolated)
            
            # Add to output
            output_value += result
    
    # Store output with coalesced memory access
    output_offset = batch_idx * out_channels * num_patches + out_ch * num_patches + patch_idx
    tl.store(output_ptr + output_offset, output_value)


class FastTritonAdaptivePiecewiseConv2d(nn.Module):
    """
    Fast Triton implementation of AdaptivePiecewiseConv2d.
    This implementation uses PyTorch for unfold and Triton for the piecewise linear interpolation.
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
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_points = num_points
        self.position_min, self.position_max = position_range
        
        # Initialize positions (shared across all channels)
        positions = torch.linspace(
            self.position_min, self.position_max, num_points
        ).view(1, 1, 1, 1, num_points)
        
        # Expand positions for each channel, input, and kernel element
        self.positions = nn.Parameter(
            positions.expand(
                out_channels, 
                in_channels, 
                kernel_size[0], 
                kernel_size[1], 
                num_points
            )
        )
        
        # Initialize values with small random values
        # Start with linear function from start to end
        start = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * 0.01
        end = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * 0.01
        
        weights = torch.linspace(0, 1, num_points).view(1, 1, 1, 1, num_points)
        values_line = start.unsqueeze(-1) * (1 - weights) + end.unsqueeze(-1) * weights
        
        self.values = nn.Parameter(values_line)
        
        # Cache for forward pass dimensions
        self._last_input_shape = None
        self._last_output_dims = None
    
    def forward(self, x):
        """
        Forward pass using a hybrid approach: PyTorch for unfold, Triton for piecewise linear.
        
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
        
        # Extract patches using PyTorch's optimized unfold operation
        # Shape: (batch_size, in_channels * kernel_height * kernel_width, out_height * out_width)
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Number of patches
        num_patches = out_height * out_width
        total_inputs = in_channels * self.kernel_size[0] * self.kernel_size[1]
        
        # Prepare output tensor
        output = torch.zeros(
            batch_size, 
            self.out_channels, 
            num_patches, 
            device=x.device, 
            dtype=x.dtype
        )
        
        # We need to make sure positions are sorted for binary search
        # Let's reshape the positions and values to match the expected layout
        positions = self.positions.detach().clone()
        values = self.values.detach().clone()
        
        # Reshape to the expected format for the kernel
        positions = positions.view(self.out_channels, total_inputs, self.num_points)
        values = values.view(self.out_channels, total_inputs, self.num_points)
        
        # Ensure positions are sorted along the last dimension
        # This is important for the binary search algorithm in the kernel
        for b in range(self.out_channels):
            for c in range(total_inputs):
                # Get indices that would sort the positions
                idx = torch.argsort(positions[b, c])
                # Sort positions and values accordingly
                positions[b, c] = torch.gather(positions[b, c], 0, idx)
                values[b, c] = torch.gather(values[b, c], 0, idx)
        
        # Determine grid size for Triton kernel
        grid = (batch_size * self.out_channels * num_patches,)
        
        # Launch Triton kernel
        fast_piecewise_kernel[grid](
            patches.contiguous(),
            positions.contiguous(),  # Already in the right shape
            values.contiguous(),     # Already in the right shape
            output.contiguous(),
            batch_size, 
            num_patches,
            in_channels, 
            self.out_channels, 
            self.kernel_size[0], 
            self.kernel_size[1],
            self.num_points,
            BLOCK_SIZE=128,
        )
        
        # Reshape output to expected format
        output = output.view(batch_size, self.out_channels, out_height, out_width)
        
        return output
    
    def move_smoothest(self, weighted: bool = True):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed for each AdaptivePiecewiseLinear layer in the MinGRU cell.
        
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            # We need at least 4 points to be able to remove one and still have enough for interpolation
            if self.num_points < 4:
                return False
            
            # Get positions and reshape for easier processing
            positions = self.positions
            
            # Compute removal errors for each interior point
            # For each interior point, calculate the error if we were to remove it
            # and interpolate between its neighbors
            interior_indices = torch.arange(1, self.num_points-1, device=positions.device)
            
            # For each interior point, we need its position and its left/right neighbors
            errors = []
            
            for idx in interior_indices:
                # Get the left and right neighbors
                left_pos = positions[..., idx-1]
                mid_pos = positions[..., idx]
                right_pos = positions[..., idx+1]
                
                # Calculate interpolation parameter t
                t = (mid_pos - left_pos) / (right_pos - left_pos)
                
                # Calculate the error as the distance from the actual position
                # to where it would be if we interpolated between neighbors
                error = torch.abs(mid_pos - (left_pos + t * (right_pos - left_pos)))
                
                # If weighted, scale by the distance between neighbors
                if weighted:
                    error = error * torch.abs(right_pos - left_pos)
                    
                errors.append(error.mean().item())
            
            # Find the point with the smallest error
            min_error_idx = torch.argmin(torch.tensor(errors)).item()
            min_error_idx = min_error_idx + 1  # Adjust for interior indices
            
            # Find the point with the largest error
            max_error_idx = torch.argmax(torch.tensor(errors)).item()
            max_error_idx = max_error_idx + 1  # Adjust for interior indices
            
            # Remove the point with the smallest error
            # We'll do this by creating a new set of positions and values without that point
            new_positions = torch.zeros_like(positions[..., :self.num_points-1])
            new_values = torch.zeros_like(self.values[..., :self.num_points-1])
            
            # Copy all points except the one to remove
            idx = 0
            for i in range(self.num_points):
                if i != min_error_idx:
                    new_positions[..., idx] = positions[..., i]
                    new_values[..., idx] = self.values[..., i]
                    idx += 1
            
            # Now insert a new point near the point with the largest error
            # We'll randomly choose left or right of the max error point
            if torch.rand(1).item() < 0.5:
                # Insert to the left
                insert_idx = max_error_idx - 1
                left_pos = positions[..., insert_idx]
                right_pos = positions[..., max_error_idx]
            else:
                # Insert to the right
                insert_idx = max_error_idx
                left_pos = positions[..., max_error_idx]
                right_pos = positions[..., max_error_idx+1]
            
            # Generate a random position between left and right
            t = torch.rand_like(left_pos) * 0.8 + 0.1  # Between 0.1 and 0.9
            new_pos = left_pos + t * (right_pos - left_pos)
            
            # Generate a random value between left and right values
            left_val = self.values[..., insert_idx]
            right_val = self.values[..., insert_idx+1]
            new_val = left_val + t * (right_val - left_val)
            
            # Create the final positions and values with the new point inserted
            final_positions = torch.zeros_like(positions)
            final_values = torch.zeros_like(self.values)
            
            # Copy all points up to the insertion point
            final_positions[..., :insert_idx+1] = new_positions[..., :insert_idx+1]
            final_values[..., :insert_idx+1] = new_values[..., :insert_idx+1]
            
            # Insert the new point
            final_positions[..., insert_idx+1] = new_pos
            final_values[..., insert_idx+1] = new_val
            
            # Copy the remaining points
            final_positions[..., insert_idx+2:] = new_positions[..., insert_idx+1:]
            final_values[..., insert_idx+2:] = new_values[..., insert_idx+1:]
            
            # Update the parameters
            self.positions.data = final_positions
            self.values.data = final_values
            
            return True

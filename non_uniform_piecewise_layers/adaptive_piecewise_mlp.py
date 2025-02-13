import torch
import torch.nn as nn
from .adaptive_piecewise_linear import AdaptivePiecewiseLinear
from typing import List

class AdaptivePiecewiseMLP(nn.Module):
    def __init__(self, width: list, num_points: int = 3, position_range=(-1, 1), anti_periodic:bool=True):
        """
        Initialize a multi-layer perceptron with adaptive piecewise linear layers.
        Each layer is an AdaptivePiecewiseLinear layer.
        
        Args:
            width (List[int]): List of widths for each layer. Length should be num_layers + 1
                         where width[i] is the input size to layer i and width[i+1] is the output size.
                         For example, width=[2,4,3,1] creates a 3-layer network with:
                         - Layer 1: 2 inputs, 4 outputs
                         - Layer 2: 4 inputs, 3 outputs
                         - Layer 3: 3 inputs, 1 output
            num_points (int): Initial number of points for each piecewise function. Default is 3.
            position_range (tuple): Tuple of (min, max) for allowed position range. Default is (-1, 1)
        """
        super().__init__()
        
        if len(width) < 2:
            raise ValueError(f"Width list must have at least 2 elements, got {len(width)}")
        
        if num_points < 2:
            raise ValueError(f"Number of points must be at least 2, got {num_points}")
        
        self.anti_periodic=anti_periodic

        # Create layers
        self.layers = nn.ModuleList([
            AdaptivePiecewiseLinear(
                num_inputs=width[i],
                num_outputs=width[i+1],
                num_points=num_points,
                position_range=position_range,
                anti_periodic=anti_periodic
            ) for i in range(len(width)-1)
        ])
    
    def forward(self, x):
        """
        Forward pass through all layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_width)
        """
        current = x
        for layer in self.layers:
            current = layer(current)
        return current
    
    def largest_error(self, error, x):
        """
        Find the input x value that corresponds to the largest error in the output.
        
        Args:
            error (torch.Tensor): Error tensor of shape (batch_size, output_width)
            x (torch.Tensor): Input tensor of shape (batch_size, input_width)
            
        Returns:
            torch.Tensor: x value that corresponds to the largest error, or None if no valid point found
        """
        with torch.no_grad():
            # Sort errors in descending order
            sorted_errors, indices = torch.sort(error.abs().view(-1), descending=True)
            
            # Convert to batch indices
            batch_indices = indices // error.size(1)
            
            # Get corresponding x values
            candidate_x = x[batch_indices]
            
            # Check each candidate until we find one that's far enough from existing points
            min_distance = 1e-6
            for i in range(len(candidate_x)):
                x_val = candidate_x[i:i+1]  # Keep batch dimension
                
                # Check distance to all existing points in the first layer
                # We only need to check the first layer since that's where the input goes
                too_close = False
                for j in range(x.size(1)):  # Check each input dimension
                    positions = self.layers[0].positions[j, 0]  # Use first output dimension as reference
                    distances = torch.abs(positions - x_val[0, j])
                    if torch.any(distances < min_distance):
                        too_close = True
                        break
                
                if not too_close:
                    return x_val
            
            # If we get here, no valid point was found
            return None
    
    def insert_points(self, x):
        """
        Insert points into all layers, using the output of each layer as input to the next.
        
        Args:
            x (torch.Tensor): Points to insert with shape (batch_size, input_width)
            
        Returns:
            bool: True if points were inserted in any layer
        """
        with torch.no_grad():
            # Forward pass to get intermediate values
            intermediate_x = [x]
            current_x = x
            for layer in self.layers:
                current_x = layer(current_x)
                intermediate_x.append(current_x)
            
            # Try inserting points in each layer
            success = True
            for i, layer in enumerate(self.layers):
                success_ = layer.insert_points(intermediate_x[i])
        
        return success
    
    def insert_nearby_point(self, x):
        """
        Insert nearby points in all layers, using the output of each layer as input to the next.
        
        Args:
            x (torch.Tensor): Reference point with shape (batch_size, input_width)
            
        Returns:
            bool: True if points were inserted in any layer
        """
        with torch.no_grad():
            # Forward pass to get intermediate values
            intermediate_x = [x]
            current_x = x
            for layer in self.layers:
                current_x = layer(current_x)
                intermediate_x.append(current_x)
            
            # Try inserting nearby points in each layer
            success = True
            for i, layer in enumerate(self.layers):
                success_ = layer.insert_nearby_point(intermediate_x[i])
        
        return success

    def remove_add(self, x):
        """
        Remove the smoothest point and add a new point at the specified location
        for each layer in the MLP.

        Args:
            x (torch.Tensor): Reference point with shape (batch_size, input_width)
                specifying where to add the new point.

        Returns:
            bool: True if points were successfully removed and added in all layers,
                 False otherwise.
        """
        with torch.no_grad():
            # Forward pass to get intermediate values
            intermediate_x = [x]
            current_x = x
            for layer in self.layers:
                current_x = layer(current_x)
                intermediate_x.append(current_x)
            
            # Try removing and adding points in each layer
            success = True
            for i, layer in enumerate(self.layers):
                success_ = layer.remove_add(intermediate_x[i])
                if not success_:
                    success = False
        
        return success

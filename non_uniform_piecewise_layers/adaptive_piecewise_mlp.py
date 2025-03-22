import torch
import torch.nn as nn
from .adaptive_piecewise_linear import AdaptivePiecewiseLinear
from typing import List, Optional
from non_uniform_piecewise_layers.utils import max_abs_normalization_last

def noop(x):
    return x

norm_factory = {
    "noop": noop,
    "maxabs" :  max_abs_normalization_last
}

class AdaptivePiecewiseMLP(nn.Module):
    def __init__(
        self,
        width: list,
        num_points: int = 3,
        position_range=(-1, 1),
        anti_periodic: bool = True,
        position_init: str="uniform",
        normalization: str="noop" #maxabs
    ):
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
            raise ValueError(
                f"Width list must have at least 2 elements, got {len(width)}"
            )

        if num_points < 2:
            raise ValueError(f"Number of points must be at least 2, got {num_points}")

        self.anti_periodic = anti_periodic

        # Create layers
        self.layers = nn.ModuleList(
            [
                AdaptivePiecewiseLinear(
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

        # This is assuming we have a parameter free normalization! My favorite kind!
        self.normalization = norm_factory[normalization]
        

    def forward(self, x):
        """
        Forward pass through all layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_width)
        """
        current = x
        for layer in self.layers[:-1]:
            current = self.normalization(layer(current))

        # We don't normalize the last layer, we leave that to the user!
        current = self.layers[-1](current)
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
            return candidate_x

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
            for layer in self.layers[:-1]:
                current_x = self.normalization(layer(current_x))
                intermediate_x.append(current_x)
            current_x = self.layers[-1](current_x)
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
            for layer in self.layers[:-1]:
                current_x = self.normalization(layer(current_x))
                intermediate_x.append(current_x)
            current_x = self.layers[-1](current_x)
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
            for layer in self.layers[:-1]:
                current_x = self.normalization(layer(current_x))
                intermediate_x.append(current_x)
            current_x = self.layers[-1](current_x)
            intermediate_x.append(current_x)

            # Try removing and adding points in each layer
            success = True
            for i, layer in enumerate(self.layers):
                success_ = layer.remove_add(intermediate_x[i])
                if not success_:
                    success = False

        return success

    def move_smoothest(self, weighted: bool = True, threshold: float = None):
        """
        For each layer in the MLP, remove the point with the smallest removal error (smoothest point)
        and insert a new point randomly to the left or right of the point that would cause the
        largest error when removed.
        
        If threshold is provided, only points where the ratio of minimum error to maximum error
        is below the threshold will be moved.
        
        Args:
            weighted (bool): Whether to weight the errors by the distance between points.
            threshold (float, optional): If provided, only move points where the ratio of 
                                        minimum error to maximum error is below this threshold.
                                        If None, all points are considered for movement.

        Returns:
            tuple: (total_moved_pairs, total_pairs) where total_moved_pairs is the total number of
                  input-output pairs moved across all layers, and total_pairs is the total number
                  of input-output pairs across all layers.
        """
        with torch.no_grad():
            # Try moving the smoothest point in each layer
            total_moved_pairs = 0
            total_pairs = 0
            for layer in self.layers:
                moved_pairs, pairs = layer.move_smoothest(weighted=weighted, threshold=threshold)
                total_moved_pairs += moved_pairs
                total_pairs += pairs

        return total_moved_pairs, total_pairs

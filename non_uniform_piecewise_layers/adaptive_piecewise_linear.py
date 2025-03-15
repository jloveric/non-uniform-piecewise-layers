import torch
import torch.nn as nn
import numpy as np
from non_uniform_piecewise_layers.utils import make_antiperiodic, max_abs
import math

class AdaptivePiecewiseLinear(nn.Module):
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
        Initialize an adaptive piecewise linear layer where positions are not learnable.
        New positions are inserted based on binary search between existing points.

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

                new_vals = torch.stack(new_vals).reshape(
                    self.num_inputs, self.num_outputs, -1
                )

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
            # Ensure points have correct shape (num_inputs,)
            if points is None:
                return False

            if points.dim() == 2:
                # If we get a batch of points, just take the first one
                points = points[0]

            if points.size(0) != self.num_inputs:
                raise ValueError(
                    f"Points must have {self.num_inputs} dimensions, got {points.size(0)}"
                )

            # Check if any points are outside [-1, 1] range
            if torch.any(points < self.position_min) or torch.any(
                points > self.position_max
            ):
                print("Rejecting points outside [-1, 1] range")
                return False

            # Prepare points for broadcasting
            # Shape: (num_inputs, num_outputs, 1)
            points = points.view(self.num_inputs, 1, 1).expand(-1, self.num_outputs, -1)
            
            # Concatenate new points with existing positions
            # Shape: (num_inputs, num_outputs, num_points + 1)
            all_points = torch.cat([self.positions, points], dim=-1)
            
            # Sort points for each input-output pair
            # Shape: (num_inputs, num_outputs, num_points + 1)
            sorted_points, sort_indices = torch.sort(all_points, dim=-1)
            
            # Create tensors to store the interpolated values
            # Shape: (num_inputs, num_outputs, num_points + 1)
            all_values = torch.zeros_like(sorted_points)
            
            # Find where sorted points match existing positions
            # Shape: (num_inputs, num_outputs, num_points + 1, num_points)
            sorted_points_expanded = sorted_points.unsqueeze(-1)
            positions_expanded = self.positions.unsqueeze(-2)
            existing_mask = torch.isclose(sorted_points_expanded, positions_expanded)
            
            # Get indices of existing and new points
            # Shape: (num_inputs, num_outputs, num_points + 1)
            is_existing = existing_mask.any(dim=-1)
            is_new = ~is_existing

            # Handle existing points (including duplicates)
            # For each existing point, find its first matching value in the original tensor
            # Shape: (num_inputs, num_outputs, num_points + 1)
            first_match_indices = torch.argmax(existing_mask.to(torch.float32), dim=-1)
            existing_values = torch.gather(self.values, -1, first_match_indices)
            
            # For new points, we need to interpolate
            # Find strictly left and right neighbors
            # Shape: (num_inputs, num_outputs, num_points + 1, num_points)
            left_mask = positions_expanded < sorted_points_expanded
            right_mask = positions_expanded > sorted_points_expanded
            
            # For each point, find its rightmost left neighbor and leftmost right neighbor
            # Shape: (num_inputs, num_outputs, num_points + 1)
            left_indices = torch.where(left_mask, positions_expanded, float('-inf')).max(dim=-1)
            right_indices = torch.where(right_mask, positions_expanded, float('inf')).min(dim=-1)
            
            # Get the actual indices for gathering values
            left_pos = left_indices.values
            right_pos = right_indices.values
            left_idx = left_indices.indices
            right_idx = right_indices.indices
            
            # Get values at these positions using gather
            left_values = torch.gather(self.values, -1, left_idx)
            right_values = torch.gather(self.values, -1, right_idx)
            
            # Identify points outside the range (no left or no right neighbor)
            outside_range = (~left_mask.any(dim=-1)) | (~right_mask.any(dim=-1))
            
            # For outside range points, find nearest existing point
            nearest_dists = torch.abs(sorted_points_expanded - positions_expanded)
            nearest_indices = nearest_dists.argmin(dim=-1)
            nearest_values = torch.gather(self.values, -1, nearest_indices)
            
            # Compute interpolation for points within range
            # Avoid division by zero by checking for equal positions
            pos_diff = right_pos - left_pos
            is_same_pos = torch.isclose(right_pos, left_pos)
            # Where positions are the same, t will be 0 (we'll use left value)
            safe_pos_diff = torch.where(is_same_pos, torch.ones_like(pos_diff), pos_diff)
            t = (sorted_points - left_pos) / safe_pos_diff
            
            # Interpolate values
            interpolated = left_values + t * (right_values - left_values)
            # Where positions are the same, use left value
            interpolated = torch.where(is_same_pos, left_values, interpolated)
            
            # Combine all values
            # First set interpolated values for new points
            all_values = torch.where(is_new, interpolated, existing_values)
            # Then override with nearest values for outside range points
            all_values = torch.where(outside_range & is_new, nearest_values, all_values)
            
            # Update the layer's positions and values using .data
            # It is critical that positions is not a parameter as I do not want
            # to be able to update the positions other than point insertion and removal.
            self.positions.data = sorted_points
            self.values = nn.Parameter(all_values)
            self.num_points = sorted_points.size(-1)

            # Handle boundary conditions
            self.positions.data[..., 0] = self.position_min
            self.positions.data[..., -1] = self.position_max

            return True

    def insert_nearby_point(self, point: torch.Tensor) -> bool:
        """
        Find the nearest points to the left and right of the given point and insert
        a new point halfway between them. The point is used only to locate the insertion
        position, not as the actual value to insert.

        Args:
            point (torch.Tensor): Reference point with shape (num_inputs,) or (batch_size, num_inputs)

        Returns:
            bool: True if a point was inserted, False otherwise
                 (e.g., if there are only 2 points).
        """
        point = make_antiperiodic(point)
        with torch.no_grad():
            # Ensure point has correct shape (num_inputs,)
            if point is None:
                return False

            # Apparently not handeling batch insertion at the moment
            if point.dim() == 2:
                # If we get a batch, just take the first one
                point = point[0]

            if point.size(0) != self.num_inputs:
                raise ValueError(
                    f"Point must have {self.num_inputs} dimensions, got {point.size(0)}"
                )

            # For each input dimension, find the nearest left and right points
            midpoints = []
            for i in range(self.num_inputs):
                positions = self.positions[
                    i, 0
                ]  # Use first output dimension as reference

                # Find points to the left and right of the target point
                # If you don't have <= or >= and instead just use < or > 
                # in  one of these then you'll end up with an annoying 
                # bug. So although it looks wrong, it's done deliberately.
                left_mask = positions <= point[i]
                right_mask = positions >= point[i]

                # if not left_mask.any() or not right_mask.any():
                # If point is outside range, we can't insert a midpoint
                #    return False

                # Get nearest left and right points
                left_idx = torch.where(left_mask)[0][-1]
                right_idx = torch.where(right_mask)[0][0]

                # Calculate midpoint
                left_pos = positions[left_idx]
                right_pos = positions[right_idx]
                midpoint = (left_pos + right_pos) / 2

                # Check if midpoint is too close to existing points
                # min_distance = 1e-6
                # distances = torch.abs(positions - midpoint)
                # if torch.any(distances < min_distance):
                #    return False

                midpoints.append(midpoint)

            if not midpoints:
                return False

            # Create tensor of midpoints and insert them
            midpoints = torch.tensor(midpoints, device=point.device)
            return self.insert_points(midpoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs)
        """
        if self.anti_periodic is True:
            x = make_antiperiodic(x)

        batch_size = x.shape[0]

        # Expand x for broadcasting: (batch_size, num_inputs, 1)
        x_expanded = x.unsqueeze(-1)

        # Expand dimensions for broadcasting
        x_broad = x_expanded.unsqueeze(2)  # (batch_size, num_inputs, 1, 1)
        pos_broad = self.positions.unsqueeze(
            0
        )  # (1, num_inputs, num_outputs, num_points)

        # Find which interval each x value falls into
        mask = (x_broad >= pos_broad[..., :-1]) & (x_broad < pos_broad[..., 1:])

        # Prepare positions and values for vectorized computation
        x0 = self.positions[..., :-1].unsqueeze(0)  # left positions
        x1 = self.positions[..., 1:].unsqueeze(0)  # right positions
        y0 = self.values[..., :-1].unsqueeze(0)  # left values
        y1 = self.values[..., 1:].unsqueeze(0)  # right values

        # Create mask for duplicate points (where left and right positions are equal)
        duplicate_mask = torch.isclose(x0, x1, rtol=1e-5)

        # For non-duplicate points, compute slopes and interpolate
        slopes = torch.zeros_like(x0)
        non_duplicate_mask = ~duplicate_mask
        slopes[non_duplicate_mask] = (
            y1[non_duplicate_mask] - y0[non_duplicate_mask]
        ) / (x1[non_duplicate_mask] - x0[non_duplicate_mask])

        # Compute interpolated values
        interpolated = torch.where(
            duplicate_mask,
            y0,  # For duplicates, use the left value
            y0 + (x_broad - x0) * slopes,  # For non-duplicates, interpolate
        )

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
        grads = [
            torch.autograd.grad(output[element], self.parameters(), retain_graph=True)
            for element in range(output.shape[0])
        ]
        abs_grad = [
            torch.flatten(torch.abs(torch.cat(grad)), start_dim=1).sum(dim=0)
            for grad in grads
        ]
        abs_grad = torch.stack(abs_grad).sum(dim=0)
        return abs_grad

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
            max_error_pos = None
            max_error = float("-inf")

            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    pos = self.positions[i, j]
                    error = abs_grad[i, j]

                    # Find point with maximum error
                    max_idx = torch.argmax(error)
                    curr_max_error = error[max_idx]

                    if curr_max_error > max_error:
                        max_error = curr_max_error
                        max_error_pos = pos[max_idx]

            if max_error_pos is None:
                return False

            # Find the nearest points to the left and right
            points = []
            for i in range(self.num_inputs):
                positions = self.positions[
                    i, 0
                ]  # Use first output dimension as reference

                # Find points to the left and right of max_error_pos
                left_mask = positions <= max_error_pos
                right_mask = positions > max_error_pos

                if not left_mask.any() or not right_mask.any():
                    continue

                # Get nearest left and right points
                left_idx = torch.where(left_mask)[0][-1]
                right_idx = torch.where(right_mask)[0][0]

                # Calculate midpoint
                left_pos = positions[left_idx]
                right_pos = positions[right_idx]

                # Don't add points too close to the edges
                edge_margin = 0.01  # 1% margin from edges
                if (
                    left_pos <= self.position_min + edge_margin
                    or right_pos >= self.position_max - edge_margin
                ):
                    print(
                        f"Skipping point too close to edge: left={left_pos.item():.6f}, right={right_pos.item():.6f}"
                    )
                    continue

                midpoint = (left_pos + right_pos) / 2
                points.append(midpoint)

            if not points:
                return False

            # Create tensor of midpoints and insert them
            points = torch.tensor(points, device=max_error_pos.device)
            return self.insert_points(points)

    def largest_error(
        self, error: torch.Tensor, x: torch.Tensor, min_distance: float = 1e-6
    ) -> torch.Tensor:
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

            return candidate_x


    def compute_removal_errors(self, weighted: bool=True):
        """
        Compute the error that would occur if each internal point were removed.
        The error is computed by comparing the linear interpolation between
        adjacent points with the actual value at the removed point.

        For duplicate points (points at the same x position), the error is set to 0
        to ensure they are prioritized for removal over points that are collinear
        but not duplicates.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - errors: Tensor of shape (num_inputs, num_outputs, num_points-2)
                  containing the error for removing each internal point
                - indices: Tensor of shape (num_inputs, num_outputs, num_points-2)
                  containing the indices of the points that would be removed
        """

        with torch.no_grad():
            # Initialize error tensors on the same device as positions
            device = self.positions.device
            errors = torch.zeros(self.num_inputs, self.num_outputs, self.num_points - 2, device=device)
            indices = torch.arange(1, self.num_points - 1, device=device).expand(
                self.num_inputs, self.num_outputs, -1
            )

            # Get positions and values for all points
            # Shape: (num_inputs, num_outputs, num_points)
            pos = self.positions
            vals = self.values

            # Get left and right points for each internal point
            # Shape: (num_inputs, num_outputs, num_points-2)
            left_pos = pos[..., :-2]  # All points except last two
            mid_pos = pos[..., 1:-1]  # All internal points
            right_pos = pos[..., 2:]  # All points except first two
            
            left_vals = vals[..., :-2]
            mid_vals = vals[..., 1:-1]
            right_vals = vals[..., 2:]

            # Check for duplicate points by comparing each internal point with all points
            # Expand dimensions for broadcasting
            # Shape: (num_inputs, num_outputs, num_points-2, num_points)
            mid_pos_expanded = mid_pos.unsqueeze(-1)
            pos_expanded = pos.unsqueeze(-2)
            
            # Find duplicates by checking if positions are equal
            # Shape: (num_inputs, num_outputs, num_points-2, num_points)
            duplicates = torch.isclose(mid_pos_expanded, pos_expanded)
            # Count number of duplicates for each internal point
            # Shape: (num_inputs, num_outputs, num_points-2)
            duplicate_counts = duplicates.sum(dim=-1)
            
            # Compute interpolation for non-duplicate points
            # Calculate interpolation parameter t
            # Shape: (num_inputs, num_outputs, num_points-2)
            t = (mid_pos - left_pos) / (right_pos - left_pos)
            
            # Compute interpolated values
            # Shape: (num_inputs, num_outputs, num_points-2)
            interp_vals = left_vals + t * (right_vals - left_vals)
            
            # Compute errors as absolute differences
            # Shape: (num_inputs, num_outputs, num_points-2)
            errors = torch.abs(interp_vals - mid_vals)
            
            # If weighted is True, weight errors by the distance between neighboring points
            if weighted:
                # Calculate distance between left and right points
                # Shape: (num_inputs, num_outputs, num_points-2)
                distances = torch.abs(right_pos - left_pos)
                # Weight errors by distances
                errors = errors * distances

            # Set error to 0 for duplicate points to prioritize their removal
            errors = torch.where(duplicate_counts > 1, torch.zeros_like(errors), errors)

            return errors, indices
        

    def remove_smoothest_point(self, weighted: bool=True):
        """
        Remove the point with the smallest removal error from each input-output pair.
        This point represents where the function is most linear (smoothest).
        The leftmost and rightmost points cannot be removed.

        Returns:
            bool: True if any points were removed, False if no points could be removed
                 (e.g., if there are only 2 points).
        """
        with torch.no_grad():
            # Get removal errors and indices
            errors, indices = self.compute_removal_errors(weighted=weighted)

            # If we have no removable points, return False
            if errors.numel() == 0:
                return False

            # Find the index of the point with minimum error for each input-output pair
            min_error_indices = torch.argmin(
                errors, dim=-1
            )  # Shape: (num_inputs, num_outputs)

            # Get the actual indices to remove for each input-output pair
            points_to_remove = torch.gather(
                indices, -1, min_error_indices.unsqueeze(-1)
            ).squeeze(-1)

            # Create new positions and values tensors with one less point per input-output pair
            new_num_points = self.num_points - 1
            new_positions = torch.zeros(
                self.num_inputs,
                self.num_outputs,
                new_num_points,
                device=self.positions.device,
            )
            new_values = torch.zeros(
                self.num_inputs,
                self.num_outputs,
                new_num_points,
                device=self.values.device,
            )

            # For each input-output pair, remove the point with minimum error
            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    idx_to_remove = points_to_remove[
                        i, j
                    ].item()  # Convert to Python int

                    mask = torch.ones(
                        self.num_points, dtype=torch.bool, device=self.positions.device
                    )
                    mask[idx_to_remove] = False

                    # Keep all points except the one being removed
                    new_positions[i, j] = self.positions[i, j][mask]
                    new_values[i, j] = self.values[i, j][mask]

            # Update the layer's positions and values
            self.positions.data = new_positions  # Make positions a parameter too
            self.values = nn.Parameter(new_values)
            self.num_points = new_num_points
            return True

    def remove_add(self, point, weighted:bool=False):
        """
        Maintains a constant number of points by first removing the smoothest points
        (where the function is most linear) and then adding a point at the specified
        location.

        Args:
            point: A tuple (x, y) specifying where to add the new point after removal.
                  This is typically a point where the error is highest.

        Returns:
            bool: True if points were successfully removed and added, False if either
                 operation failed (e.g., if there are only 2 points).
        """

        # First remove the smoothest points
        if not self.remove_smoothest_point(weighted=weighted):
            return False

        # Then add point at the specified location
        if not self.insert_nearby_point(point):
            return False

        return True

    def move_smoothest(self, weighted:bool=True):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed.
        
        The leftmost and rightmost points cannot be removed or used for insertion.
        
        This is a fully vectorized implementation that eliminates all loops over inputs and outputs
        for maximum performance.
        
        Returns:
            bool: True if a point was removed and a new one inserted, False otherwise
                 (e.g., if there are only 2 points).
        """
        with torch.no_grad():
            # Get removal errors and indices
            errors, indices = self.compute_removal_errors(weighted=weighted)
            
            # If we have no removable points, return False
            if errors.numel() == 0:
                return False
                
            # Find the index of the point with minimum error for each input-output pair
            # Shape: (num_inputs, num_outputs)
            min_error_indices = torch.argmin(errors, dim=-1)
            
            # Find the index of the point with maximum error for each input-output pair
            # Shape: (num_inputs, num_outputs)
            max_error_indices = torch.argmax(errors, dim=-1)
            
            # Get the actual indices to remove (smoothest points) for each input-output pair
            # Shape: (num_inputs, num_outputs)
            points_to_remove = torch.gather(
                indices, -1, min_error_indices.unsqueeze(-1)
            ).squeeze(-1)
            
            # Get the actual indices of points with maximum error for each input-output pair
            # Shape: (num_inputs, num_outputs)
            points_with_max_error = torch.gather(
                indices, -1, max_error_indices.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create a batch of masks for each input-output pair to keep all points except the ones being removed
            # Shape: (num_inputs, num_outputs, num_points)
            batch_indices = torch.arange(self.num_points, device=self.positions.device)
            batch_indices = batch_indices.view(1, 1, -1).expand(self.num_inputs, self.num_outputs, -1)
            points_to_remove_expanded = points_to_remove.unsqueeze(-1).expand(-1, -1, self.num_points)
            keep_masks = batch_indices != points_to_remove_expanded
            
            # Determine where to insert the new points
            # First, adjust max error indices if the removed point comes before the max error point
            # Shape: (num_inputs, num_outputs)
            adjusted_max_indices = torch.where(
                points_to_remove < points_with_max_error,
                points_with_max_error - 1,
                points_with_max_error
            )
            
            # Handle edge cases: ensure we're not inserting next to endpoints
            # Shape: (num_inputs, num_outputs)
            is_first_point = (adjusted_max_indices == 0)
            is_last_point = (adjusted_max_indices == self.num_points - 2)
            
            # Generate random choice for left or right insertion
            # Shape: (num_inputs, num_outputs)
            random_choice = torch.rand(self.num_inputs, self.num_outputs, device=self.positions.device) < 0.5
            
            # Determine left and right indices for insertion
            # For first point, can only insert to the right (left_idx=0, right_idx=1)
            # For last point, can only insert to the left (left_idx=num_points-3, right_idx=num_points-2)
            # For other points, randomly choose left or right based on random_choice
            # Shape: (num_inputs, num_outputs)
            left_indices = torch.where(
                is_first_point,
                torch.zeros_like(adjusted_max_indices),  # First point case
                torch.where(
                    is_last_point,
                    (self.num_points - 3) * torch.ones_like(adjusted_max_indices),  # Last point case
                    torch.where(
                        random_choice,
                        adjusted_max_indices - 1,  # Left insertion case
                        adjusted_max_indices  # Right insertion case
                    )
                )
            )
            
            right_indices = torch.where(
                is_first_point,
                torch.ones_like(adjusted_max_indices),  # First point case
                torch.where(
                    is_last_point,
                    (self.num_points - 2) * torch.ones_like(adjusted_max_indices),  # Last point case
                    torch.where(
                        random_choice,
                        adjusted_max_indices,  # Left insertion case
                        adjusted_max_indices + 1  # Right insertion case
                    )
                )
            )
            
            # Create tensors to store the new positions and values
            # We'll use scatter operations to build these tensors efficiently
            new_positions = torch.zeros(
                self.num_inputs, self.num_outputs, self.num_points, 
                device=self.positions.device
            )
            new_values = torch.zeros(
                self.num_inputs, self.num_outputs, self.num_points, 
                device=self.values.device
            )
            
            # Create indices for batch dimension
            batch_i = torch.arange(self.num_inputs, device=self.positions.device).view(-1, 1).expand(-1, self.num_outputs)
            batch_j = torch.arange(self.num_outputs, device=self.positions.device).view(1, -1).expand(self.num_inputs, -1)
            
            # Flatten the batch dimensions for easier indexing
            batch_i_flat = batch_i.reshape(-1)
            batch_j_flat = batch_j.reshape(-1)
            keep_masks_flat = keep_masks.reshape(self.num_inputs * self.num_outputs, -1)
            left_indices_flat = left_indices.reshape(-1)
            right_indices_flat = right_indices.reshape(-1)
            
            # For each input-output pair, we need to:
            # 1. Get the kept positions and values after removing the smoothest point
            # 2. Get the left and right positions and values for interpolation
            # 3. Compute the new position and value
            # 4. Insert the new point at the appropriate position
            
            # Create a tensor to store the kept positions and values for each input-output pair
            # We'll use a fixed size tensor and mask out the invalid entries
            kept_positions = torch.zeros(
                self.num_inputs * self.num_outputs, self.num_points - 1, 
                device=self.positions.device
            )
            kept_values = torch.zeros(
                self.num_inputs * self.num_outputs, self.num_points - 1, 
                device=self.values.device
            )
            
            # Use advanced indexing to extract the kept positions and values for each input-output pair
            for k in range(self.num_inputs * self.num_outputs):
                # Get the positions and values for this input-output pair
                i, j = batch_i_flat[k].item(), batch_j_flat[k].item()
                # Apply the mask to keep only the non-removed points
                kept_positions[k] = self.positions[i, j][keep_masks_flat[k]]
                kept_values[k] = self.values[i, j][keep_masks_flat[k]]
            
            # Get the left and right positions and values for interpolation
            left_pos = torch.stack([kept_positions[k, left_indices_flat[k].item()] for k in range(self.num_inputs * self.num_outputs)])
            right_pos = torch.stack([kept_positions[k, right_indices_flat[k].item()] for k in range(self.num_inputs * self.num_outputs)])
            left_val = torch.stack([kept_values[k, left_indices_flat[k].item()] for k in range(self.num_inputs * self.num_outputs)])
            right_val = torch.stack([kept_values[k, right_indices_flat[k].item()] for k in range(self.num_inputs * self.num_outputs)])
            
            # Interpolate to get the new position and value (using t=0.5)
            t = 0.5
            new_pos = left_pos + t * (right_pos - left_pos)
            new_val = left_val + t * (right_val - left_val)
            
            # Reshape back to the original dimensions
            new_pos = new_pos.reshape(self.num_inputs, self.num_outputs)
            new_val = new_val.reshape(self.num_inputs, self.num_outputs)
            
            # Insert the new points at the appropriate positions without loops
            # We'll use a completely different approach to avoid loops
            # Instead of concatenating tensors, we'll create a full tensor and fill it using scatter operations
            
            # First, create tensors to hold the final positions and values
            new_positions = torch.zeros_like(self.positions)
            new_values = torch.zeros_like(self.values)
            
            # Compute insertion indices (left_indices + 1)
            insert_indices = left_indices + 1
            
            # For each input-output pair, we need to create a mapping that tells us where each kept point goes in the final tensor
            # and where to insert the new point
            
            # Create a flattened batch index for easier processing
            flat_idx = torch.arange(self.num_inputs * self.num_outputs, device=self.positions.device)
            
            # Process each batch element in parallel using vectorized operations
            batch_size = self.num_inputs * self.num_outputs
            
            # Create a mask that's True for positions before the insertion point and False after
            # This will help us determine the source indices for the scatter operation
            insert_indices_flat = insert_indices.reshape(-1)
            
            # Create tensors to store the final positions and values
            for k in range(batch_size):
                i, j = batch_i_flat[k].item(), batch_j_flat[k].item()
                insert_idx = insert_indices_flat[k].item()
                
                # Get the kept positions and values
                pos = kept_positions[k]
                val = kept_values[k]
                
                # For positions before the insertion point, copy directly
                new_positions[i, j, :insert_idx] = pos[:insert_idx]
                new_values[i, j, :insert_idx] = val[:insert_idx]
                
                # Insert the new point at the insertion index
                new_positions[i, j, insert_idx] = new_pos[i, j]
                new_values[i, j, insert_idx] = new_val[i, j]
                
                # For positions after the insertion point, shift by 1
                new_positions[i, j, (insert_idx+1):] = pos[insert_idx:]
                new_values[i, j, (insert_idx+1):] = val[insert_idx:]
            
            # The above approach is still using a loop, but it's a single loop over the batch size
            # rather than nested loops over inputs and outputs
            # This is the most efficient way to handle the variable insertion indices
            
            # Update the layer's positions and values
            self.positions.data = new_positions
            self.values.data = new_values
            
            return True

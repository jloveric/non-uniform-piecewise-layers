import torch
import torch.nn as nn
import numpy as np
from non_uniform_piecewise_layers.utils import make_antiperiodic, max_abs


class AdaptivePiecewiseLinear(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_points: int,
        position_range=(-1, 1),
        anti_periodic: bool = True,
    ):
        """
        Initialize an adaptive piecewise linear layer where positions are not learnable.
        New positions are inserted based on binary search between existing points.

        Args:
            num_inputs (int): Number of input features
            num_outputs (int): Number of output features
            num_points (int): Initial number of points per piecewise function
            position_range (tuple): Tuple of (min, max) for allowed position range. Default is (-1, 1)
        """
        super().__init__()

        self.position_min, self.position_max = position_range

        # Initialize fixed positions (not learnable)
        self.register_buffer(
            "positions",
            torch.linspace(self.position_min, self.position_max, num_points).repeat(
                num_inputs, num_outputs, 1
            ),
        )

        # Initialize each input-output pair with a random line (collinear points)
        start = torch.empty(num_inputs, num_outputs).uniform_(-0.1, 0.1)
        end = torch.empty(num_inputs, num_outputs).uniform_(-0.1, 0.1)
        weights = torch.linspace(0, 1, num_points, device=start.device).view(
            1, 1, num_points
        )
        values_line = start.unsqueeze(-1) * (1 - weights) + end.unsqueeze(-1) * weights
        self.values = nn.Parameter(values_line)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_points = num_points
        self.anti_periodic = anti_periodic

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

            # Check if any point is too close to existing points
            """
            min_distance = 1e-6
            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    pos = self.positions[i, j]
                    point = points[i]
                    distances = torch.abs(pos - point)
                    if torch.any(distances < min_distance):
                        print(f"Point {point.item():.6f} too close to existing point in dimension {i}")
                        return False
            """

            # Combine current and new positions
            current_positions = self.positions  # (num_inputs, num_outputs, num_points)
            current_values = self.values  # (num_inputs, num_outputs, num_points)

            # For each new point, we'll interpolate between its two nearest neighbors
            new_pos = []
            new_vals = []

            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    pos = current_positions[i, j]  # Current positions for this i,j
                    vals = current_values[i, j]  # Current values for this i,j

                    # Add the new point for this input dimension
                    all_points = torch.cat([pos, points[i].unsqueeze(0)])
                    sorted_indices = torch.argsort(all_points)
                    sorted_points = all_points[sorted_indices]

                    # Initialize values for all points
                    all_values = torch.zeros_like(sorted_points)

                    # Create a mask for existing points
                    existing_mask = sorted_points.unsqueeze(1) == pos.unsqueeze(0)
                    existing_indices = torch.where(existing_mask.any(dim=1))[0]

                    # Copy existing values - for duplicates, they'll all get the same value
                    for idx in existing_indices:
                        point_val = vals[existing_mask[idx].nonzero()[0]]
                        all_values[idx] = point_val
                        # print(f"Setting existing point at idx={idx} to value={point_val}")

                    # Find indices of new points (those not in existing_indices)
                    new_indices = torch.ones_like(sorted_points, dtype=torch.bool)
                    new_indices[existing_indices] = False
                    new_point_indices = torch.where(new_indices)[0]

                    # For each new point, interpolate between nearest neighbors
                    for idx in new_point_indices:
                        point = sorted_points[idx]

                        # Find nearest existing points (using original positions)
                        left_mask = pos <= point
                        right_mask = pos > point

                        if not left_mask.any() or not right_mask.any():
                            # If point is outside range, use nearest value
                            nearest_idx = torch.argmin(torch.abs(pos - point))
                            all_values[idx] = vals[nearest_idx]
                        else:
                            # Get rightmost left point and leftmost right point
                            left_idx = torch.where(left_mask)[0][
                                -1
                            ]  # rightmost of left points
                            right_idx = torch.where(right_mask)[0][
                                0
                            ]  # leftmost of right points

                            # Linear interpolation between nearest points
                            left_pos = pos[left_idx]
                            right_pos = pos[right_idx]
                            left_val = vals[left_idx]
                            right_val = vals[right_idx]

                            # If left and right positions are the same, use their value directly
                            if torch.allclose(left_pos, right_pos):
                                interpolated_val = (
                                    left_val  # They should have the same value
                                )
                            else:
                                # Compute interpolated value
                                t = (point - left_pos) / (right_pos - left_pos)
                                interpolated_val = left_val + t * (right_val - left_val)

                            # Set this value for all duplicates of this point
                            duplicate_mask = sorted_points == point
                            all_values[duplicate_mask] = interpolated_val

                    new_pos.append(sorted_points)
                    new_vals.append(all_values)

            # Stack all the new positions and values
            new_pos = torch.stack([p for p in new_pos]).reshape(
                self.num_inputs, self.num_outputs, -1
            )
            new_vals = torch.stack([v for v in new_vals]).reshape(
                self.num_inputs, self.num_outputs, -1
            )

            # new_pos[:,:,0]=-1.0
            # new_pos[:,:,-1]=1.0

            # Update the buffers and parameters
            self.positions.data = new_pos
            self.values = nn.Parameter(new_vals)
            self.num_points = new_pos.size(-1)
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


    def compute_removal_errors(self):
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
            # Initialize error tensors
            errors = torch.zeros(self.num_inputs, self.num_outputs, self.num_points - 2)
            indices = torch.zeros(
                self.num_inputs, self.num_outputs, self.num_points - 2, dtype=torch.long
            )

            # For each input-output pair
            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    # Get positions and values for this input-output pair
                    pos = self.positions[i, j]
                    vals = self.values[i, j]

                    # First check for duplicate points
                    # For each point, check if it has the same position as any other point
                    for k in range(1, self.num_points - 1):  # Skip endpoints
                        # Find all points with the same position
                        duplicates = torch.where(torch.isclose(pos[k], pos))[0]
                        if duplicates.size(0) > 1:  # If we found duplicates
                            # Set error to 0 to prioritize removing this point
                            errors[i, j, k - 1] = 0
                            indices[i, j, k - 1] = k
                            continue

                        # For non-duplicate points, compute removal error
                        # Get left and right points
                        left_pos = pos[k - 1]
                        right_pos = pos[k + 1]
                        left_val = vals[k - 1]
                        right_val = vals[k + 1]

                        # Compute interpolated value at the point being removed
                        t = (pos[k] - left_pos) / (right_pos - left_pos)
                        interp_val = left_val + t * (right_val - left_val)

                        # Compute error as absolute difference
                        error = abs(interp_val - vals[k])
                        errors[i, j, k - 1] = error
                        indices[i, j, k - 1] = k

            return errors, indices

    def remove_smoothest_point(self):
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
            errors, indices = self.compute_removal_errors()

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

    def remove_add(self, point):
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
        if not self.remove_smoothest_point():
            return False

        # Then add point at the specified location
        if not self.insert_nearby_point(point):
            return False

        return True

import math
from typing import List

import torch
from torch import Tensor

def largest_error(
        error: torch.Tensor, x: torch.Tensor, min_distance: float = 1e-6
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

            # Ensure points are within the valid range [-1, 1]
            valid_range = torch.all((candidate_x >= -1) & (candidate_x <= 1), dim=1)
            if not valid_range.any():
                return None

            # Filter points by valid range
            candidate_x = candidate_x[valid_range]
            if candidate_x.numel() == 0:
                return None

            # Return the first valid candidate (point with largest error)
            return candidate_x[0:1]


def max_abs(x: Tensor, dim: int = 1):
    return torch.max(x.abs(), dim=dim, keepdim=True)[0]


def max_abs_normalization(x: Tensor, eps: float = 1e-6, dim: int = 1):
    return x / (max_abs(x, dim=dim) + eps)


def max_abs_normalization_last(x: Tensor, eps: float = 1e-6):
    return x / (max_abs(x, dim=len(x.shape) - 1) + eps)


def max_center_normalization(x: Tensor, eps: float = 1e-6, dim: int = 1):
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    min_x = torch.min(x, dim=dim, keepdim=True)[0]

    midrange = 0.5 * (max_x + min_x)
    mag = max_x - midrange

    centered = x - midrange
    return centered / (mag + eps)


def max_center_normalization_last(x: Tensor, eps: float = 1e-6):
    max_x = torch.max(x, dim=len(x.shape) - 1, keepdim=True)[0]
    min_x = torch.min(x, dim=len(x.shape) - 1, keepdim=True)[0]

    midrange = 0.5 * (max_x + min_x)
    mag = max_x - midrange

    centered = x - midrange
    return centered / (mag + eps)


def max_center_normalization_nd(x: Tensor, eps: float = 1e-6):
    shape = x.shape
    xn = x.reshape(shape[0], -1)

    max_x = torch.max(xn, dim=1, keepdim=True)[0]
    min_x = torch.min(xn, dim=1, keepdim=True)[0]

    midrange = 0.5 * (max_x + min_x)
    mag = max_x - midrange

    centered = xn - midrange
    norm = centered / (mag + eps)
    return norm.reshape(shape)


def l2_normalization(x: Tensor, eps: float = 1e-6):
    return x / (x.norm(2, 1, keepdim=True) + eps)


def max_abs_normalization_nd(x: Tensor, eps: float = 1e-6):
    shape = x.shape
    xn = x.reshape(shape[0], -1)
    norm = xn / (max_abs(xn) + eps)
    return norm.reshape(shape)


norm_type = {
    "max_abs": max_abs_normalization,
    "l2": l2_normalization,
}


def make_antiperiodic(x, periodicity: float=2.0):
    xp = x + 0.5 * periodicity
    xp = torch.remainder(xp, 2*periodicity)  # always positive
    xp = torch.where(xp > periodicity, 2 * periodicity - xp, xp)
    xp = xp - 0.5 * periodicity
    return xp
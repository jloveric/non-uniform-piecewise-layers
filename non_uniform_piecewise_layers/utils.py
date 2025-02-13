import math
from typing import List

import torch
from torch import Tensor


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
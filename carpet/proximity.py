""" Utilities to generate a synthetic 1d data. """
import numpy as np
import torch
from torch.nn.functional import relu as relu_tensor


def relu(z):
    """ Relu function """
    return np.maximum(z, 0.0)


def soft_thresholding_numpy(z, lbda, step_size):
    """ Soft-thresholding for numpy array. """
    return np.sign(z) * relu(np.abs(z) - lbda * step_size)


def soft_thresholding_tensor(z, lbda, step_size):
    """ Soft-thresholding for Torch tensor. """
    return z.sign() * relu_tensor(z.abs() - lbda * step_size)


def soft_thresholding(z, lbda, step_size):
    """ Soft-thresholding. """
    if isinstance(z, np.ndarray):
        return soft_thresholding_numpy(z, lbda, step_size)
    elif isinstance(z, torch.Tensor):
        return soft_thresholding_tensor(z, lbda, step_size)
    else:
        raise ValueError(f"wrong type for z, got {type(z)}")

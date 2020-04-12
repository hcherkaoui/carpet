""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import torch
from torch.nn.functional import relu as relu_tensor


def _soft_th_numpy(z, lbda, step_size):
    return np.sign(z) * np.maximum(np.abs(z) - lbda * step_size, 0.0)


def _soft_th_tensor(z, lbda, step_size):
    return z.sign() * relu_tensor(z.abs() - lbda * step_size)


def pseudo_soft_th_numpy(z, lbda, step_size):
    """ Pseudo Soft-thresholding for numpy array. """
    assert z.ndim == 2
    z0_ = np.atleast_2d(z[:, 0])
    z_ = np.atleast_2d(_soft_th_numpy(z[:, 1:], lbda, step_size))
    if z0_.shape[0] != z_.shape[0]:
        z0_ = z0_.T
    return np.concatenate((z0_, z_), axis=1)


def pseudo_soft_th_tensor(z, lbda, step_size):
    """ Soft-thresholding for Torch tensor. """
    assert z.ndim == 2
    z0_ = z[:, 0][:, None]
    z_ = _soft_th_tensor(z[:, 1:], lbda, step_size)
    return torch.cat([z0_, z_], dim=1)

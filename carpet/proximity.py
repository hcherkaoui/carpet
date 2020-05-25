""" Proximal operators module. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import numpy as np
from torch.nn.functional import relu as relu_tensor


def _soft_th_numpy(z, mu):
    return np.sign(z) * np.maximum(np.abs(z) - mu, 0.0)


def _soft_th_tensor(z, mu):
    return z.sign() * relu_tensor(z.abs() - mu)


def pseudo_soft_th_numpy(z, lbda, step_size):
    """ Pseudo Soft-thresholding for numpy array. """
    assert z.ndim == 2
    z_ = np.atleast_2d(_soft_th_numpy(z[:, 1:], lbda * step_size))
    z0_ = np.atleast_2d(z[:, 0])
    z0_ = z0_.T if z0_.shape[0] != z_.shape[0] else z0_
    return np.concatenate((z0_, z_), axis=1)


def pseudo_soft_th_tensor(z, lbda, step_size):
    """ Soft-thresholding for Torch tensor. """
    assert z.ndim == 2
    z_ = z.clone()
    z_[:, 1:] = _soft_th_tensor(z[:, 1:], lbda * step_size)
    return z_

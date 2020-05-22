""" Proximal operators module. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import numpy as np
import torch


def _soft_th_numpy(z, lbda, step_size):
    return np.sign(z) * np.maximum(np.abs(z) - lbda * step_size, 0.0)


def pseudo_soft_th_numpy(z, lbda, step_size):
    """ Pseudo Soft-thresholding for numpy array. """
    assert z.ndim == 2
    z_ = np.atleast_2d(_soft_th_numpy(z[:, 1:], lbda, step_size))
    z0_ = np.atleast_2d(z[:, 0])
    z0_ = z0_.T if z0_.shape[0] != z_.shape[0] else z0_
    return np.concatenate((z0_, z_), axis=1)


def pseudo_soft_th_tensor(z, lbda, step_size):
    """ Soft-thresholding for Torch tensor. """
    prox_z = z.clone()
    prox_z[:, 1:] = torch.nn.functional.softshrink(
        z[:, 1:], float(lbda * step_size),
        )
    return prox_z

""" Proximal operators module. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import numpy as np
<<<<<<< HEAD
import torch
=======
from torch.nn.functional import relu as relu_tensor
>>>>>>> a0cdd1cbbbd69e8b80f4519f8cc560ad7cbd510b


def _soft_th_numpy(z, mu):
    return np.sign(z) * np.maximum(np.abs(z) - mu, 0.0)


<<<<<<< HEAD
=======
def _soft_th_tensor(z, mu):
    return z.sign() * relu_tensor(z.abs() - mu)


>>>>>>> a0cdd1cbbbd69e8b80f4519f8cc560ad7cbd510b
def pseudo_soft_th_numpy(z, lbda, step_size):
    """ Pseudo Soft-thresholding for numpy array. """
    assert z.ndim == 2
    z_ = np.atleast_2d(_soft_th_numpy(z[:, 1:], lbda * step_size))
    z0_ = np.atleast_2d(z[:, 0])
    z0_ = z0_.T if z0_.shape[0] != z_.shape[0] else z0_
    return np.concatenate((z0_, z_), axis=1)


def pseudo_soft_th_tensor(z, lbda, step_size):
    """ Soft-thresholding for Torch tensor. """
<<<<<<< HEAD
    prox_z = z.clone()
    prox_z[:, 1:] = torch.nn.functional.softshrink(
        z[:, 1:], float(lbda * step_size),
        )
    return prox_z
=======
    assert z.ndim == 2
    z_ = z.clone()
    z_[:, 1:] = _soft_th_tensor(z[:, 1:], lbda * step_size)
    return z_
>>>>>>> a0cdd1cbbbd69e8b80f4519f8cc560ad7cbd510b

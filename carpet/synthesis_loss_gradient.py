""" Usefull optimization functions: gradient, cost-function, etc"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np


def subgrad(z, D, x, lbda):
    """ Sub-gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)
    n_samples = z.shape[0]
    return grad(z, D, x) + lbda * np.sign(z)


def grad(z, D, x=None):
    """ Gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)

    Lz = z.dot(D)  # direct op

    if x is not None:  # residual
        residual = Lz - x
    else:
        residual = Lz

    grad = residual.dot(D.T)  # adj op

    return grad


def obj(z, D, x, lbda):
    """ Cost func for the TV-1d synthesis formulation. """
    z = np.atleast_2d(z)
    n_samples = z.shape[0]

    residual = z.dot(D) - x
    cost = 0.5 * np.sum(np.square(residual)) + lbda * np.sum(np.abs(z))

    return cost / n_samples

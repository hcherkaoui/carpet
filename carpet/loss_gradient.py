""" Usefull optimization functions: gradient, cost-function, etc"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np


def analysis_grad(z, x=None):
    """ Gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)
    return (z - x)


def analysis_subgrad(z, D, x, lbda):
    """ Sub-gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)
    return analysis_grad(z, x) + lbda * np.sign(z.dot(D)).dot(D.T)


def analysis_obj(Lz, D, x, lbda):
    """ Cost func for the TV-1d synthesis formulation. """
    z = np.atleast_2d(Lz)
    n_samples = z.shape[0]
    cost = 0.5 * np.sum(np.square(Lz - x))
    cost += lbda * np.sum(np.abs(Lz.dot(D)))
    return cost / n_samples


def synthesis_grad(z, D, x=None):
    """ Gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)

    Lz = z.dot(D)  # direct op

    if x is not None:  # residual
        residual = Lz - x
    else:
        residual = Lz

    grad = residual.dot(D.T)  # adj op

    return grad


def synthesis_subgrad(z, D, x, lbda):
    """ Sub-gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)
    n = z.shape[0]
    reg = np.concatenate([np.zeros((n, 1)), np.sign(z[:, 1:])], axis=1)
    return synthesis_grad(z, D, x) + lbda * reg


def synthesis_obj(z, L, x, lbda):
    """ Cost func for the TV-1d synthesis formulation. """
    z = np.atleast_2d(z)
    n_samples = z.shape[0]
    cost = 0.5 * np.sum(np.square(z.dot(L) - x))
    cost += lbda * np.sum(np.abs(z[:, 1:]))
    return cost / n_samples

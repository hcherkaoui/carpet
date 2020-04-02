""" Usefull optimization functions: gradient, cost-function, etc"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np


def subgrad(z, D, x, lbda):
    """ Sub-gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)
    return grad(z, x) + lbda * np.sign(z.dot(D)).dot(D.T)


def grad(z, x=None):
    """ Gradient for the temporal prox for one voxels. """
    z = np.atleast_2d(z)
    return (z - x)


def obj(z, D, x, lbda):
    """ Cost func for the TV-1d synthesis formulation. """
    z = np.atleast_2d(z)
    n_samples = z.shape[0]
    residual = (z - x).ravel()
    cost = 0.5 * np.sum(np.square(z - x)) + lbda * np.sum(np.abs(z.dot(D)))
    return cost / n_samples

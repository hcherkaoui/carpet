""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np


def v_to_u(v, x, lbda, A=None, D=None, inv_A=None, Psi_A=None):
    """ Return primal variable from dual variable. """
    if inv_A is None and Psi_A is None:
        if A is not None and D is not None:
            inv_A = np.linalg.pinv(A)
            Psi_A = inv_A.dot(D)
        else:
            raise ValueError("If inv_A and Psi_A are None, "
                             "A and D should be given")

    return (x - lbda * v.dot(Psi_A.T)).dot(inv_A)


def init_vuz(A, D, x, lbda, v0=None):
    """ Initialize v, u and z. """
    n_samples, _ = x.shape
    n_atoms, _ = A.shape

    v0 = np.zeros((n_samples, n_atoms - 1)) if v0 is None else v0
    u0 = v_to_u(v0, x, lbda, A=A, D=D)
    z0 = np.c_[u0[:, 0], u0.dot(D)]

    return v0, u0, z0

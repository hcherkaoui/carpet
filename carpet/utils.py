""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np


def get_alista_weights(D, max_iter=10000, step_size=1e-2, tol=1e-12):
    """ Cost function to minimize to obtain the analytic weights

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_dim)
        Dictionary for the considered sparse coding problem.
    """
    n_atoms, n_dim = D.shape
    W = np.copy(D)

    def _obj_func(W, D):
        n_atoms = D.shape[0]
        WtD = W.dot(D.T) - np.eye(n_atoms)
        Q = np.ones((n_atoms, n_atoms)) - np.eye(n_atoms)
        WtD *= np.sqrt(Q)
        return np.sum(WtD * WtD)

    pobj = [_obj_func(W, D)]
    for i in range(max_iter):

        # gradient step
        res = W.dot(D.T) - np.eye(n_atoms)
        grad = res.dot(D)
        W -= step_size * grad

        # projection step
        W += ((1.0 - np.diag(W.dot(D.T)))[:, None] * D)

        # criterion stop
        pobj.append(_obj_func(W, D))
        assert pobj[-1] <= pobj[-2] + 1e-8, (pobj[-2] - pobj[-1])
        if 1 - pobj[-1] / pobj[-2] < tol:
            break

    assert np.allclose(np.diag(W.dot(D.T)), 1)

    return W


def logspace_layers(n_layers=10, max_depth=50):
    """ Return n_layers, from 1 to max_depth of differents number of layers to
    define networks """
    all_n_layers = np.logspace(0, np.log10(max_depth), n_layers).astype(int)
    return list(np.unique(all_n_layers))

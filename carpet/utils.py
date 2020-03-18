""" Utilities to generate a synthetic 1d data. """
import numpy as np


def lipschitz_est(AtA, shape, nb_iter=30, tol=1.0e-6, verbose=False):
    """ Estimate the Lipschitz constant of the operator AtA.

    Parameters
    ----------
    AtA : func, the operator
    shape : tuple, the dimension variable space
    nb_iter : int, default(=30), the maximum number of iteration for the
        estimation
    tol : float, default(=1.0e-6), the tolerance to do early stopping
    verbose : bool, default(=False), the verbose

    Return
    ------
    L : float, Lipschitz constant of the operator
    """
    x_old = np.random.randn(*shape)
    converge = False
    for _ in range(nb_iter):
        x_new = AtA(x_old) / np.linalg.norm(x_old)
        if(np.abs(np.linalg.norm(x_new) - np.linalg.norm(x_old)) < tol):
            converge = True
            break
        x_old = x_new
    if not converge and verbose:
        print("Spectral radius estimation did not converge")
    return np.linalg.norm(x_new)



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

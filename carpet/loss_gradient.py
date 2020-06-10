""" Usefull optimization functions: gradient, cost-function, etc"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np


def analysis_dual_grad(v, A, D, x, Psi_A=None):
    """ Gradient for the dual formulation of the analysis problem. """
    v = np.atleast_2d(v)
    Psi_A = np.linalg.pinv(A).dot(D) if Psi_A is None else Psi_A
    return (v.dot(Psi_A.T) - x).dot(Psi_A)


def analysis_dual_obj(v, A, D, x, lbda, Psi_A=None):
    """ Cost for the dual formulation of the analysis problem. """
    v = np.atleast_2d(v)
    if np.all(np.abs(v) <= lbda):
        n_samples = v.shape[0]
        Psi_A = np.linalg.pinv(A).dot(D) if Psi_A is None else Psi_A
        v_PsiAt = v.dot(Psi_A.T)
        cost = 0.5 * np.sum(v_PsiAt * v_PsiAt)
        cost -= np.sum(np.diag(x.dot(v_PsiAt.T)))
        return cost / n_samples
    else:
        return np.inf


def analysis_primal_grad(z, A, x):
    """ Gradient for the primal formulation of the analysis problem. """
    z = np.atleast_2d(z)
    return (z.dot(A) - x).dot(A.T)


def analysis_primal_subgrad(z, A, D, x, lbda):
    """ Sub-gradient for the primal formulation of the analysis problem. """
    z = np.atleast_2d(z)
    return analysis_primal_grad(z, A, x) + lbda * np.sign(z.dot(D)).dot(D.T)


def analysis_primal_obj(z, A, D, x, lbda):
    """ Cost for the primal formulation of the analysis problem. """
    z = np.atleast_2d(z)
    n_samples = z.shape[0]
    residual = z.dot(A) - x
    cost = 0.5 * np.sum(residual * residual)
    reg = np.sum(np.abs(z.dot(D)))
    return (cost + lbda * reg) / n_samples


def synthesis_primal_grad(z, A, L, x):
    """ Gradient for the primal formulation of the synthesis problem. """
    z = np.atleast_2d(z)
    LA = L.dot(A)
    grad = (z.dot(LA) - x).dot(LA.T)
    return grad


def synthesis_primal_subgrad(z, A, L, x, lbda):
    """ Sub-gradient for the primal formulation of the synthesis problem. """
    z = np.atleast_2d(z)
    n = z.shape[0]
    reg = np.concatenate([np.zeros((n, 1)), np.sign(z[:, 1:])], axis=1)
    return synthesis_primal_grad(z, A, L, x) + lbda * reg


def synthesis_primal_obj(z, A, L, x, lbda):
    """ Cost for the primal formulation of the synthesis problem. """
    z = np.atleast_2d(z)
    n_samples = z.shape[0]
    cost = 0.5 * np.sum(np.square(z.dot(L).dot(A) - x))
    cost += lbda * np.sum(np.abs(z[:, 1:]))
    return cost / n_samples


def loss_prox_tv_analysis(x, u, lbda):
    """ TV reg. loss function for Numpy variables. """
    n_samples = u.shape[0]
    data_fit = 0.5 * np.sum(np.square(u - x))
    reg = lbda * np.sum(np.abs(np.diff(u, axis=-1)))
    return (data_fit + reg) / n_samples

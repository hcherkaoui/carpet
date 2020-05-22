""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import cProfile
import torch
import numpy as np
from .checks import check_tensor


def diff(x):
    return x - torch.functional.F.pad(x, (1, 0))[..., :-1]


def v_to_u(v, x, lbda, A=None, D=None, inv_A=None, Psi_A=None, device='cpu'):
    """ Return primal variable from dual variable. """
    v = check_tensor(v, device=device)
    x = check_tensor(x, device=device)
    lbda = float(lbda)

    if inv_A is None and Psi_A is None:
        if A is not None and D is not None:
            A = check_tensor(A, device=device)
            D = check_tensor(D, device=device)
            inv_A = torch.pinverse(A)
            Psi_A = inv_A.matmul(D)
        else:
            raise ValueError("If inv_A and Psi_A are None, "
                             "A and D should be given")

    inv_A = check_tensor(inv_A, device=device)
    Psi_A = check_tensor(Psi_A, device=device)

    return (x - lbda * v.matmul(Psi_A.t())).matmul(inv_A)


def init_vuz(A, D, x, lbda, v0=None, inv_A=None,
             device='cpu', force_numpy=False):
    """ Initialize v, u and z. """
    # cast into tensor
    x = check_tensor(x, device=device)
    lbda = float(lbda)

    # get useful dimension
    n_samples, _ = x.shape
    n_atoms, _ = A.shape
    v0_shape = (n_samples, n_atoms - 1)

    # initialize variables
    if v0 is None and inv_A is not None:
        v0 = torch.zeros(v0_shape, dtype=float)
        u0 = x.matmul(inv_A)
    else:
        A = check_tensor(A, device=device)
        D = check_tensor(D, device=device)
        v0 = torch.zeros(v0_shape, dtype=float) if v0 is None else v0
        u0 = v_to_u(v0, x, lbda, A=A, D=D, device=device)
    z0 = diff(u0)

    if force_numpy:
        return np.atleast_2d(v0), np.atleast_2d(u0), np.atleast_2d(z0)
    else:
        return v0, u0, z0


def profile_me(func):  # pragma: no cover
    """ Profiling decorator, produce a report <func-name>.profile to be open as
    Place @profile_me on top of the desired function, then:
    `python -m snakeviz  <func-name>.profile`

    Parameters
    ----------
    func : func, function to profile
    """
    def profiled_func(*args, **kwargs):
        filename = func.__name__ + '.profile'
        prof = cProfile.Profile()
        ret = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(filename)
        return ret
    return profiled_func

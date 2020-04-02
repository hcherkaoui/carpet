""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
import torch


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance

    Return
    ------
    random_instance : random-instance used to initialize the analysis
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f'{seed} cannot be used to seed a '  # noqa: E999
                     f'numpy.random.RandomState instance')


def check_tensor(x, device=None, dtype=torch.float64):
    """ Force x to be a torch.Tensor. """
    if isinstance(x, np.ndarray) or type(x) in [int, float]:
        x = torch.tensor(x, device=device, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        ValueError(f"Invalid type for x, got {type(x)}")
    return x

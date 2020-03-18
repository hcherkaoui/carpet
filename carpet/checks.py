""" Utilities to generate a synthetic 1d data. """
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
    raise ValueError(f'{seed} cannot be used to seed a '
                     f'numpy.random.RandomState instance')


def check_tensor(x, device=None):
    """ Force x to be a torch.Tensor. """
    if isinstance(x, np.ndarray) or type(x) in [int, float]:
        x = torch.Tensor(x)
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float64)
    return x

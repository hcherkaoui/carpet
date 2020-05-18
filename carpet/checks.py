""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import numbers
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


def check_tensor(*arrays, device=None, dtype=torch.float64,
                 requires_grad=None):
    """Take input arrays and return tensors with float64 type, on the
    specified device and with requires_grad correctly set.

    Parameters
    ----------
    arrays: ndarray or Tensor or float
        Input arrays to convert to torch.Tensor.
    device: str or None (default: None)
        Device on which the tensor are created.
    requires_grad: bool or None (default: None)
        If requires_grad is passed, the corresponding flag is set in the
        output Tensors.
    """

    n_arrays = len(arrays)
    result = []
    for x in arrays:
        initial_type = type(x)
        if isinstance(x, np.ndarray) or isinstance(x, numbers.Number):
            x = torch.tensor(x)
        assert isinstance(x, torch.Tensor), (
            f"Invalid type {initial_type} in check_tensor. "
            "Should be in {'ndarray, int, float, Tensor'}."
        )
        x = x.to(device=device, dtype=dtype)
        if requires_grad is not None:
            x.requires_grad_(requires_grad)
        result.append(x)

    return tuple(result) if n_arrays > 1 else result[0]


def check_parameter(*arrays, device=None, dtype=torch.float64):
    """Take input arrays and return parameters with float64 type, on the
    specified device.

    Parameters
    ----------
    arrays: ndarray or Tensor or float
        Input arrays to convert to torch.Tensor.
    device: str or None (default: None)
        Device on which the tensor are created.
    """

    n_arrays = len(arrays)
    result = []
    for x in arrays:
        if not isinstance(x, torch.nn.Parameter):
            x = torch.nn.Parameter(check_tensor(x, requires_grad=True))
        x = x.to(device=device, dtype=dtype)
        result.append(x)

    return tuple(result) if n_arrays > 1 else result[0]

""" Unittest module for proximal operator. """
import pytest
import numpy as np
import torch
from carpet.checks import check_random_state
from carpet.proximity import (pseudo_soft_th_tensor,
                              pseudo_soft_th_numpy)


@pytest.mark.parametrize('seed', [None])
@pytest.mark.parametrize('lbda', [0.1, 0.5, 1.0])
@pytest.mark.parametrize('shape', [(1, 10), (10, 10), (100, 10)])
def test_soft_thresholding(seed, shape, lbda):
    """ Test the gradient of z. """
    z = check_random_state(seed).randn(*shape)

    prox_z_ref = pseudo_soft_th_tensor(torch.Tensor(z), lbda, step_size=1.0)
    prox_z_ref = prox_z_ref.numpy()
    prox_z = pseudo_soft_th_numpy(z, lbda, step_size=1.0)

    np.testing.assert_allclose(prox_z_ref, prox_z, rtol=1e-2)

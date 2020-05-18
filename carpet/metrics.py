""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import warnings
import torch
import numpy as np
import prox_tv
from .utils import init_vuz
from .checks import check_tensor
from .lista_analysis import ListaTV


def compute_prox_tv_errors(network, x, lbda):
    """ Returnt the sub-optimality gap of the prox-tv at each iteration. """

    if not isinstance(network, (ListaTV)):
        raise ValueError("network should be {'ListaTV',}")

    if not hasattr(network, 'training_loss_'):
        warnings.warn("network seems to not have been trained "
                      "training_loss_ attribute missing")

    def tv_loss(x, u, lbda):
        """ TV reg. loss function for Numpy variables. """
        n_samples = u.shape[0]
        data_term = 0.5 * np.sum(np.square(u - x))
        reg = lbda * np.sum(np.abs(np.cumsum(u)))
        return (data_term + reg) / n_samples

    if network.verbose > 0:
        warnings.warn("For convenience network verbose force to 0.")
        network.verbose = 0

    x = check_tensor(x, device=network.device)

    _, u, _ = init_vuz(network.A, network.D, x, lbda, inv_A=network.inv_A_,
                       device='cpu')

    l_diff_loss = []
    for layer_params in network.layers_parameters:
        # retrieve parameters
        mul_lbda = layer_params.get('threshold', 1.0 / network.l_)
        Wx = layer_params['Wx']
        Wu = layer_params['Wu']
        # apply one 'iteration'
        u = u.matmul(Wu) + x.matmul(Wx)
        # approx prox-tv
        approx_prox_z = network.prox_tv(x=u, lbda=lbda * mul_lbda)
        approx_prox_u = torch.cumsum(approx_prox_z, dim=1)
        approx_prox_u_npy = approx_prox_u.detach().numpy()
        # true prox-tv
        u_npy = u.detach().numpy()
        lbda_npy = float(lbda * mul_lbda)
        prox_u_npy = np.array([prox_tv.tv1_1d(u_, lbda_npy) for u_ in u_npy])
        # quantify sub-optimality of the approx-prox
        diff_loss = (tv_loss(u_npy, approx_prox_u_npy, lbda_npy) -  # approx
            tv_loss(u_npy, prox_u_npy, lbda_npy)                    # true
            )
        l_diff_loss.append(diff_loss)
        # store feasible point for next iteration
        u = approx_prox_u

    return l_diff_loss

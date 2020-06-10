""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import warnings
import torch
import numpy as np
from prox_tv import tv1_1d

from .utils import init_vuz
from .checks import check_tensor
from .lista_analysis import ListaTV
from .lista_analysis import LEARN_PROX_PER_LAYER
from .loss_gradient import loss_prox_tv_analysis


def compute_prox_tv_errors(network, x, lbda):
    """Return the sub-optimality gap of the prox-tv at each iteration.
    """

    if not isinstance(network, ListaTV):
        raise ValueError("network should be of type {'ListaTV'}.")

    if not hasattr(network, 'training_loss_'):
        warnings.warn("network has not been trained before computing "
                      "prox_tv_errors.")

    x = check_tensor(x, device=network.device)

    _, u, _ = init_vuz(network.A, network.D, x, lbda, inv_A=network.inv_A_,
                       device=network.device)

    l_diff_loss = []
    for layer_id in range(network.n_layers):
        layer_params = network.parameter_groups[f'layer-{layer_id}']

        # retrieve parameters
        Wx = layer_params['Wx']
        Wu = layer_params['Wu']

        # Get the correct prox depending on the layer_id and learn_prox
        mul_lbda = layer_params.get('threshold', 1.0 / network.l_)
        mul_lbda = max(0, mul_lbda)
        if network.learn_prox == LEARN_PROX_PER_LAYER:
            prox_tv = network.prox_tv[layer_id]
        else:
            prox_tv = network.prox_tv

        # apply one 'iteration'
        u_half = u.matmul(Wu) + x.matmul(Wx)
        u_half_npy = u_half.detach().cpu().numpy()

        # prox-tv as applied by the network
        z_k = prox_tv(u_half, lbda * mul_lbda)
        u = torch.cumsum(z_k, dim=1)
        approx_prox_u_npy = u.detach().cpu().numpy()

        # exact prox-tv with taut-string algorithm
        lbda_npy = float(lbda * mul_lbda)
        prox_u_npy = np.array([tv1_1d(u_, lbda_npy)
                               for u_ in u_half_npy])

        # log sub-optimality of the prox applied by the network
        diff_loss = (
            loss_prox_tv_analysis(u_half_npy, approx_prox_u_npy, lbda_npy) -
            loss_prox_tv_analysis(u_half_npy, prox_u_npy, lbda_npy)
        )
        l_diff_loss.append(diff_loss)

    return l_diff_loss

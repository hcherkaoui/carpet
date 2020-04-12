""" Unittest module for proximal operator. """
import pytest
import numpy as np
from carpet.lista_synthesis import ALL_LISTA
from carpet.checks import check_random_state, check_tensor
from carpet.datasets import synthetic_1d_dataset
from carpet.loss_gradient import synthesis_obj


@pytest.mark.parametrize('seed', [None])
@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('parametrization', ['lista', 'coupled', 'step'])
def test_init_parameters(seed, m, n, lbda, parametrization):
    """ Test the gradient of z. """
    rng = check_random_state(seed)
    n_layers = 5

    L = np.triu(np.ones((m, )))
    x, _, _ = synthetic_1d_dataset(D=L, n=10, s=0.4, snr=1.0, seed=rng)
    x /= np.max(np.abs(x.dot(L.T)), axis=1, keepdims=True)

    z0_tensor = check_tensor(np.zeros_like(x), 'cpu')
    x_tensor = check_tensor(x, 'cpu')

    lista_1 = ALL_LISTA[parametrization](D=L, n_layers=n_layers, max_iter=10,
                                         device='cpu', verbose=0)
    lista_1.fit(x_tensor, lbda=lbda)
    parameters = lista_1.export_parameters()

    loss_lista_1 = []
    for n_layer_ in range(n_layers + 1):
        z_hat_1 = lista_1.transform(x=x_tensor, lbda=lbda, z0=z0_tensor,
                                    output_layer=n_layer_)
        loss_lista_1.append(synthesis_obj(z=z_hat_1, L=L, x=x, lbda=lbda))
    loss_lista_1 = np.array(loss_lista_1)

    lista_2 = ALL_LISTA[parametrization](
                                D=L, n_layers=n_layers, max_iter=10,
                                device='cpu', initial_parameters=parameters,
                                verbose=0
                                )

    loss_lista_2 = []
    for n_layer_ in range(n_layers + 1):
        z_hat_2 = lista_2.transform(x=x_tensor, lbda=lbda, z0=z0_tensor,
                                    output_layer=n_layer_)
        loss_lista_2.append(synthesis_obj(z=z_hat_2, L=L, x=x, lbda=lbda))
    loss_lista_2 = np.array(loss_lista_2)

    np.testing.assert_allclose(z_hat_1, z_hat_2)
    np.testing.assert_allclose(loss_lista_1, loss_lista_2)

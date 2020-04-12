""" Unittests module for untrained LISTA. """
import pytest
import numpy as np
from carpet.lista_synthesis import ALL_LISTA
from carpet.checks import check_random_state, check_tensor
from carpet.proximity import pseudo_soft_th_numpy
from carpet.datasets import synthetic_1d_dataset
from carpet.optimization import fista
from carpet.loss_gradient import synthesis_grad, synthesis_obj


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('parametrization', ['lista', 'coupled', 'step'])
def test_untrained_lista(lbda, parametrization, n, m):
    """ Test the gradient of z. """
    rng = check_random_state(None)
    L = np.triu(np.ones((m, m)))
    x, _, _ = synthetic_1d_dataset(D=L, n=n, s=0.5, snr=1.0, seed=rng)

    n_layers = 10
    step_size = 1.0 / np.linalg.norm(L.T.dot(L), ord=2)
    z0 = x
    z0_ = check_tensor(x, 'cpu')

    lista = ALL_LISTA[parametrization](D=L, n_layers=n_layers, device='cpu')
    z_hat_untrained_lista = lista.transform(x=x, lbda=lbda, z0=z0_,
                                            output_layer=n_layers)
    loss_untrained_lista = [synthesis_obj(z=z0, L=L, x=x, lbda=lbda)]
    for n_layer_ in range(1, n_layers + 1):
        z_hat = lista.transform(x=x, lbda=lbda, z0=z0_, output_layer=n_layer_)
        loss_untrained_lista.append(synthesis_obj(z=z_hat, L=L, x=x,
                                                  lbda=lbda))
    loss_untrained_lista = np.array(loss_untrained_lista)

    params = dict(grad=lambda z: synthesis_grad(z, L, x),
                  obj=lambda z: synthesis_obj(z, L, x, lbda),
                  prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                  x0=z0, momentum=None, restarting=None, max_iter=n_layers,
                  step_size=step_size, early_stopping=False, debug=True,
                  verbose=0,
                  )
    z_hat_ista, loss_ista = fista(**params)

    np.testing.assert_allclose(z_hat_ista, z_hat_untrained_lista, rtol=1e-4,
                               atol=1e-3)
    np.testing.assert_allclose(loss_ista, loss_untrained_lista, rtol=1e-4,
                               atol=1e-3)

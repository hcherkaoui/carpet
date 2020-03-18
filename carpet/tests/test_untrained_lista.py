""" Unittests module for untrained LISTA. """
import pytest
import numpy as np
from carpet.lista import Lista
from carpet.checks import check_random_state, check_tensor
from carpet.proximity import soft_thresholding
from carpet.datasets import synthetic_1d_dataset
from carpet.optimization import fista
from carpet.synthesis_loss_gradient import grad, obj


@pytest.mark.parametrize('seed', [None])
@pytest.mark.parametrize('lbda_ratio', [0.0, 0.9])
@pytest.mark.parametrize('n', [1, 10, 50])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('parametrization', ['lista', 'coupled', 'hessian',
                                             'step'])
def test_untrained_lista(seed, lbda_ratio, parametrization, n, m):
    """ Test the gradient of z. """
    rng = check_random_state(seed)

    x, _, _ = synthetic_1d_dataset(n=n, m=m, s=0.1, snr=1.0, seed=rng)

    _, m = x.shape
    D = np.triu(np.ones((m, )))

    Dtx = x.dot(D.T)
    lbda_max = np.max(Dtx)
    lbda = lbda_ratio * lbda_max

    max_iter = 10
    lipsc = np.linalg.norm(D.T.dot(D), ord=2)
    step_size = 1.0 / lipsc

    z0 = np.zeros_like(x)
    z0_ = check_tensor(z0, 'cpu')

    lista = Lista(D=D, n_layers=max_iter, parametrization=parametrization,
                  max_iter=100, device='cpu', verbose=0)
    z_hat_untrained_lista = lista.transform(x=x, lmbd=lbda, z0=z0_,
                                            output_layer=max_iter)
    loss_untrained_lista = []
    for n_layer_ in range(max_iter + 1):
        z_hat = lista.transform(x=x, lmbd=lbda, z0=z0_, output_layer=n_layer_)
        loss_untrained_lista.append(obj(z=z_hat, D=D, x=x, lbda=lbda))
    loss_untrained_lista = np.array(loss_untrained_lista)

    def _obj(z):
        return obj(z, D, x, lbda=lbda)

    def _grad(z):
        return grad(z, D, x)

    def _prox(z, step_size):
        return soft_thresholding(z, lbda, step_size)

    params = dict(grad=_grad, obj=_obj, prox=_prox, x0=z0, momentum='ista',
                  restarting=None, max_iter=max_iter, step_size=step_size,
                  early_stopping=False, debug=True, verbose=2,
                  )
    z_hat_ista, loss_ista = fista(**params)

    np.testing.assert_allclose(z_hat_ista, z_hat_untrained_lista, rtol=1e-5,
                               atol=1e-3)
    np.testing.assert_allclose(loss_ista, loss_untrained_lista, rtol=1e-5,
                               atol=1e-3)

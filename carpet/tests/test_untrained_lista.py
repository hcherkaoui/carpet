""" Unittests module for untrained LISTA. """
import pytest
import numpy as np
from carpet import LearnTVAlgo
from carpet.checks import check_random_state
from carpet.utils import init_vuz
from carpet.proximity import pseudo_soft_th_numpy
from carpet.datasets import synthetic_1d_dataset
from carpet.optimization import fista, condatvu
from carpet.loss_gradient import (synthesis_primal_grad, synthesis_primal_obj,
                                  analysis_primal_grad, analysis_primal_obj)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('parametrization', ['origista', 'coupledista',
                                             'stepista'])
def test_untrained_synthesis_lista(lbda, parametrization, n):
    """ Test the gradient of z. """
    rng = check_random_state(None)
    x, _, _, L, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    _, _, z0 = init_vuz(A, D, x, lbda)

    n_layers = 10
    LA = L.dot(A)
    step_size = 1.0 / np.linalg.norm(LA.dot(LA.T), ord=2)

    lista = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=n_layers,
                        device='cpu')
    loss_untrained_lista = [synthesis_primal_obj(z=z0, A=A, L=L, x=x,
                                                 lbda=lbda)]
    for n_layer_ in range(1, n_layers + 1):
        z_hat = lista.transform(x=x, lbda=lbda, output_layer=n_layer_)
        loss_untrained_lista.append(synthesis_primal_obj(z=z_hat, A=A, L=L,
                                                         x=x, lbda=lbda))
    loss_untrained_lista = np.array(loss_untrained_lista)

    params = dict(grad=lambda z: synthesis_primal_grad(z, A, L, x),
                  obj=lambda z: synthesis_primal_obj(z, A, L, x, lbda),
                  prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                  x0=z0, momentum=None, restarting=None, max_iter=n_layers,
                  step_size=step_size, early_stopping=False, debug=True,
                  verbose=0,
                  )
    _, _, loss_ista = fista(**params)

    np.testing.assert_allclose(loss_ista, loss_untrained_lista, atol=1e-20)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('parametrization', ['coupledcondatvu',
                                             'stepcondatvu'])
def test_untrained_analysis_lista(lbda, parametrization, n):
    """ Test the gradient of z. """
    rng = check_random_state(None)
    x, _, _, _, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    v0, u0, _ = init_vuz(A, D, x, lbda)

    n_layers = 10
    rho = 1.0
    sigma = 0.5
    L_D = np.linalg.norm(D.dot(D.T), ord=2)
    L_A = np.linalg.norm(A.dot(A.T), ord=2)
    tau = 1.0 / (L_A / 2.0 + sigma * L_D**2)

    lista = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=n_layers,
                        device='cpu')
    loss_untrained_condat = [analysis_primal_obj(u0, A, D, x, lbda)]
    for n_layer_ in range(1, n_layers + 1):
        z = lista.transform(x=x, lbda=lbda, output_layer=n_layer_)
        loss_untrained_condat.append(analysis_primal_obj(z, A, D, x, lbda))
    loss_untrained_condat = np.array(loss_untrained_condat)

    params = dict(
             grad=lambda u: analysis_primal_grad(u, A, x),
             obj=lambda u: analysis_primal_obj(u, A, D, x, lbda),
             prox=lambda z: pseudo_soft_th_numpy(z, lbda, 1.0 / sigma),
             psi=lambda u: u.dot(D), adj_psi=lambda v: v.dot(D.T),
             v0=v0, z0=u0, lbda=lbda, sigma=sigma, tau=tau, rho=rho,
             max_iter=n_layers, early_stopping=False, debug=True, verbose=0,
             )
    _, _, _, _, loss_condat = condatvu(**params)

    np.testing.assert_allclose(loss_condat, loss_untrained_condat, atol=1e-20)

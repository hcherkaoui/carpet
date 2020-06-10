""" Unittest module for proximal operator. """
import pytest
import numpy as np
from carpet import LearnTVAlgo
from carpet.utils import init_vuz, v_to_u
from carpet.checks import check_random_state
from carpet.datasets import synthetic_1d_dataset
from carpet.loss_gradient import synthesis_primal_obj, analysis_primal_obj


@pytest.mark.parametrize('seed', [None])
@pytest.mark.parametrize('lbda', [0.0, 0.5])
def test_coherence_init(lbda, seed):
    rng = check_random_state(seed)
    x, _, _, L, D, A = synthetic_1d_dataset()

    v0 = None
    v0, u0, z0 = init_vuz(A, D, x, v0=v0)
    cost_1 = synthesis_primal_obj(z0, A, L, x, lbda)
    cost_2 = analysis_primal_obj(u0, A, D, x, lbda)
    cost_3 = analysis_primal_obj(v_to_u(v0, x, A, D), A, D, x, lbda)

    np.testing.assert_allclose(cost_1, cost_2)
    np.testing.assert_allclose(cost_1, cost_3)

    v0 = rng.randn(*v0.shape)
    v0, u0, z0 = init_vuz(A, D, x, v0=v0)
    synthesis_primal_obj(z0, A, L, x, lbda)
    cost_1 = synthesis_primal_obj(z0, A, L, x, lbda)
    cost_2 = analysis_primal_obj(u0, A, D, x, lbda)
    cost_3 = analysis_primal_obj(v_to_u(v0, x, A, D), A, D, x, lbda)

    np.testing.assert_allclose(cost_1, cost_2)
    np.testing.assert_allclose(cost_1, cost_3)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [3, 50])
@pytest.mark.parametrize('parametrization', ['origista', 'coupledista',
                                             'stepista', 'origtv'])
def test_init_parameters(n, lbda, parametrization):
    """ Test the gradient of z. """
    rng = check_random_state(27)
    x, _, _, L, _, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    n_layers = 5

    # limit the number of inner layers for origtv to avoid long computations
    kwargs = {}
    if parametrization == 'origtv':
        kwargs['n_inner_layers'] = 5

    lista_1 = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=n_layers,
                          max_iter=10, net_solver_type='one_shot', **kwargs)
    lista_1.fit(x, lbda=lbda)
    params = lista_1.export_parameters()

    loss_lista_1 = []
    for n_layer_ in range(n_layers + 1):
        z_1 = lista_1.transform(x=x, lbda=lbda, output_layer=n_layer_)
        loss_lista_1.append(synthesis_primal_obj(z=z_1, A=A, L=L, x=x,
                                                 lbda=lbda))
    loss_lista_1 = np.array(loss_lista_1)

    lista_2 = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=n_layers,
                          initial_parameters=params, max_iter=10, **kwargs)

    loss_lista_2 = []
    for n_layer_ in range(n_layers + 1):
        z_2 = lista_2.transform(x=x, lbda=lbda, output_layer=n_layer_)
        loss_lista_2.append(synthesis_primal_obj(z=z_2, A=A, L=L, x=x,
                                                 lbda=lbda))
    loss_lista_2 = np.array(loss_lista_2)

    np.testing.assert_allclose(z_1, z_2)
    np.testing.assert_allclose(loss_lista_1, loss_lista_2)

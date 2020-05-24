import pytest
import numpy as np


from carpet import ListaTV
from carpet.checks import check_tensor
from carpet.lista_analysis import ALL_LEARN_PROX
from carpet.lista_analysis import LEARN_PROX_FALSE, LEARN_PROX_GLOBAL


@pytest.mark.parametrize('net_solver_type',
                         ['one_shot', 'recursive', 'greedy'])
def test_learn_prox_training(net_solver_type):

    n_atoms = 2
    n_samples, n_dims = 5, 3
    n_layers = 4
    lbda = .2
    max_iter = 10

    x = np.random.randn(n_samples, n_dims)
    A = np.random.randn(n_atoms, n_dims)
    z0 = np.random.randn(n_samples, n_atoms)

    network = ListaTV(A, n_layers, learn_prox=LEARN_PROX_FALSE,
                      n_inner_layers=5,
                      max_iter=max_iter, net_solver_type=net_solver_type)

    z_untrain = network.prox_tv.transform(z0, lbda)
    network.fit(x, lbda)

    z_train = network.prox_tv.transform(z0, lbda)
    assert np.allclose(z_untrain, z_train)

    network = ListaTV(A, n_layers, learn_prox=LEARN_PROX_GLOBAL,
                      n_inner_layers=5,
                      max_iter=max_iter, net_solver_type=net_solver_type)

    z_untrain = network.prox_tv.transform(z0, lbda)
    network.fit(x, lbda)

    z_train = network.prox_tv.transform(z0, lbda)
    assert not np.allclose(z_untrain, z_train)


@pytest.mark.parametrize('learn_prox', ALL_LEARN_PROX)
def test_learn_prox_init(learn_prox):

    n_atoms = 2
    n_samples, n_dims = 5, 3
    n_layers = 4
    lbda = .2
    max_iter = 10

    x = np.random.randn(n_samples, n_dims)
    A = np.random.randn(n_atoms, n_dims)

    network = ListaTV(A, n_layers, learn_prox=learn_prox,
                      n_inner_layers=5, max_iter=max_iter)
    z_untrain = network.transform(x, lbda)
    network.fit(x, lbda)
    z_train = network.transform(x, lbda)

    assert not np.allclose(z_train, z_untrain)
    params = network.export_parameters()

    network_clone = ListaTV(A, n_layers, learn_prox=learn_prox,
                            n_inner_layers=5, initial_parameters=params)

    z_clone = network_clone.transform(x, lbda)

    assert np.allclose(z_clone, z_train)


@pytest.mark.parametrize('learn_prox', ALL_LEARN_PROX)
def test_learn_prox_gradient(learn_prox):
    # Check that the parameter_groups' parameters are used correctly.

    n_atoms = 2
    n_samples, n_dims = 5, 3
    n_layers = 4
    lbda = .2
    max_iter = 10

    x = check_tensor(np.random.randn(n_samples, n_dims))
    A = np.random.randn(n_atoms, n_dims)

    network = ListaTV(A, n_layers, learn_prox=learn_prox,
                      n_inner_layers=5, max_iter=max_iter)
    network._compute_gradient(x, lbda, layer_id=n_layers)
    for group_name, group_params in network.parameter_groups.items():
        for name, p in group_params.items():
            assert p.requires_grad
            assert p.grad is not None, (
                f"{group_name}:{name} gradient is null"
            )

    # check that initial_parameters are registered correctly
    params = network.export_parameters()

    network_clone = ListaTV(A, n_layers, learn_prox=learn_prox,
                            n_inner_layers=5, initial_parameters=params)
    network_clone._compute_gradient(x, lbda, layer_id=n_layers)
    for group_name, group_params in network_clone.parameter_groups.items():
        for name, p in group_params.items():
            if group_name != 'layer-0' or name != 'Wu':
                assert p.requires_grad
                assert p.grad is not None, (
                    f"{group_name}:{name} gradient is null"
                )

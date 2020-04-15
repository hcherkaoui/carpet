""" Unittest module for gradient. """
import pytest
import numpy as np
from scipy.optimize import approx_fprime
from carpet.lista_synthesis import ALL_LISTA
from carpet.lista_analysis import ALL_LTV
from carpet.checks import check_random_state, check_tensor
from carpet.loss_gradient import (analysis_obj, analysis_grad,
                                  analysis_subgrad, synthesis_obj,
                                  synthesis_grad, synthesis_subgrad)
from carpet.datasets import synthetic_1d_dataset


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n', [1, 20])
def test_coherence_analysis_synthesis_loss(lbda, m, n):
    """ Test coherence regarding the loss function between analysis and
    synthesis loss. """
    rng = check_random_state(None)

    L = np.triu(np.ones((m, m)))
    D = (np.eye(m, k=-1) - np.eye(m, k=0))[:, :-1]
    x, Lz, z = synthetic_1d_dataset(D=L, n=n, s=0.5, snr=0.0, seed=rng)

    analysis_loss = analysis_obj(Lz, D, x, lbda)
    synthesis_loss = synthesis_obj(z, L, x, lbda)

    np.testing.assert_allclose(analysis_loss, synthesis_loss, atol=1e-30)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n', [1, 20])
@pytest.mark.parametrize('parametrization', ['lista', 'coupled', 'step'])
def test_coherence_synthesis_loss(parametrization, lbda, n, m):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)

    L = np.triu(np.ones((m, m)))
    x, _, _ = synthetic_1d_dataset(D=L, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)
    z_ = check_tensor(z, device='cpu')

    cost = synthesis_obj(z, L, x, lbda)
    lista = ALL_LISTA[parametrization](D=L, n_layers=10, device='cpu')
    cost_ref = lista._loss_fn(x, lbda, z_)

    np.testing.assert_allclose(cost_ref, cost, atol=1e-30)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('parametrization', ['stepsubgradient',
                                             'coupledcondatvu',
                                             'stepcondatvu'])
def test_coherence_analysis_loss(parametrization, lbda, n, m):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, )))
    x, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)
    z_ = check_tensor(z, device='cpu')

    n, m = x.shape
    D = np.triu(np.ones((m, )))

    cost = analysis_obj(z, D, x, lbda=lbda)
    ltv = ALL_LTV[parametrization](D=D, n_layers=10, device='cpu')
    cost_ref = ltv._loss_fn(x, lbda=lbda, z_hat=z_)

    np.testing.assert_allclose(cost_ref, cost, atol=1e-30)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n', [1, 20])
@pytest.mark.parametrize('parametrization', ['lista', 'coupled', 'step'])
def test_coherence_training_synthesis_loss(parametrization, lbda, m, n):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)

    L = np.triu(np.ones((m, m)))
    D = (np.eye(m, k=-1) - np.eye(m, k=0))[:, :-1]
    x, _, _ = synthetic_1d_dataset(D=L, n=n, s=0.5, snr=0.0, seed=rng)

    z0 = np.c_[x[:, 0], x.dot(D)]  # init don't matter here
    train_loss = [synthesis_obj(z0, L, x, lbda)]
    train_loss_ = [synthesis_obj(z0, L, x, lbda)]

    for n_layers in range(1, 10):
        algo = ALL_LISTA[parametrization](D=L, n_layers=n_layers, max_iter=10)
        algo.fit(x, lbda=lbda)
        train_loss_.append(algo.training_loss_[-1])
        z_hat = algo.transform(x, lbda, output_layer=n_layers)
        train_loss.append(synthesis_obj(z_hat, L, x, lbda))

    np.testing.assert_allclose(train_loss_, train_loss, atol=1e-30)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n', [1, 20])
@pytest.mark.parametrize('parametrization', ['stepsubgradient',
                                             'coupledcondatvu',
                                             'stepcondatvu'])
def test_coherence_training_analysis_loss(parametrization, lbda, m, n):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)

    L = np.triu(np.ones((m, m)))
    D = (np.eye(m, k=-1) - np.eye(m, k=0))[:, :-1]
    x_train, _, _ = synthetic_1d_dataset(D=L, n=n, s=0.5, snr=0.0, seed=rng)

    z0_train = np.zeros_like(x_train)  # init don't matter here
    train_loss = [analysis_obj(z0_train, D, x_train, lbda)]
    train_loss_ = [analysis_obj(z0_train, D, x_train, lbda)]

    for n_layers in range(1, 10):
        lista = ALL_LTV[parametrization](D=D, n_layers=n_layers, max_iter=10)
        lista.fit(x_train, lbda=lbda)
        train_loss_.append(lista.training_loss_[-1])
        z_train = lista.transform(x_train, lbda, output_layer=n_layers)
        train_loss.append(analysis_obj(z_train, D, x_train, lbda))

    np.testing.assert_allclose(train_loss_, train_loss, atol=1e-30)


@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [2, 5])
def test_synthesis_grad(n, m):
    """ Test the gradient of LASSO. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, m)))
    x, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)

    # Finite grad z
    def finite_grad(z):
        def f(z):
            z = z.reshape(n, m)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return synthesis_obj(z, D, x, lbda=0.0) * n
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = synthesis_grad(z, D, x)

    np.testing.assert_allclose(grad_ref, grad_test, rtol=5e-2)  # bad precision


@pytest.mark.parametrize('n', [1, 5])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('lbda', [0.1, 0.5])
def test_synthesis_subgrad(n, m, lbda):
    """ Test the sub-gradient of LASSO. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, m)))
    x, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)

    # Finite grad z
    def finite_grad(z):
        def f(z):
            z = z.reshape(n, m)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return synthesis_obj(z, D, x, lbda=lbda) * n
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = synthesis_subgrad(z, D, x, lbda)

    np.testing.assert_allclose(grad_ref, grad_test, atol=1e-5)  # bad precision


@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [10, 20])
def test_analysis_grad(n, m):
    """ Test the gradient of LASSO. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, )))
    x, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)
    D = np.triu(np.ones((m, )))

    # Finite grad z
    def finite_grad(z):
        def f(z):
            z = z.reshape(n, m)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return analysis_obj(z, D, x, lbda=0.0) * n
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = analysis_grad(z, x)

    np.testing.assert_allclose(grad_ref, grad_test, atol=1e-5)  # bad precision


@pytest.mark.parametrize('n', [1, 10, 100])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('lbda', [0.1, 0.5])
def test_analysis_subgrad(n, m, lbda):
    """ Test the sub-gradient of LASSO. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, )))
    x, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)
    D = np.triu(np.ones((m, )))

    # Finite grad z
    def finite_grad(z):
        def f(z):
            z = z.reshape(n, m)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return analysis_obj(z, D, x, lbda=lbda) * n
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1.0e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = analysis_subgrad(z, D, x, lbda)

    np.testing.assert_allclose(grad_ref, grad_test, atol=1e-5)  # bad precision

""" Unittest module for gradient. """
import pytest
import numpy as np
from scipy.optimize import approx_fprime
from carpet import LearnTVAlgo
from carpet.utils import init_vuz
from carpet.checks import check_random_state, check_tensor
from carpet.loss_gradient import (analysis_primal_obj, analysis_primal_grad,
                                  analysis_primal_subgrad,
                                  synthesis_primal_obj,
                                  synthesis_primal_grad,
                                  synthesis_primal_subgrad,
                                  analysis_dual_obj, analysis_dual_grad)
from carpet.datasets import synthetic_1d_dataset


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 20])
@pytest.mark.parametrize('parametrization', ['origista', 'coupledista',
                                             'stepista'])
def test_coherence_synthesis_loss(parametrization, lbda, n):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)
    x, _, _, L, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    _, _, z = init_vuz(A, D, x, lbda)
    z_ = check_tensor(z, device='cpu')

    cost = synthesis_primal_obj(z, A, L, x, lbda)
    lista = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=10,
                        device='cpu')
    cost_ref = lista._loss_fn(x, lbda, z_)

    np.testing.assert_allclose(cost_ref, cost, atol=1e-30)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('parametrization', ['stepsubgradient',
                                             'coupledcondatvu',
                                             'stepcondatvu'])
def test_coherence_analysis_loss(parametrization, lbda, n):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)
    x, _, _, _, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    _, _, z = init_vuz(A, D, x, lbda)
    z_ = check_tensor(z, device='cpu')

    cost = analysis_primal_obj(z, A, D, x, lbda=lbda)
    ltv = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=10,
                      device='cpu')
    cost_ref = ltv._loss_fn(x, lbda, z_)

    np.testing.assert_allclose(cost_ref, cost, atol=1e-30)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 20])
@pytest.mark.parametrize('parametrization', ['origista', 'coupledista',
                                             'stepista'])
def test_coherence_training_synthesis_loss(parametrization, lbda, n):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)
    x, _, _, L, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)

    _, _, z0 = init_vuz(A, D, x, lbda)
    train_loss = [synthesis_primal_obj(z0, A, L, x, lbda)]
    train_loss_ = [synthesis_primal_obj(z0, A, L, x, lbda)]

    for n_layers in range(1, 10):
        algo = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=n_layers,
                           max_iter=10)
        algo.fit(x, lbda=lbda)
        train_loss_.append(algo.training_loss_[-1])
        z_hat = algo.transform(x, lbda, output_layer=n_layers)
        train_loss.append(synthesis_primal_obj(z_hat, A, L, x, lbda))

    np.testing.assert_allclose(train_loss_, train_loss, atol=1e-30)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 20])
@pytest.mark.parametrize('parametrization', ['stepsubgradient',
                                             'coupledcondatvu',
                                             'stepcondatvu'])
def test_coherence_training_analysis_loss(parametrization, lbda, n):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)
    x, _, _, _, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)

    _, u0, _ = init_vuz(A, D, x, lbda)
    train_loss = [analysis_primal_obj(u0, A, D, x, lbda)]
    train_loss_ = [analysis_primal_obj(u0, A, D, x, lbda)]

    for n_layers in range(1, 10):
        lista = LearnTVAlgo(algo_type=parametrization, A=A, n_layers=n_layers,
                            max_iter=10)
        lista.fit(x, lbda=lbda)
        train_loss_.append(lista.training_loss_[-1])
        u = lista.transform(x, lbda, output_layer=n_layers)
        train_loss.append(analysis_primal_obj(u, A, D, x, lbda))

    np.testing.assert_allclose(train_loss_, train_loss, atol=1e-30)


@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [2, 5])
def test_synthesis_grad(n, m):
    """ Test the gradient of LASSO. """
    rng = check_random_state(None)
    x, _, z, L, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.rand(*z.shape)
    n_atoms = D.shape[0]

    def finite_grad(z):
        def f(z):
            z = z.reshape(n, n_atoms)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return synthesis_primal_obj(z, A, L, x, lbda=0.0) * n
        grad = approx_fprime(xk=z.ravel(), f=f, epsilon=1e-6)
        return grad.reshape(n, n_atoms)

    grad_ref = finite_grad(z)
    grad_test = synthesis_primal_grad(z, A, L, x)

    np.testing.assert_allclose(grad_ref, grad_test, rtol=5e-2)  # bad precision


@pytest.mark.parametrize('n', [1, 5])
@pytest.mark.parametrize('lbda', [0.1, 0.5])
def test_synthesis_subgrad(n, lbda):
    """ Test the sub-gradient of LASSO. """
    rng = check_random_state(None)
    x, _, z, L, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.rand(*z.shape)
    n_atoms = D.shape[0]

    def finite_grad(z):
        def f(z):
            z = z.reshape(n, n_atoms)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return synthesis_primal_obj(z, A, L, x, lbda=lbda) * n
        grad = approx_fprime(xk=z.ravel(), f=f, epsilon=1e-6)
        return grad.reshape(n, n_atoms)

    grad_ref = finite_grad(z)
    grad_test = synthesis_primal_subgrad(z, A, L, x, lbda)

    np.testing.assert_allclose(grad_ref, grad_test, atol=1e-5)  # bad precision


@pytest.mark.parametrize('n', [1, 50])
def test_analysis_grad(n):
    """ Test the gradient of LASSO. """
    rng = check_random_state(None)
    x, u, _, _, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    u = rng.rand(*u.shape)
    n_atoms = D.shape[0]

    def finite_grad(u):
        def f(u):
            u = u.reshape(n, n_atoms)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return analysis_primal_obj(u, A, D, x, lbda=0.0) * n
        grad = approx_fprime(xk=u.ravel(), f=f, epsilon=1e-6)
        return grad.reshape(n, n_atoms)

    grad_ref = finite_grad(u)
    grad_test = analysis_primal_grad(u, A, x)

    np.testing.assert_allclose(grad_ref, grad_test, atol=1e-5)  # bad precision


@pytest.mark.parametrize('n', [1, 10, 100])
@pytest.mark.parametrize('lbda', [0.1, 0.5])
def test_analysis_subgrad(n, lbda):
    """ Test the sub-gradient of LASSO. """
    rng = check_random_state(None)
    x, u, _, _, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    u = rng.rand(*u.shape)
    n_atoms = D.shape[0]

    def finite_grad(u):
        def f(u):
            u = u.reshape(n, n_atoms)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return analysis_primal_obj(u, A, D, x, lbda=lbda) * n
        grad = approx_fprime(xk=u.ravel(), f=f, epsilon=1.0e-6)
        return grad.reshape(n, n_atoms)

    grad_ref = finite_grad(u)
    grad_test = analysis_primal_subgrad(u, A, D, x, lbda)

    np.testing.assert_allclose(grad_ref, grad_test, atol=1e-5)  # bad precision


@pytest.mark.parametrize('n', [1, 10, 100])
@pytest.mark.parametrize('lbda', [0.1, 0.5])
def test_analysis_dual_grad(n, lbda):
    """ Test the gradient of dual analysis. """
    rng = check_random_state(None)
    x, _, _, _, D, A = synthetic_1d_dataset(n=n, s=0.5, snr=0.0, seed=rng)
    eps = 1e-3
    v_dim = D.shape[1]
    v = np.clip(rng.randn(n, v_dim), -(lbda - eps), (lbda - eps))
    Psi_A = np.linalg.pinv(A).dot(D)

    # Finite grad v
    def finite_grad(v):
        def f(v):
            v = v.reshape(n, v_dim)
            # the actual considered loss is not normalized but for
            # convenience we want to check the sample-loss average
            return analysis_dual_obj(v, A, D, x, lbda, Psi_A=Psi_A) * n
        grad = approx_fprime(xk=v.ravel(), f=f, epsilon=1.0e-6)
        return grad.reshape(n, v_dim)

    grad_ref = finite_grad(v)
    grad_test = analysis_dual_grad(v, A, D, x, lbda, Psi_A=Psi_A)

    np.testing.assert_allclose(grad_ref, grad_test, atol=1e-5)  # bad precision

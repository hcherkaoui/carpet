""" Unittest module for gradient. """
import pytest
import numpy as np
from scipy.optimize import approx_fprime
from carpet.lista import Lista
from carpet.checks import check_random_state, check_tensor
from carpet.synthesis_loss_gradient import obj, grad, subgrad
from carpet.datasets import synthetic_1d_dataset


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [10, 20])
def test_loss(lbda, n, m):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, )))
    x, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)
    z_ = check_tensor(z, device='cpu')

    n, m = x.shape
    D = np.triu(np.ones((m, )))

    cost = obj(z, D, x, lbda=lbda)
    lista = Lista(D=D, n_layers=10, device='cpu')
    cost_ref = lista._loss_fn(x, lmbd=lbda, z_hat=z_)

    np.testing.assert_allclose(cost_ref, cost, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [10, 20])
def test_grad(n, m):
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
            return obj(z, D, x, lbda=0.0)
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1.0e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = grad(z, D, x)

    np.testing.assert_allclose(grad_ref, grad_test, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('n', [1, 10, 100])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('lbda', [0.1, 0.5])
def test_subgrad(n, m, lbda):
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
            return obj(z, D, x, lbda=lbda)
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1.0e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = subgrad(z, D, x, lbda)

    np.testing.assert_allclose(grad_ref, grad_test, rtol=1e-5, atol=1e-3)

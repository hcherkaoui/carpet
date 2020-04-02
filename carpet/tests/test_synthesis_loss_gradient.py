""" Unittest module for gradient. """
import pytest
import numpy as np
from scipy.optimize import approx_fprime
from carpet.lista import ALL_LISTA
from carpet.checks import check_random_state, check_tensor
from carpet.synthesis_loss_gradient import obj, grad, subgrad
from carpet.datasets import synthetic_1d_dataset


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('parametrization', ['lista', 'coupled', 'step'])
def test_coherence_training_loss(parametrization, lbda, m, n=10):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, m)))
    x_train, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)

    z0_train = np.zeros_like(x_train.dot(D.T))
    train_loss = [obj(z0_train, D, x_train, lbda)]
    train_loss_ = [obj(z0_train, D, x_train, lbda)]
    all_n_layers = range(1, 10)

    for n_layers in all_n_layers:
        lista = ALL_LISTA[parametrization](
                        D=D, n_layers=n_layers, max_iter=10,
                        device='cpu', name=parametrization,
                        per_layer='one_shot', verbose=10)
        lista.fit(x_train, lbda=lbda)
        train_loss_.append(lista.training_loss_[-1])
        z_train = lista.transform(x_train, lbda, output_layer=n_layers)
        train_loss.append(obj(z_train, D, x_train, lbda))

    np.testing.assert_allclose(train_loss_, train_loss)


@pytest.mark.parametrize('lbda', [0.0, 0.5])
@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('parametrization', ['lista', 'coupled', 'step'])
def test_loss(parametrization, lbda, n, m):
    """ Test coherence regarding the loss function between learnt and fixed
    algorithms. """
    rng = check_random_state(None)

    D = np.triu(np.ones((m, m)))
    x, _, _ = synthetic_1d_dataset(D=D, n=n, s=0.5, snr=0.0, seed=rng)
    z = rng.randn(*x.shape)
    z_ = check_tensor(z, device='cpu')

    cost = obj(z, D, x, lbda=lbda)
    lista = ALL_LISTA[parametrization](D=D, n_layers=10, device='cpu')
    cost_ref = lista._loss_fn(x, lbda=lbda, z_hat=z_)

    np.testing.assert_allclose(cost_ref, cost)


@pytest.mark.parametrize('n', [1, 50])
@pytest.mark.parametrize('m', [5, 10])
def test_grad(n, m):
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
            return obj(z, D, x, lbda=0.0) * n
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1.0e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = grad(z, D, x)

    np.testing.assert_allclose(grad_ref, grad_test, rtol=5e-2)  # bad precision


@pytest.mark.parametrize('n', [1, 10, 100])
@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('lbda', [0.1, 0.5])
def test_subgrad(n, m, lbda):
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
            return obj(z, D, x, lbda=lbda) * n
        return approx_fprime(xk=z.ravel(), f=f, epsilon=1.0e-6).reshape(n, m)

    grad_ref = finite_grad(z)
    grad_test = subgrad(z, D, x, lbda)

    np.testing.assert_allclose(grad_ref, grad_test, rtol=5e-2)  # bad precision

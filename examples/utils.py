""" Utils module for examples. """
import time
import numpy as np
from carpet.lista import Lista
from carpet.synthesis_loss_gradient import grad, subgrad, obj
from carpet.optimization import fista
from carpet.utils import lipschitz_est
from carpet.proximity import soft_thresholding


def lista_like_tv(training_dataset, testing_dataset, D, lbda, type_='lista',
                  n_layers=10):
    """ LISTA-like solver for TV problem. """
    previous_parameters = None

    # get the loss function evolution for a given 'algorithm'
    train_loss, test_loss = [], []
    for n_layer_ in range(1, n_layers + 1):
        # declare network
        parametrization = 'lista' if type_ == 'ista' else type_
        lista = Lista(D=D, n_layers=n_layers, parametrization=parametrization,
                        max_iter=100, device='cpu', name=type_, verbose=1)
        t0_ = time.time()

        # initialize network
        if previous_parameters is not None:
            # iterate on layers
            for layer_parameters in previous_parameters:
                # iterate on different parameters
                for parameter_name, parameter in layer_parameters.items():
                        lista.set_parameters(parameter_name, parameter)

        if type_ != 'ista':  # train network
            lista.fit(training_dataset, lmbd=lbda)

        # save parameters
        previous_parameters = lista.export_parameters()

        print(f"[{type_}] model fitted in {time.time() - t0_:.1f}s")

        # get train and test error
        z_train = lista.transform(training_dataset, lbda,
                                  output_layer=n_layer_)
        train_loss.append(obj(z_train, D, training_dataset, lbda))

        z_test = lista.transform(testing_dataset, lbda, output_layer=n_layer_)
        test_loss.append(obj(z_test, D, testing_dataset, lbda))

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)

    # manually add loss func init value
    test_loss_init = obj(np.zeros_like(testing_dataset),
                         D, testing_dataset, lbda)
    test_loss = np.insert(test_loss, 0, test_loss_init)
    train_loss_init = obj(np.zeros_like(training_dataset),
                            D, training_dataset, lbda)
    train_loss = np.insert(train_loss, 0, train_loss_init)

    return train_loss, test_loss


def ista_like_tv(training_dataset, testing_dataset, D, lbda, type_='ista',
                 n_layers=1000):
    """ ISTA-like solver for TV problem. """
    # define initialization
    z0 = np.zeros_like(testing_dataset)

    # define type of iteration
    if type_ == 'fista':
        momentum = 'fista'
        restarting = None
    elif type_ == 'rsfista':
        momentum = 'fista'
        restarting = 'obj'
    else:
        momentum = 'ista'
        restarting = None

    # define function obj
    def _obj(z):
        return obj(z, D, testing_dataset, lbda)

    # define gradient
    if type_ == 'sub-gradient':
        def _grad(z):
            return subgrad(z, D, testing_dataset, lbda)
    else:
        def _grad(z):
            return grad(z, D, testing_dataset)

    if type_ == 'sub-gradient':
        def _prox(z, step_size):
            return z
    else:
        def _prox(z, step_size):
            return soft_thresholding(z, lbda, step_size)

    # define step-size
    if type_ == 'sub-gradient':
        # hard to majorate Lipschitz constant for sub-grad
        def AtA(z):
            return subgrad(z, D, testing_dataset, lbda)
        lipsc = 100. * lipschitz_est(AtA, z0.shape)
    else:
        lipsc = np.linalg.norm(D.T.dot(D), ord=2)
    step_size = 1.0 / lipsc

    # ista like iteration
    params = dict(grad=_grad, obj=_obj, prox=_prox, x0=z0,  momentum=momentum,
                  restarting=restarting, max_iter=n_layers,
                  step_size=step_size, early_stopping=False, debug=True,
                  verbose=1)
    _, loss = fista(**params)

    return None, loss

""" Utils module for examples. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import numpy as np
from carpet.lista import ALL_LISTA, ALL_LTV
from carpet.synthesis_loss_gradient import grad as grad_synth
from carpet.synthesis_loss_gradient import obj as obj_synth
from carpet.analysis_loss_gradient import subgrad as subgrad_analy
from carpet.analysis_loss_gradient import obj as obj_analy
from carpet.optimization import fista
from carpet.proximity import soft_thresholding


def lista_like_synth_tv(x_train, x_test, D, lbda, all_n_layers, type_='lista'):
    """ LISTA-like solver for synthesis TV problem. """
    z0_train = np.zeros_like(x_train.dot(D.T))
    z0_test = np.zeros_like(x_test.dot(D.T))

    previous_parameters = None
    train_loss = [obj_synth(z0_train, D, x_train, lbda)]
    test_loss = [obj_synth(z0_test, D, x_test, lbda)]
    for n_layers in all_n_layers:

        # declare network
        parametrization = 'lista' if type_ == 'ista' else type_
        if previous_parameters is not None:
            lista = ALL_LISTA[parametrization](D=D, n_layers=n_layers,
                                max_iter=100, device='cpu', name=type_,
                                initial_parameters=previous_parameters,
                                verbose=1)
        else:
            lista = ALL_LISTA[parametrization](D=D, n_layers=n_layers,
                              max_iter=100, device='cpu', name=type_,
                              verbose=1)

        t0_ = time.time()
        if type_ != 'ista':  # train network
            lista.fit(x_train, lmbd=lbda)

        print(f"[{type_}] model fitted in {time.time() - t0_:.1f}s")

        # save parameters
        previous_parameters = lista.export_parameters()

        # get train and test error
        z_train = lista.transform(x_train, lbda, output_layer=n_layers)
        train_loss.append(obj_synth(z_train, D, x_train, lbda))

        z_test = lista.transform(x_test, lbda, output_layer=n_layers)
        test_loss.append(obj_synth(z_test, D, x_test, lbda))

    return np.array(train_loss), np.array(test_loss)


def ista_like_synth_tv(x_train, x_test, D, lbda, all_n_layers, type_='ista'):
    """ ISTA-like solver for synthesis TV problem. """
    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(D.T.dot(D), ord=2)

    params = dict(
            grad=lambda z: grad_synth(z, D, x_train),
            obj=lambda z: obj_synth(z, D, x_train, lbda),
            prox=lambda z, step_size: soft_thresholding(z, lbda, step_size),
            x0=np.zeros_like(x_train),  momentum=type_, restarting=None,
            max_iter=max_iter, step_size=step_size, early_stopping=False,
            debug=True, verbose=1,
            )
    _, train_loss = fista(**params)
    print('')

    params = dict(
            grad=lambda z: grad_synth(z, D, x_test),
            obj=lambda z: obj_synth(z, D, x_test, lbda),
            prox=lambda z, step_size: soft_thresholding(z, lbda, step_size),
            x0=np.zeros_like(x_test),  momentum=type_, restarting=None,
            max_iter=max_iter, step_size=step_size, early_stopping=False,
            debug=True, verbose=1,
            )
    _, test_loss = fista(**params)

    return train_loss[[0] + all_n_layers], test_loss[[0] + all_n_layers]


def lista_like_analy_tv(x_train, x_test, D, lbda, all_n_layers, type_='lista'):
    """ LISTA-like solver for analysis TV problem. """
    previous_parameters = None

    z0_train = np.zeros_like(x_train)
    z0_test = np.zeros_like(x_test)

    train_loss_init = obj_analy(z0_train, D, x_train, lbda)
    test_loss_init = obj_analy(z0_test, D, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]
    for n_layers in all_n_layers:

        # declare network
        if previous_parameters is not None:
            lista = ALL_LTV[type_](D=D, n_layers=n_layers, max_iter=100,
                                device='cpu', name=type_, per_layer='one_shot',
                                initial_parameters=previous_parameters,
                                verbose=10)
        else:
            lista = ALL_LTV[type_](D=D, n_layers=n_layers, max_iter=100,
                                device='cpu', name=type_, per_layer='one_shot',
                                verbose=10)

        t0_ = time.time()
        lista.fit(x_train, lmbd=lbda)

        # # TODO plot here training loss
        # import matplotlib.pyplot as plt
        # plt.plot(lista.training_loss_)
        # plt.show()

        print(f"[{type_}] model fitted in {time.time() - t0_:.1f}s")

        # save parameters
        previous_parameters = lista.export_parameters()

        # get train and test error
        z_train = lista.transform(x_train, lbda, output_layer=n_layers)
        train_loss.append(obj_analy(z_train, D, x_train, lbda))

        z_test = lista.transform(x_test, lbda, output_layer=n_layers)
        test_loss.append(obj_analy(z_test, D, x_test, lbda))

    return np.array(train_loss), np.array(test_loss)


def ista_like_analy_tv(x_train, x_test, D, lbda, all_n_layers, type_='ista'):
    """ ISTA-like solver for analysis TV problem. """
    max_iter = all_n_layers[-1]
    step_size = 1.0e-8

    params = dict(
            grad=lambda z: subgrad_analy(z, D, x_train, lbda),
            obj=lambda z: obj_analy(z, D, x_train, lbda),
            prox=lambda z, step_size: z, x0=np.zeros_like(x_train),
            momentum=type_, restarting=None, max_iter=max_iter,
            step_size=step_size, early_stopping=False, debug=True, verbose=1,
            )
    _, train_loss = fista(**params)
    print('')

    params = dict(
            grad=lambda z: subgrad_analy(z, D, x_test, lbda),
            obj=lambda z: obj_analy(z, D, x_test, lbda),
            prox=lambda z, step_size: z, x0=np.zeros_like(x_test),
            momentum='ista', restarting=None, max_iter=max_iter,
            step_size=step_size, early_stopping=False, debug=True, verbose=1,
            )
    _, test_loss = fista(**params)

    return train_loss[[0] + all_n_layers], test_loss[[0] + all_n_layers]

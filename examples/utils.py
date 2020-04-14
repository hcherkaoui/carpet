""" Utils module for examples. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import numpy as np
from carpet.lista_synthesis import ALL_LISTA
from carpet.lista_analysis import ALL_LTV
from carpet.loss_gradient import (analysis_obj, analysis_grad, synthesis_grad,
                                  synthesis_obj, analysis_obj)
from carpet.optimization import fista, condatvu
from carpet.proximity import pseudo_soft_th_numpy


def learned_lasso_like_tv(x_train, x_test, L, lbda, all_n_layers, type_):
    """ NN-algo solver for synthesis TV problem. """
    params = None

    train_loss_init = synthesis_obj(x_train, L, x_train, lbda)
    test_loss_init = synthesis_obj(x_test, L, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]

    for n_layers in all_n_layers:

        # declare network
        parametrization = 'lista' if type_ == 'ista' else type_
        if params is not None:
            algo = ALL_LISTA[parametrization](D=L, n_layers=n_layers,
                                              max_iter=200, device='cpu',
                                              initial_parameters=params,
                                              verbose=0)
        else:
            algo = ALL_LISTA[parametrization](D=L, n_layers=n_layers,
                                              max_iter=200, device='cpu',
                                              verbose=0)

        t0_ = time.time()
        if type_ != 'ista':  # train network
            algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        z_train = algo.transform(x_train, lbda, output_layer=n_layers)
        train_loss_ = synthesis_obj(z_train, L, x_train, lbda)
        train_loss.append(train_loss_)
        z_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = synthesis_obj(z_test, L, x_test, lbda)
        test_loss.append(test_loss_)

        print(f"[{algo.name}-layers#{n_layers}] model fitted in "
              f"{delta_:.1f}s train-loss={train_loss_:.6e} "
              f"test-loss={test_loss_:.6e}")

    return np.array(train_loss), np.array(test_loss)


def lasso_like_tv(x_train, x_test, D, lbda, all_n_layers, type_):
    """ Iterative-algo solver for synthesis TV problem. """
    name = 'ISTA' if type_ == 'chambolle' else 'FISTA'
    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(D.T.dot(D), ord=2)

    momentum = None if type_ == 'ista' else type_

    print("[ISTA iterative] training loss")
    params = dict(
                grad=lambda z: synthesis_grad(z, D, x_train),
                obj=lambda z: synthesis_obj(z, D, x_train, lbda),
                prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                x0=x_train,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=1,
                )
    _, train_loss = fista(**params)

    print("[ISTA iterative] testing loss")
    params = dict(
                grad=lambda z: synthesis_grad(z, D, x_test),
                obj=lambda z: synthesis_obj(z, D, x_test, lbda),
                prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                x0=x_test,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=1,
                )
    _, test_loss = fista(**params)

    print(f"[{name}] iterations finished in train-loss={train_loss[-1]:.6e} "
          f"test-loss={test_loss[-1]:.6e}")

    return train_loss[[0] + all_n_layers], test_loss[[0] + all_n_layers]


def learned_chambolle_tv(x_train, x_test, D, lbda, all_n_layers, type_):
    """ NN-algo solver for analysis TV problem. """
    params = None

    train_loss_init = analysis_obj(x_train, D, x_train, lbda)
    test_loss_init = analysis_obj(x_test, D, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]

    for n_layers in all_n_layers:

        # declare network
        parametrization = 'coupledchambolle' if type_ == 'chambolle' else type_
        if params is not None:
            algo = ALL_LTV[parametrization](D=D, n_layers=n_layers,
                                            max_iter=200, device='cpu',
                                            initial_parameters=params,
                                            verbose=0)
        else:
            algo = ALL_LTV[parametrization](D=D, n_layers=n_layers,
                                            max_iter=200, device='cpu',
                                            verbose=0)

        t0_ = time.time()
        if type_ not in ['chambolle', 'condatvu']:  # train network
            algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        z_train = algo.transform(x_train, lbda, output_layer=n_layers)
        train_loss_ = analysis_obj(z_train, D, x_train, lbda)
        train_loss.append(train_loss_)
        z_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = analysis_obj(z_test, D, x_test, lbda)
        test_loss.append(test_loss_)

        print(f"[{algo.name}-layers#{n_layers}] model fitted in "
              f"{delta_:.1f}s train-loss={train_loss_:.6e} "
              f"test-loss={test_loss_:.6e}")

    return np.array(train_loss), np.array(test_loss)


def chambolle_tv(x_train, x_test, D, lbda, all_n_layers, type_):
    """ Chambolle solver for analysis TV problem. """
    name = 'ISTA' if type_ == 'chambolle' else 'FISTA'
    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(D.dot(D.T), ord=2)
    momentum = None if type_ == 'chambolle' else 'fista'

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    n_dim = D.shape[1]

    def _grad(v, D, x, lbda):
        return (v.dot(D.T) - x / lbda).dot(D)

    def _obj(v, D, x, lbda):
        v = np.atleast_2d(v)
        z = x - lbda * v.dot(D.T)  # switch to primal formulation
        return analysis_obj(z, D, x, lbda)

    def _prox(z, step_size):
        return np.clip(z, -1.0, 1.0)

    print("[ISTA iterative] training loss")
    params = dict(
            grad=lambda v: _grad(v, D, x_train, lbda),
            obj=lambda v: _obj(v, D, x_train, lbda),
            prox=_prox, x0=np.zeros((n_train_samples, n_dim)),
            momentum=momentum, restarting=None, max_iter=max_iter,
            step_size=step_size, early_stopping=False, debug=True, verbose=1,
            )
    _, train_loss = fista(**params)

    print("[ISTA iterative] testing loss")
    params = dict(
            grad=lambda v: _grad(v, D, x_test, lbda),
            obj=lambda v: _obj(v, D, x_test, lbda),
            prox=_prox, x0=np.zeros((n_test_samples, n_dim)),
            momentum=momentum, restarting=None, max_iter=max_iter,
            step_size=step_size, early_stopping=False, debug=True, verbose=1,
            )
    _, test_loss = fista(**params)

    print(f"[{name}] iterations finished in train-loss={train_loss[-1]:.6e} "
          f"test-loss={test_loss[-1]:.6e}")

    return train_loss[[0] + all_n_layers], test_loss[[0] + all_n_layers]


def condatvu_tv(x_train, x_test, D, lbda, all_n_layers, type_):
    """ Condat-Vu solver for analysis TV problem. """
    max_iter = all_n_layers[-1]
    rho = 1.0
    sigma = 0.5
    L_D, L_I = np.linalg.norm(D.dot(D.T), ord=2), 1.0  # lipschtiz constant
    tau = 1.0 / (L_I / 2.0 + sigma * L_D**2)

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    dim_z = D.shape[1]

    print("[Condat-Vu iterative] training loss")
    params = dict(
             grad=lambda z: analysis_grad(z, x_train),
             obj=lambda z: analysis_obj(z, D, x_train, lbda),
             prox=lambda z: pseudo_soft_th_numpy(z, lbda, 1.0 / sigma),
             psi=lambda z: z.dot(D),
             adj_psi=lambda z: z.dot(D.T),
             v0=np.zeros((n_train_samples, dim_z)),
             z0=x_train,
             lbda=lbda, sigma=sigma, tau=tau, rho=rho,
             max_iter=max_iter, early_stopping=False, debug=True, verbose=0,
             )
    _, _, train_loss = condatvu(**params)

    print("[Condat-Vu iterative] testing loss")
    params = dict(
             grad=lambda z: analysis_grad(z, x_test),
             obj=lambda z: analysis_obj(z, D, x_test, lbda),
             prox=lambda z: pseudo_soft_th_numpy(z, lbda, 1.0 / sigma),
             psi=lambda z: z.dot(D),
             adj_psi=lambda z: z.dot(D.T),
             v0=np.zeros((n_test_samples, dim_z)),
             z0=x_test,
             lbda=lbda, sigma=sigma, tau=tau, rho=rho,
             max_iter=max_iter, early_stopping=False, debug=True, verbose=0,
             )
    _, _, test_loss = condatvu(**params)

    print(f"[Condat-Vu] iterations finished in "
          f"train-loss={train_loss[-1]:.6e} test-loss={test_loss[-1]:.6e}")

    # print(train_loss[[0] + all_n_layers])
    # print(test_loss[[0] + all_n_layers])

    return train_loss[[0] + all_n_layers], test_loss[[0] + all_n_layers]

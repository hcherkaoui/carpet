""" Utils module for examples. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import numpy as np
from carpet.lista_synthesis import ALL_LISTA
from carpet.lista_analysis import ALL_LTV
from carpet.loss_gradient import (analysis_obj, analysis_grad, synthesis_grad,
                                  synthesis_obj)
from carpet.optimization import fista, condatvu
from carpet.proximity import pseudo_soft_th_numpy


def learned_lasso_like_tv(x_train, x_test, D, L, lbda, all_n_layers, type_):
    """ NN-algo solver for synthesis TV problem. """
    params = None

    z0_train = np.c_[x_train[:, 0], x_train.dot(D)]
    train_loss_init = synthesis_obj(z0_train, L, x_train, lbda)
    z0_test = np.c_[x_test[:, 0], x_test.dot(D)]
    test_loss_init = synthesis_obj(z0_test, L, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]
    train_reg_init = np.sum(np.abs(z0_train))
    test_reg_init = np.sum(np.abs(z0_test))
    train_reg, test_reg = [train_reg_init], [test_reg_init]

    for n_layers in all_n_layers:

        # declare network
        if params is not None:
            algo = ALL_LISTA[type_](D=L, n_layers=n_layers, max_iter=300,
                                    device='cpu', initial_parameters=params,
                                    verbose=0)
        else:
            algo = ALL_LISTA[type_](D=L, n_layers=n_layers, max_iter=300,
                                    device='cpu', verbose=0)

        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        z_train = algo.transform(x_train, lbda, output_layer=n_layers)
        train_loss_ = synthesis_obj(z_train, L, x_train, lbda)
        train_loss.append(train_loss_)
        train_reg.append(np.sum(np.abs(z_train)))

        z_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = synthesis_obj(z_test, L, x_test, lbda)
        test_loss.append(test_loss_)
        test_reg.append(np.sum(np.abs(z_test)))

        print(f"[{algo.name}|layers#{n_layers:3d}] model fitted in "
              f"{delta_:4.1f}s train-loss={train_loss_:.6e} "
              f"test-loss={test_loss_:.6e}")

    return np.array(train_loss), np.array(test_loss), np.array(train_reg), \
           np.array(test_reg)


def lasso_like_tv(x_train, x_test, D, L, lbda, all_n_layers, type_):
    """ Iterative-algo solver for synthesis TV problem. """
    name = 'ISTA' if type_ == 'chambolle' else 'FISTA'
    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(L.T.dot(L), ord=2)

    z0_train = np.c_[x_train[:, 0], x_train.dot(D)]
    z0_test = np.c_[x_test[:, 0], x_test.dot(D)]

    momentum = None if type_ == 'ista' else type_

    print("[ISTA iterative] training loss")
    params = dict(
                grad=lambda z: synthesis_grad(z, L, x_train),
                obj=lambda z: synthesis_obj(z, L, x_train, lbda),
                prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                x0=z0_train,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=1,
                )
    _, saved_z_train, train_loss = fista(**params)

    print("[ISTA iterative] testing loss")
    params = dict(
                grad=lambda z: synthesis_grad(z, L, x_test),
                obj=lambda z: synthesis_obj(z, L, x_test, lbda),
                prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                x0=z0_test,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=1,
                )
    _, saved_z_test, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    saved_z_train = [saved_z_train[i] for i in [0] + all_n_layers]
    train_reg = np.array([np.sum(np.abs(saved_z_train_))
                          for saved_z_train_ in saved_z_train])
    saved_z_test = [saved_z_test[i] for i in [0] + all_n_layers]
    test_reg = np.array([np.sum(np.abs(saved_z_test_))
                         for saved_z_test_ in saved_z_test])

    print(f"[{name}] iterations finished in train-loss={train_loss[-1]:.6e} "
          f"test-loss={test_loss[-1]:.6e}")

    return train_loss, test_loss, train_reg, test_reg


def learned_chambolle_tv(x_train, x_test, D, L, lbda, all_n_layers, type_):
    """ NN-algo solver for analysis TV problem. """
    params = None

    Lz0_train = x_train  #v0_train = 0 (implicitly)
    Lz0_test = x_test  #v0_test = 0 (implicitly)

    train_loss_init = analysis_obj(Lz0_train, D, x_train, lbda)
    test_loss_init = analysis_obj(Lz0_test, D, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]
    train_reg_init = np.sum(np.abs(Lz0_train.dot(D)))
    test_reg_init = np.sum(np.abs(Lz0_test.dot(D)))
    train_reg, test_reg = [train_reg_init], [test_reg_init]

    for n_layers in all_n_layers:

        # declare network
        if params is not None:
            algo = ALL_LTV[type_](D=D, n_layers=n_layers, max_iter=300,
                                  device='cpu', initial_parameters=params,
                                  verbose=0)
        else:
            algo = ALL_LTV[type_](D=D, n_layers=n_layers, max_iter=300,
                                  device='cpu', verbose=0)

        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        Lz_train = algo.transform(x_train, lbda, output_layer=n_layers)
        train_loss_ = analysis_obj(Lz_train, D, x_train, lbda)
        train_loss.append(train_loss_)
        train_reg.append(np.sum(np.abs(Lz_train.dot(D))))

        Lz_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = analysis_obj(Lz_test, D, x_test, lbda)
        test_loss.append(test_loss_)
        test_reg.append(np.sum(np.abs(Lz_test.dot(D))))

        print(f"[{algo.name}|layers#{n_layers:3d}] model fitted in "
              f"{delta_:4.1f}s train-loss={train_loss_:.6e} "
              f"test-loss={test_loss_:.6e}")

    return np.array(train_loss), np.array(test_loss), np.array(train_reg), \
           np.array(test_reg)


def chambolle_tv(x_train, x_test, D, L, lbda, all_n_layers, type_):
    """ Chambolle solver for analysis TV problem. """
    name = 'ISTA' if type_ == 'chambolle' else 'FISTA'
    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(D.dot(D.T), ord=2)
    momentum = None if type_ == 'chambolle' else 'fista'

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    n_dim = D.shape[1]

    v0_train = np.zeros((n_train_samples, n_dim))
    v0_test = np.zeros((n_test_samples, n_dim))

    def _grad(v, D, x, lbda):
        v = np.atleast_2d(v)
        return (v.dot(D.T) - x / lbda).dot(D)

    def _obj(v, D, x, lbda):
        v = np.atleast_2d(v)
        Lz = x - lbda * v.dot(D.T)  # switch to primal formulation
        return analysis_obj(Lz, D, x, lbda)

    def _prox(z, step_size):
        return np.clip(z, -1.0, 1.0)

    print("[ISTA iterative] training loss")
    params = dict(
            grad=lambda v: _grad(v, D, x_train, lbda),
            obj=lambda v: _obj(v, D, x_train, lbda),
            prox=_prox, x0=v0_train, momentum=momentum, restarting=None,
            max_iter=max_iter, step_size=step_size, early_stopping=False,
            debug=True, verbose=1,
            )
    _, saved_v_train, train_loss = fista(**params)

    print("[ISTA iterative] testing loss")
    params = dict(
            grad=lambda v: _grad(v, D, x_test, lbda),
            obj=lambda v: _obj(v, D, x_test, lbda),
            prox=_prox, x0=v0_test, momentum=momentum, restarting=None,
            max_iter=max_iter, step_size=step_size, early_stopping=False,
            debug=True, verbose=1,
            )
    _, saved_v_test, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    saved_Lz_train = [x_train - lbda * saved_v_train[i].dot(D.T)
                      for i in [0] + all_n_layers]
    train_reg = np.array([np.sum(np.abs(saved_Lz_train_.dot(D)))
                          for saved_Lz_train_ in saved_Lz_train])
    saved_Lz_test = [x_test - lbda * saved_v_test[i].dot(D.T)
                     for i in [0] + all_n_layers]
    test_reg = np.array([np.sum(np.abs(saved_Lz_test_.dot(D)))
                         for saved_Lz_test_ in saved_Lz_test])

    print(f"[{name}] iterations finished in train-loss={train_loss[-1]:.6e} "
          f"test-loss={test_loss[-1]:.6e}")

    return train_loss, test_loss, train_reg, test_reg


def condatvu_tv(x_train, x_test, D, L, lbda, all_n_layers, type_):
    """ Condat-Vu solver for analysis TV problem. """
    max_iter = all_n_layers[-1]
    rho = 1.0
    sigma = 0.5
    L_D, L_I = np.linalg.norm(D.dot(D.T), ord=2), 1.0  # lipschtiz constant
    tau = 1.0 / (L_I / 2.0 + sigma * L_D**2)

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    n_dim = D.shape[1]

    Lz0_train = x_train
    Lz0_test = x_test
    v0_train = np.zeros((n_train_samples, n_dim))
    v0_test = np.zeros((n_test_samples, n_dim))

    print("[Condat-Vu iterative] training loss")
    params = dict(
             grad=lambda Lz: analysis_grad(Lz, x_train),
             obj=lambda Lz: analysis_obj(Lz, D, x_train, lbda),
             prox=lambda u: pseudo_soft_th_numpy(u, lbda, 1.0 / sigma),
             psi=lambda Lz: Lz.dot(D),
             adj_psi=lambda v: v.dot(D.T),
             v0=v0_train,
             z0=Lz0_train,
             lbda=lbda, sigma=sigma, tau=tau, rho=rho,
             max_iter=max_iter, early_stopping=False, debug=True, verbose=1,
             )
    _, _, saved_Lz_train, _, train_loss = condatvu(**params)

    print("[Condat-Vu iterative] testing loss")
    params = dict(
             grad=lambda Lz: analysis_grad(Lz, x_test),
             obj=lambda Lz: analysis_obj(Lz, D, x_test, lbda),
             prox=lambda u: pseudo_soft_th_numpy(u, lbda, 1.0 / sigma),
             psi=lambda Lz: Lz.dot(D),
             adj_psi=lambda v: v.dot(D.T),
             v0=v0_test,
             z0=Lz0_test,
             lbda=lbda, sigma=sigma, tau=tau, rho=rho,
             max_iter=max_iter, early_stopping=False, debug=True, verbose=1,
             )
    _, _, saved_Lz_test, _, test_loss = condatvu(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    saved_Lz_train = [saved_Lz_train[i] for i in [0] + all_n_layers]
    train_reg = np.array([np.sum(np.abs(saved_Lz_train_.dot(D)))
                          for saved_Lz_train_ in saved_Lz_train])
    saved_Lz_test = [saved_Lz_test[i] for i in [0] + all_n_layers]
    test_reg = np.array([np.sum(np.abs(saved_Lz_test_.dot(D)))
                         for saved_Lz_test_ in saved_Lz_test])

    print(f"[Condat-Vu] iterations finished in "
          f"train-loss={train_loss[-1]:.6e} test-loss={test_loss[-1]:.6e}")

    return train_loss, test_loss, train_reg, test_reg

""" Utils module for examples. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import numpy as np
import pandas as pd
from joblib import Memory
from prox_tv import tv1_1d
from carpet import LearnTVAlgo
from carpet.utils import init_vuz, v_to_u
from carpet.loss_gradient import (analysis_primal_obj, analysis_primal_grad,
                                  analysis_dual_grad, synthesis_primal_grad,
                                  synthesis_primal_obj, tv_reg, l1_reg)
from carpet.optimization import fista, condatvu
from carpet.proximity import pseudo_soft_th_numpy


memory = Memory('__cache_dir__', verbose=0)


def _synthesis_learned_algo(x_train, x_test, A, D, L, lbda, all_n_layers,
                            type_, verbose=1):
    """ NN-algo solver for synthesis TV problem. """
    params = None

    _, _, z0_test = init_vuz(A, D, x_test, lbda)
    _, _, z0_train = init_vuz(A, D, x_train, lbda)

    train_loss_init = synthesis_primal_obj(z0_train, A, L, x_train, lbda)
    test_loss_init = synthesis_primal_obj(z0_test, A, L, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]
    train_reg, test_reg = [l1_reg(z0_train)], [l1_reg(z0_test)]

    for n_layers in all_n_layers:

        # declare network
        if params is not None:
            algo = LearnTVAlgo(algo_type=type_, A=A, n_layers=n_layers,
                               max_iter=300, device='cpu',
                               initial_parameters=params, verbose=0)
        else:
            algo = LearnTVAlgo(algo_type=type_, A=A, n_layers=n_layers,
                               max_iter=300, device='cpu', verbose=0)

        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        z_train = algo.transform(x_train, lbda, output_layer=n_layers)
        train_loss_ = synthesis_primal_obj(z_train, A, L, x_train, lbda)
        train_loss.append(train_loss_)
        train_reg.append(l1_reg(z_train))

        z_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = synthesis_primal_obj(z_test, A, L, x_test, lbda)
        test_loss.append(test_loss_)
        test_reg.append(l1_reg(z_test))

        if verbose > 0:
            print(f"[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s train-loss={train_loss_:.8e} "
                  f"test-loss={test_loss_:.8e}")

    to_return = (np.array(train_loss), np.array(test_loss),
                 np.array(train_reg), np.array(test_reg))

    return to_return


synthesis_learned_algo = memory.cache(_synthesis_learned_algo)


def _analysis_learned_algo(x_train, x_test, A, D, L, lbda, all_n_layers, type_,
                           verbose=1):
    """ NN-algo solver for analysis TV problem. """
    params = None

    _, u0_train, _ = init_vuz(A, D, x_train, lbda)
    _, u0_test, _ = init_vuz(A, D, x_test, lbda)

    train_loss_init = analysis_primal_obj(u0_train, A, D, x_train, lbda)
    test_loss_init = analysis_primal_obj(u0_test, A, D, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]
    train_reg, test_reg = [tv_reg(u0_train, D)], [tv_reg(u0_test, D)]

    algo_type = 'origtv' if ('untrained' in type_) else type_

    for n_layers in all_n_layers:

        # declare network
        if params is not None:
            algo = LearnTVAlgo(algo_type=algo_type, A=A, n_layers=n_layers,
                               max_iter=300, device='cpu',
                               initial_parameters=params, verbose=0)
        else:
            algo = LearnTVAlgo(algo_type=algo_type, A=A, n_layers=n_layers,
                               max_iter=300, device='cpu', verbose=0)

        t0_ = time.time()
        if 'untrained' not in type_:
            algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        u_train = algo.transform(x_train, lbda, output_layer=n_layers)
        train_loss_ = analysis_primal_obj(u_train, A, D, x_train, lbda)
        train_loss.append(train_loss_)
        train_reg.append(tv_reg(u_train, D))

        u_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = analysis_primal_obj(u_test, A, D, x_test, lbda)
        test_loss.append(test_loss_)
        test_reg.append(tv_reg(u_test, D))

        if verbose > 0:
            print(f"[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s train-loss={train_loss_:.8e} "
                  f"test-loss={test_loss_:.8e}")

    to_return = (np.array(train_loss), np.array(test_loss),
                 np.array(train_reg), np.array(test_reg))

    return to_return


analysis_learned_algo = memory.cache(_analysis_learned_algo)


def synthesis_iter_algo(x_train, x_test, A, D, L, lbda, all_n_layers, type_,
                        verbose=1):
    """ Iterative-algo solver for synthesis TV problem. """
    name = 'ISTA' if type_ == 'chambolle' else 'FISTA'
    max_iter = all_n_layers[-1]
    LA = L.dot(A)
    step_size = 1.0 / np.linalg.norm(LA, ord=2) ** 2

    _, _, z0_test = init_vuz(A, D, x_test, lbda)
    _, _, z0_train = init_vuz(A, D, x_train, lbda)

    momentum = None if type_ == 'ista' else type_

    if verbose > 0:
        print("[ISTA iterative] training loss")
    params = dict(
                grad=lambda z: synthesis_primal_grad(z, A, L, x_train),
                obj=lambda z: synthesis_primal_obj(z, A, L, x_train, lbda),
                prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                x0=z0_train,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=verbose,
                )
    _, saved_z_train, train_loss = fista(**params)

    if verbose > 0:
        print("[ISTA iterative] testing loss")
    params = dict(
                grad=lambda z: synthesis_primal_grad(z, A, L, x_test),
                obj=lambda z: synthesis_primal_obj(z, A, L, x_test, lbda),
                prox=lambda z, s: pseudo_soft_th_numpy(z, lbda, s),
                x0=z0_test,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=verbose,
                )
    _, saved_z_test, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    saved_z_train = [saved_z_train[i] for i in [0] + all_n_layers]
    train_reg = np.array([l1_reg(saved_z_train_)
                          for saved_z_train_ in saved_z_train])

    saved_z_test = [saved_z_test[i] for i in [0] + all_n_layers]
    test_reg = np.array([l1_reg(saved_z_test_)
                         for saved_z_test_ in saved_z_test])

    if verbose > 0:
        print(f"[{name}] iterations finished "
              f"train-loss={train_loss[-1]:.8e} test-loss={test_loss[-1]:.8e}")

    return train_loss, test_loss, train_reg, test_reg


def analysis_primal_iter_algo(x_train, x_test, A, D, L, lbda, all_n_layers,
                              type_, verbose=1):
    """ Iterative-algo solver for synthesis TV problem. """
    name = 'ISTA' if type_ == 'ista' else 'FISTA'
    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(A, ord=2) ** 2

    _, u0_test, _ = init_vuz(A, D, x_test, lbda)
    _, u0_train, _ = init_vuz(A, D, x_train, lbda)

    momentum = None if type_ == 'ista' else type_

    if verbose > 0:
        print(f"[analysis {name} iterative] training loss")
    params = dict(
                grad=lambda z: analysis_primal_grad(z, A, x_train),
                obj=lambda z: analysis_primal_obj(z, A, D, x_train, lbda),
                prox=lambda z, s: np.array([tv1_1d(z_, lbda * s) for z_ in z]),
                x0=u0_train,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=verbose,
                )
    _, saved_z_train, train_loss = fista(**params)

    if verbose > 0:
        print(f"[analysis {name} iterative] testing loss")
    params = dict(
                grad=lambda z: analysis_primal_grad(z, A, x_test),
                obj=lambda z: analysis_primal_obj(z, A, D, x_test, lbda),
                prox=lambda z, s: np.array([tv1_1d(z_, lbda * s) for z_ in z]),
                x0=u0_test,  momentum=momentum, restarting=None,
                max_iter=max_iter, step_size=step_size, early_stopping=False,
                debug=True, verbose=verbose,
                )
    _, saved_z_test, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    saved_z_train = [saved_z_train[i] for i in [0] + all_n_layers]
    train_reg = np.array([l1_reg(saved_z_train_)
                          for saved_z_train_ in saved_z_train])

    saved_z_test = [saved_z_test[i] for i in [0] + all_n_layers]
    test_reg = np.array([l1_reg(saved_z_test_)
                         for saved_z_test_ in saved_z_test])

    if verbose > 0:
        print(f"[{name}] iterations finished "
              f"train-loss={train_loss[-1]:.6e} test-loss={test_loss[-1]:.6e}")

    return train_loss, test_loss, train_reg, test_reg


def analysis_dual_iter_algo(x_train, x_test, A, D, L, lbda, all_n_layers,
                            type_):
    """ Chambolle solver for analysis TV problem. """
    inv_A = np.linalg.pinv(A)
    Psi_A = inv_A.dot(D)

    def _grad(v, x):
        return analysis_dual_grad(v, A, D, x, lbda, Psi_A=Psi_A)

    def _obj(v, x):
        v = np.atleast_2d(v)
        u = v_to_u(v, x, lbda, Psi_A=Psi_A, inv_A=inv_A)
        return analysis_primal_obj(u, A, D, x, lbda)

    def _prox(z, step_size):
        return np.clip(z, -lbda, lbda)

    v0_test, _, _ = init_vuz(A, D, x_test, lbda)
    v0_train, _, _ = init_vuz(A, D, x_train, lbda)

    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(Psi_A.T, ord=2) ** 2
    momentum = None if type_ == 'chambolle' else 'fista'
    name = 'ISTA' if type_ == 'chambolle' else 'FISTA'

    print("[ISTA iterative] training loss")
    params = dict(grad=lambda v: _grad(v, x_train),
                  obj=lambda v: _obj(v, x_train), prox=_prox,
                  x0=v0_train, momentum=momentum, restarting=None,
                  max_iter=max_iter, step_size=step_size, early_stopping=False,
                  debug=True, verbose=1,
                  )
    _, saved_v_train, train_loss = fista(**params)

    print("[ISTA iterative] testing loss")
    params = dict(grad=lambda v: _grad(v, x_test),
                  obj=lambda v: _obj(v, x_test), prox=_prox,
                  x0=v0_test, momentum=momentum, restarting=None,
                  max_iter=max_iter, step_size=step_size, early_stopping=False,
                  debug=True, verbose=1,
                  )
    _, saved_v_test, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    saved_Lz_train = [
        v_to_u(saved_v_train[i], x_train, lbda, inv_A=inv_A, Psi_A=Psi_A)
        for i in [0] + all_n_layers]
    train_reg = np.array([tv_reg(saved_Lz_train_, D)
                          for saved_Lz_train_ in saved_Lz_train])
    saved_Lz_test = [
        v_to_u(saved_v_test[i], x_test, lbda, inv_A=inv_A, Psi_A=Psi_A)
        for i in [0] + all_n_layers]
    test_reg = np.array([tv_reg(saved_Lz_test_, D)
                         for saved_Lz_test_ in saved_Lz_test])

    print(f"[{name}] iterations finished train-loss={train_loss[-1]:.6e} "
          f"test-loss={test_loss[-1]:.6e}")

    return train_loss, test_loss, train_reg, test_reg


def analysis_primal_dual_iter_algo(x_train, x_test, A, D, L, lbda,
                                   all_n_layers, type_, verbose=1):
    """ Condat-Vu solver for analysis TV problem. """
    max_iter = all_n_layers[-1]
    rho = 1.0
    sigma = 0.5
    L_A = np.linalg.norm(A, ord=2) ** 2
    L_D = np.linalg.norm(D, ord=2) ** 2
    tau = 1.0 / (L_A / 2.0 + sigma * L_D**2)

    v0_test, u0_test, _ = init_vuz(A, D, x_test, lbda, force_numpy=True)
    v0_train, u0_train, _ = init_vuz(A, D, x_train, lbda, force_numpy=True)

    if verbose > 0:
        print("[Condat-Vu iterative] training loss")
    params = dict(
             grad=lambda Lz: analysis_primal_grad(Lz, A, x_train),
             obj=lambda Lz: analysis_primal_obj(Lz, A, D, x_train, lbda),
             prox=lambda u: pseudo_soft_th_numpy(u, lbda, 1.0 / sigma),
             psi=lambda Lz: Lz.dot(D),
             adj_psi=lambda v: v.dot(D.T),
             v0=v0_train,
             z0=u0_train,
             lbda=lbda, sigma=sigma, tau=tau, rho=rho, max_iter=max_iter,
             early_stopping=False, debug=True, verbose=verbose,
             )
    _, _, saved_Lz_train, _, train_loss = condatvu(**params)

    if verbose > 0:
        print("[Condat-Vu iterative] testing loss")
    params = dict(
             grad=lambda Lz: analysis_primal_grad(Lz, A, x_test),
             obj=lambda Lz: analysis_primal_obj(Lz, A, D, x_test, lbda),
             prox=lambda u: pseudo_soft_th_numpy(u, lbda, 1.0 / sigma),
             psi=lambda Lz: Lz.dot(D),
             adj_psi=lambda v: v.dot(D.T),
             v0=v0_test,
             z0=u0_test,
             lbda=lbda, sigma=sigma, tau=tau, rho=rho, max_iter=max_iter,
             early_stopping=False, debug=True, verbose=verbose,
             )
    _, _, saved_Lz_test, _, test_loss = condatvu(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    saved_Lz_train = [saved_Lz_train[i] for i in [0] + all_n_layers]
    train_reg = np.array([tv_reg(saved_Lz_train_, D)
                          for saved_Lz_train_ in saved_Lz_train])

    saved_Lz_test = [saved_Lz_test[i] for i in [0] + all_n_layers]
    test_reg = np.array([tv_reg(saved_Lz_test_, D)
                         for saved_Lz_test_ in saved_Lz_test])

    if verbose > 0:
        print(f"[Condat-Vu] iterations finished "
              f"train-loss={train_loss[-1]:.8e} test-loss={test_loss[-1]:.8e}")

    return train_loss, test_loss, train_reg, test_reg


def analysis_learned_taut_string(x_train, x_test, A, D, L, lbda, all_n_layers,
                                 type_=None, verbose=1):
    """ NN-algo solver for analysis TV problem. """
    params = None

    # helper to record train/test loss
    log = []

    def record_loss(u_train, u_test):
        log.append(dict(
            train_loss=analysis_primal_obj(u_train, A, D, x_train, lbda),
            test_loss=analysis_primal_obj(u_test, A, D, x_test, lbda),
            train_reg=tv_reg(u_train, D),
            test_reg=tv_reg(u_test, D)
        ))

    _, u0_train, _ = init_vuz(A, D, x_train, lbda)
    _, u0_test, _ = init_vuz(A, D, x_test, lbda)
    record_loss(u0_train, u0_test)

    for n_layers in all_n_layers:

        # Declare network for the given number of layers. Warm-init the first
        # layers with parameters learned with previous networks if any.
        algo = LearnTVAlgo(algo_type='lpgd_taut_string', A=A,
                           n_layers=n_layers, max_iter=500, device='cpu',
                           initial_parameters=params, verbose=0)

        # train
        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        u_train = algo.transform(x_train, lbda, output_layer=n_layers)
        u_test = algo.transform(x_test, lbda, output_layer=n_layers)
        record_loss(u_train, u_test)

        if verbose > 0:
            print(f"[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s")

    df = pd.DataFrame(log)
    to_return = (df['train_loss'].values, df['test_loss'].values,
                 df['train_reg'].values, df['test_reg'].values)

    return to_return

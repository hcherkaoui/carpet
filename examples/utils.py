""" Utils module for examples. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
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
                                  synthesis_primal_obj)
from carpet.optimization import fista, condatvu
from carpet.proximity import _soft_th_numpy, pseudo_soft_th_numpy


memory = Memory('__cache_dir__', verbose=0)


@memory.cache
def synthesis_learned_algo(x_train, x_test, A, D, L, lbda, all_n_layers,
                           type_, max_iter=300, device=None, net_kwargs=None,
                           verbose=1):
    """ NN-algo solver for synthesis TV problem. """
    net_kwargs = dict() if net_kwargs is None else net_kwargs
    params = None

    _, _, z0_test = init_vuz(A, D, x_test)
    _, _, z0_train = init_vuz(A, D, x_train)

    train_loss_init = synthesis_primal_obj(z0_train, A, L, x_train, lbda)
    test_loss_init = synthesis_primal_obj(z0_test, A, L, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]

    for n_layers in all_n_layers:

        # declare network
        algo = LearnTVAlgo(algo_type=type_, A=A, n_layers=n_layers,
                           max_iter=max_iter, device=device,
                           initial_parameters=params, verbose=verbose,
                           **net_kwargs)

        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        z_train = algo.transform(x_train, lbda, output_layer=n_layers)
        train_loss_ = synthesis_primal_obj(z_train, A, L, x_train, lbda)
        train_loss.append(train_loss_)

        z_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = synthesis_primal_obj(z_test, A, L, x_test, lbda)
        test_loss.append(test_loss_)

        if verbose > 0:
            print(f"\r[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s train-loss={train_loss_:.4e} "
                  f"test-loss={test_loss_:.4e}")

    to_return = (np.array(train_loss), np.array(test_loss))

    return to_return


@memory.cache
def analysis_learned_algo(x_train, x_test, A, D, L, lbda, all_n_layers, type_,
                          max_iter=300, device=None, net_kwargs=None,
                          verbose=1):
    """ NN-algo solver for analysis TV problem. """
    net_kwargs = dict() if net_kwargs is None else net_kwargs
    params = None

    _, u0_train, _ = init_vuz(A, D, x_train)
    _, u0_test, _ = init_vuz(A, D, x_test)

    train_loss_init = analysis_primal_obj(u0_train, A, D, x_train, lbda)
    test_loss_init = analysis_primal_obj(u0_test, A, D, x_test, lbda)
    train_loss, test_loss = [train_loss_init], [test_loss_init]

    algo_type = 'origtv' if ('untrained' in type_) else type_

    for n_layers in all_n_layers:

        # declare network
        algo = LearnTVAlgo(algo_type=algo_type, A=A, n_layers=n_layers,
                           max_iter=max_iter, device=device,
                           initial_parameters=params, verbose=verbose,
                           **net_kwargs)

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

        u_test = algo.transform(x_test, lbda, output_layer=n_layers)
        test_loss_ = analysis_primal_obj(u_test, A, D, x_test, lbda)
        test_loss.append(test_loss_)

        if verbose > 0:
            print(f"\r[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s train-loss={train_loss_:.4e} "
                  f"test-loss={test_loss_:.4e}")

    to_return = (np.array(train_loss), np.array(test_loss))

    return to_return


@memory.cache
def analysis_learned_taut_string(x_train, x_test, A, D, L, lbda, all_n_layers,
                                 type_=None, max_iter=300, device=None,
                                 net_kwargs=None, verbose=1):
    """ NN-algo solver for analysis TV problem. """
    net_kwargs = dict() if net_kwargs is None else net_kwargs
    params = None
    l_loss = []

    def record_loss(l_loss, u_train, u_test):
        l_loss.append(dict(
            train_loss=analysis_primal_obj(u_train, A, D, x_train, lbda),
            test_loss=analysis_primal_obj(u_test, A, D, x_test, lbda),
        ))
        return l_loss

    _, u0_train, _ = init_vuz(A, D, x_train)
    _, u0_test, _ = init_vuz(A, D, x_test)
    record_loss(l_loss, u0_train, u0_test)

    for n, n_layers in enumerate(all_n_layers):

        # Declare network for the given number of layers. Warm-init the first
        # layers with parameters learned with previous networks if any.
        algo = LearnTVAlgo(algo_type='lpgd_taut_string', A=A,
                           n_layers=n_layers, max_iter=max_iter, device=device,
                           initial_parameters=params, verbose=verbose,
                           **net_kwargs)

        # train
        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        u_train = algo.transform(x_train, lbda, output_layer=n_layers)
        u_test = algo.transform(x_test, lbda, output_layer=n_layers)
        l_loss = record_loss(l_loss, u_train, u_test)

        if verbose > 0:
            train_loss = l_loss[n]['train_loss']
            test_loss = l_loss[n]['test_loss']
            print(f"\r[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s train-loss={train_loss:.4e} "
                  f"test-loss={test_loss:.4e}")

    df = pd.DataFrame(l_loss)
    to_return = (df['train_loss'].values, df['test_loss'].values)

    return to_return


def synthesis_iter_algo(x_train, x_test, A, D, L, lbda, all_n_layers, type_,
                        max_iter=300, device=None, net_kwargs=None,
                        verbose=1):
    """ Iterative-algo solver for synthesis TV problem. """
    net_kwargs = dict() if net_kwargs is None else net_kwargs
    name = 'ISTA' if type_ == 'chambolle' else 'FISTA'
    max_iter = all_n_layers[-1]
    LA = L.dot(A)
    step_size = 1.0 / np.linalg.norm(LA, ord=2) ** 2

    _, _, z0_test = init_vuz(A, D, x_test)
    _, _, z0_train = init_vuz(A, D, x_train)

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
    _, train_loss = fista(**params)

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
    _, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    if verbose > 0:
        print(f"[{name}] iterations finished "
              f"train-loss={train_loss[-1]:.4e} test-loss={test_loss[-1]:.4e}")

    return train_loss, test_loss


def analysis_primal_iter_algo(x_train, x_test, A, D, L, lbda, all_n_layers,
                              type_, max_iter=300, device=None,
                              net_kwargs=None, verbose=1):
    """ Iterative-algo solver for synthesis TV problem. """
    net_kwargs = dict() if net_kwargs is None else net_kwargs
    name = 'ISTA' if type_ == 'ista' else 'FISTA'
    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(A, ord=2) ** 2

    _, u0_test, _ = init_vuz(A, D, x_test)
    _, u0_train, _ = init_vuz(A, D, x_train)

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
    _, train_loss = fista(**params)

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
    _, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    if verbose > 0:
        print(f"\r[{name}] iterations finished "
              f"train-loss={train_loss[-1]:.6e} test-loss={test_loss[-1]:.6e}")

    return train_loss, test_loss


def analysis_dual_iter_algo(x_train, x_test, A, D, L, lbda, all_n_layers,
                            type_, max_iter=300, device=None,
                            net_kwargs=None, verbose=1):
    """ Chambolle solver for analysis TV problem. """
    net_kwargs = dict() if net_kwargs is None else net_kwargs
    Psi_A = np.linalg.pinv(A).dot(D)
    inv_AtA = np.linalg.pinv(A.dot(A.T))

    def _grad(v, x):
        return analysis_dual_grad(v, A, D, x, Psi_A=Psi_A)

    def _obj(v, x):
        v = np.atleast_2d(v)
        u = v_to_u(v, x, A=A, D=D, inv_AtA=inv_AtA)
        return analysis_primal_obj(u, A, D, x, lbda)

    def _prox(v, step_size):
        # XXX step_size is here to homogenize API
        v = np.atleast_2d(v)
        return np.clip(v, -lbda, lbda)

    v0_test, _, _ = init_vuz(A, D, x_test)
    v0_train, _, _ = init_vuz(A, D, x_train)

    max_iter = all_n_layers[-1]
    step_size = 1.0 / np.linalg.norm(Psi_A, ord=2) ** 2
    momentum = None if type_ == 'ista' else 'fista'
    name = 'ISTA' if type_ == 'ista' else 'FISTA'

    print("[ISTA iterative] training loss")
    params = dict(
        grad=lambda v: _grad(v, x_train),
        obj=lambda v: _obj(v, x_train), prox=_prox,
        x0=v0_train, momentum=momentum, restarting=None,
        max_iter=max_iter, step_size=step_size, early_stopping=False,
        debug=True, verbose=verbose,
    )
    _, train_loss = fista(**params)

    print("[ISTA iterative] testing loss")
    params = dict(
        grad=lambda v: _grad(v, x_test),
        obj=lambda v: _obj(v, x_test), prox=_prox,
        x0=v0_test, momentum=momentum, restarting=None,
        max_iter=max_iter, step_size=step_size, early_stopping=False,
        debug=True, verbose=verbose
    )
    _, test_loss = fista(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    print(f"\r[{name}] iterations finished train-loss={train_loss[-1]:.6e} "
          f"test-loss={test_loss[-1]:.6e}")

    return train_loss, test_loss


def analysis_primal_dual_iter_algo(x_train, x_test, A, D, L, lbda,
                                   all_n_layers, type_, max_iter=300,
                                   device=None, net_kwargs=None, verbose=1):
    """ Condat-Vu solver for analysis TV problem. """
    net_kwargs = dict() if net_kwargs is None else net_kwargs
    max_iter = all_n_layers[-1]
    rho = 1.0
    sigma = 0.5
    L_A = np.linalg.norm(A, ord=2) ** 2
    L_D = np.linalg.norm(D, ord=2) ** 2
    tau = 1.0 / (L_A / 2.0 + sigma * L_D)

    v0_test, u0_test, _ = init_vuz(A, D, x_test, force_numpy=True)
    v0_train, u0_train, _ = init_vuz(A, D, x_train, force_numpy=True)

    if verbose > 0:
        print("[Condat-Vu iterative] training loss")
    params = dict(
        grad=lambda u: analysis_primal_grad(u, A, x_train),
        obj=lambda u: analysis_primal_obj(u, A, D, x_train, lbda),
        prox=lambda t: _soft_th_numpy(t, lbda / sigma),
        psi=lambda u: u.dot(D), adj_psi=lambda v: v.dot(D.T),
        v0=v0_train, z0=u0_train, lbda=lbda, sigma=sigma, tau=tau, rho=rho,
        max_iter=max_iter, early_stopping=False, debug=True, verbose=verbose,
    )
    _, _, train_loss = condatvu(**params)

    if verbose > 0:
        print("[Condat-Vu iterative] testing loss")
    params = dict(
        grad=lambda u: analysis_primal_grad(u, A, x_test),
        obj=lambda u: analysis_primal_obj(u, A, D, x_test, lbda),
        prox=lambda t: _soft_th_numpy(t, lbda / sigma),
        psi=lambda u: u.dot(D), adj_psi=lambda v: v.dot(D.T),
        v0=v0_test, z0=u0_test, lbda=lbda, sigma=sigma, tau=tau, rho=rho,
        max_iter=max_iter, early_stopping=False, debug=True, verbose=verbose,
    )
    _, _, test_loss = condatvu(**params)

    train_loss = train_loss[[0] + all_n_layers]
    test_loss = test_loss[[0] + all_n_layers]

    if verbose > 0:
        print(f"\r[Condat-Vu] iterations finished "
              f"train-loss={train_loss[-1]:.4e} test-loss={test_loss[-1]:.4e}")

    return train_loss, test_loss

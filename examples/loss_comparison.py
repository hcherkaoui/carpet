""" Compare the convergence rate for the synthesis/analysis 1d TV-l1 problem.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from prox_tv import tv1_1d
from carpet.datasets import synthetic_1d_dataset
from carpet.loss_gradient import analysis_obj
from utils import (lasso_like_tv, learned_lasso_like_tv, chambolle_tv,
                   learned_chambolle_tv, condatvu_tv)


def logspace_layers(n_layers=10, max_depth=50):
    """ Return n_layers, from 1 to max_depth of differents number of layers to
    define networks """
    all_n_layers = np.logspace(0, np.log10(max_depth), n_layers).astype(int)
    return list(np.unique(all_n_layers))


if __name__ == '__main__':

    print(__doc__)
    print('*' * 80)

    t0 = time.time()

    ploting_dir = 'outputs_plots'
    if not os.path.exists(ploting_dir):
        os.makedirs(ploting_dir)

    ###########################################################################
    # Define variables and data

    # Define variables
    n_samples = 2000
    n_samples_testing = 1000
    m = 10
    s = 0.2
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=10, max_depth=50)
    ticks_layers = np.array([0] + all_n_layers)
    lbda = 0.75

    seed = np.random.randint(0, 1000)
    print(f'Seed used = {seed}')  # noqa: E999

    # Generate data
    L = np.triu(np.ones((m, m)))
    D = (np.eye(m, k=-1) - np.eye(m, k=0))[:, :-1]
    x, _, z = synthetic_1d_dataset(D=L, n=n_samples, s=s, snr=snr, seed=seed)

    x_train = x[n_samples_testing:, :]
    x_test = x[:n_samples_testing, :]
    z_train = z[n_samples_testing:, :]
    z_test = z[:n_samples_testing, :]

    l1_z_train = np.sum(np.abs(z_train))
    l1_z_test = np.sum(np.abs(z_test))

    ###########################################################################
    # Main experiment

    names = [
             'learned-TV LISTA-Original',
             'learned-TV LISTA-Coupled',
             'learned-TV LISTA-Step',
             'learned-TV Condat-Vu-Coupled',
             'learned-TV Condat-Vu-Step',
             'learned-TV Chamb-Original',
             'learned-TV Chamb-Coupled',
             'learned-TV Chamb-Step',
             'TV ISTA-iterative',
             'TV FISTA-iterative',
             'TV Condat-Vu-iterative',
             'TV Chamb-iterative',
             'TV Fast-Chamb-iterative',
             ]
    funcs_bench = [
                   learned_lasso_like_tv,
                   learned_lasso_like_tv,
                   learned_lasso_like_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                   lasso_like_tv,
                   lasso_like_tv,
                   condatvu_tv,
                   chambolle_tv,
                   chambolle_tv,
                   ]
    l_type_ = [
               'lista',
               'coupled',
               'step',
               'coupledcondatvu',
               'stepcondatvu',
               'lchambolle',
               'coupledchambolle',
               'stepchambolle',
               'ista',
               'fista',
               None,
               'chambolle',
               'fast-chambolle',
               ]

    def _run_experiment(names, funcs_bench, l_type_, x_train, x_test, L, lbda,
                        all_n_layers):
        """ Experiment launcher. """
        print("=" * 80)

        l_train_loss, l_test_loss, l_train_reg, l_test_reg = [], [], [], []
        for name, func_bench, type_ in zip(names, funcs_bench, l_type_):
            print(f"[main script] running {name}")
            print("-" * 80)

            results = func_bench(x_train, x_test, D, L, lbda=lbda, type_=type_,
                                 all_n_layers=all_n_layers)
            train_loss, test_loss, train_reg, test_reg = results
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)
            l_train_reg.append(train_reg)
            l_test_reg.append(test_reg)

            print("=" * 80)

        return l_train_loss, l_test_loss, l_train_reg, l_test_reg

    run_experiment = Memory('__cache_dir__', verbose=0).cache(_run_experiment)

    results = run_experiment(names, funcs_bench, l_type_, x_train, x_test, L,
                             lbda, all_n_layers)
    l_train_loss, l_test_loss, l_train_reg, l_test_reg = results

    ###########################################################################
    # Plotting
    lw = 3
    eps_plots = 1.0e-10
    z_hat_train_star = np.c_[[tv1_1d(x_train_, lbda) for x_train_ in x_train]]
    z_hat_test_star = np.c_[[tv1_1d(x_test_, lbda) for x_test_ in x_test]]
    min_train_loss = analysis_obj(z_hat_train_star, D, x_train, lbda)
    min_test_loss = analysis_obj(z_hat_test_star, D, x_test, lbda)

    fig, l_axis = plt.subplots(nrows=2, sharex=True, figsize=(15, 10),
                               num=f"[{__file__}] Loss functions")
    axis_train, axis_test = l_axis

    for name, train_loss in zip(names, l_train_loss):
        marker = '^' if 'Chamb' in name else 'o'
        marker = 's' if 'Condat' in name else marker
        ls = 'dotted' if 'iterative' in name else 'solid'
        train_loss -= (min_train_loss - eps_plots)
        axis_train.loglog(ticks_layers + 1, train_loss, marker=marker, lw=lw,
                          ms=3*lw, ls=ls, label=name)
    axis_train.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                      borderaxespad=0.0, fontsize=15)
    axis_train.grid()
    axis_train.set_xlabel("Layers [-]", fontsize=15)
    axis_train.set_ylabel('$F(.) - F(z^*)$', fontsize=15)
    axis_train.set_title('Loss function comparison on training set',
                         fontsize=15)

    for name, test_loss in zip(names, l_test_loss):
        marker = '^' if 'Chamb' in name else 'o'
        marker = 's' if 'Condat' in name else marker
        ls = 'dotted' if 'iterative' in name else 'solid'
        test_loss -= (min_test_loss - eps_plots)
        axis_test.loglog(ticks_layers + 1, test_loss, marker=marker, lw=lw,
                         ms=3*lw, ls=ls, label=name)
    axis_test.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                     borderaxespad=0.0, fontsize=15)
    axis_test.grid()
    axis_test.set_xticks(ticks_layers + 1)
    axis_test.set_xticklabels(ticks_layers)
    axis_test.set_title('Loss function comparison on testing set', fontsize=15)

    axis_test.set_xlabel("Layers [-]", fontsize=15)
    axis_test.set_ylabel("$F(.) - F(z^*)$", fontsize=15)
    fig.tight_layout()

    filename = os.path.join(ploting_dir, "loss_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    fig, l_axis = plt.subplots(nrows=2, sharex=True, figsize=(15, 10),
                               num=f"[{__file__}] Reg. terms")
    axis_train, axis_test = l_axis

    for name, train_reg in zip(names, l_train_reg):
        marker = '^' if 'Chamb' in name else 'o'
        marker = 's' if 'Condat' in name else marker
        ls = 'dotted' if 'iterative' in name else 'solid'
        axis_train.loglog(ticks_layers + 1, train_reg, marker=marker, lw=lw,
                          ms=3*lw, ls=ls, label=name)
    axis_train.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                      borderaxespad=0.0, fontsize=15)
    axis_train.grid()
    axis_train.set_xlabel("Layers [-]", fontsize=15)
    axis_train.set_ylabel('$||z||_1$', fontsize=15)
    axis_train.set_title('Reg. term comparison on training set', fontsize=15)

    for name, test_reg in zip(names, l_test_reg):
        marker = '^' if 'Chamb' in name else 'o'
        marker = 's' if 'Condat' in name else marker
        ls = 'dotted' if 'iterative' in name else 'solid'
        axis_test.loglog(ticks_layers + 1, test_reg, marker=marker, lw=lw,
                         ms=3*lw, ls=ls, label=name)
    axis_test.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                     borderaxespad=0.0, fontsize=15)
    axis_test.grid()
    axis_test.set_xticks(ticks_layers + 1)
    axis_test.set_xticklabels(ticks_layers)
    axis_train.set_title('Reg. term comparison on testing set', fontsize=15)

    axis_test.set_xlabel("Layers [-]", fontsize=15)
    axis_test.set_ylabel("$||z||_1$", fontsize=15)
    fig.tight_layout()

    filename = os.path.join(ploting_dir, "reg_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

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
from carpet.utils import logspace_layers
from utils import (lasso_like_tv, learned_lasso_like_tv, chambolle_tv,
                   learned_chambolle_tv, condatvu_tv)


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
    n_samples = 1000
    n_samples_testing = 100
    m = 10
    s = 0.2
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=10, max_depth=100)
    ticks_layers = np.array([0] + all_n_layers)
    lbda = 0.75

    seed = np.random.randint(0, 1000)
    # seed = 113
    print(f'Seed used = {seed}')  # noqa: E999

    # Generate data
    L = np.triu(np.ones((m, m)))
    D = (np.eye(m, k=-1) - np.eye(m, k=0))[:, :-1]
    x, _, _ = synthetic_1d_dataset(D=L, n=n_samples, s=s, snr=snr, seed=seed)

    x_train = x[n_samples_testing:, :]
    x_test = x[:n_samples_testing, :]

    ###########################################################################
    # Main experiment

    names = [
             'LISTA original',
             'LISTA coupled',
             'LISTA step',
            #  'ISTA iterative',
             'FISTA-iterative',
             'learned-Condat-Vu',
             'Condat-Vu-iterative',
             'learned-TV Chamb-Original',
             'learned-TV Chamb-Coupled',
             'learned-TV Chamb-Step',
            #  'Chamb iterative',
            #  'Fast Chamb iterative',
             ]
    funcs_bench = [
                   learned_lasso_like_tv,
                   learned_lasso_like_tv,
                   learned_lasso_like_tv,
                   lasso_like_tv,
                #    lasso_like_tv,
                   learned_chambolle_tv,
                   condatvu_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                #    chambolle_tv,
                #    chambolle_tv,
                   ]
    l_type_ = [
               'lista',
               'coupled',
               'step',
            #    'ista',
               'fista',
               'condatvu',
               None,
               'lchambolle',
               'coupledchambolle',
               'stepchambolle',
            #    'chambolle',
            #    'fast-chambolle',
               ]

    def _run_experiment(names, funcs_bench, l_type_, x_train, x_test, L, lbda,
                        all_n_layers):
        """ Experiment launcher. """
        l_train_loss, l_test_loss = [], []
        print("=" * 80)
        for name, func_bench, type_ in zip(names, funcs_bench, l_type_):
            print(f"[main script] running {name}")
            print("-" * 80)
            A = D if ('Chamb' in name) or ('Condat' in name) else L
            train_loss, test_loss = func_bench(x_train, x_test, A,
                            lbda=lbda, type_=type_, all_n_layers=all_n_layers)
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)
            print("=" * 80)
        return l_train_loss, l_test_loss

    run_experiment = Memory('__cache_dir__', verbose=0).cache(_run_experiment)

    l_train_loss, l_test_loss = run_experiment(names, funcs_bench, l_type_,
                                               x_train, x_test, L, lbda,
                                               all_n_layers)

    ###########################################################################
    # Plotting
    lw = 3
    eps_plots = 1.0e-10
    z_hat_train_star = np.c_[[tv1_1d(x_train_, lbda) for x_train_ in x_train]]
    z_hat_test_star = np.c_[[tv1_1d(x_test_, lbda) for x_test_ in x_test]]
    min_train_loss = analysis_obj(z_hat_train_star, D, x_train, lbda)
    min_test_loss = analysis_obj(z_hat_test_star, D, x_test, lbda)

    fig, l_axis = plt.subplots(nrows=2, sharex=True, figsize=(12, 8),
                               num=f"[{__file__}] Loss functions")
    axis_train, axis_test = l_axis

    for name, train_loss in zip(names, l_train_loss):
        marker = '^' if 'Chamb' in name else 'o'
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

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

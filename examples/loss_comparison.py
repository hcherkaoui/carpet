""" Compare the convergence rate for the synthesis/analysis 1d TV-l1 problem.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from carpet.datasets import synthetic_1d_dataset
from carpet.checks import check_random_state
from utils import (lasso_like_tv, learned_lasso_like_tv, analysis_tv,
                   chambolle_tv, learned_chambolle_tv, condatvu_tv,
                   learned_analysis_taut_string)


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
    n_samples = 500
    n_samples_testing = 250
    n_atoms = 8
    n_dim = 5
    s = 0.2
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=10, max_depth=40)
    ticks_layers = np.array([0] + all_n_layers)
    lbda = 1.0

    seed = np.random.randint(0, 1000)
    rng = check_random_state(seed)
    print(f'Seed used = {seed}')  # noqa: E999

    # Generate data
    results = synthetic_1d_dataset(n_atoms=n_atoms, n_dim=n_dim, n=n_samples,
                                   s=s, snr=snr, seed=seed)
    x, _, z, L, D, A = results

    x_train = x[n_samples_testing:, :]
    x_test = x[:n_samples_testing, :]

    ###########################################################################
    # Main experiment
    methods = [
        # ('TV LISTA-Original', learned_lasso_like_tv, 'origista',
        #  'tab:orange', '*', 'solid'),
        # ('TV LISTA-Coupled', learned_lasso_like_tv, 'coupledista',
        #  'tab:orange', '^', 'solid'),
        # ('TV LISTA-Step', learned_lasso_like_tv, 'stepista',
        #  'tab:orange', 'o', 'solid'),
        # ('TV Condat-Vu-Coupled', learned_chambolle_tv, 'coupledcondatvu',
        #  'tab:green', '^', 'solid'),
        ('Analysis learned taut-string', learned_analysis_taut_string, None,
         'tab:red', '*', '-.'),
        ('TV TV-Original', learned_chambolle_tv, 'origtv',
         'tab:red', '*', 'solid'),
        # ('TV Chamb-Original', learned_chambolle_tv, 'origchambolle',
        #  'tab:blue', '*', 'solid'),
        # ('TV Chamb-Coupled', learned_chambolle_tv, 'coupledchambolle',
        #  'tab:blue', '^', 'solid'),
        # ('TV synthesis FISTA-iterative', lasso_like_tv, 'fista',
        #  'tab:orange', 's', 'dashed'),
        # ('TV Condat-Vu-iterative', condatvu_tv, None,
        #  'tab:green', 's', 'dashed'),
        # ('TV Fast-Chamb-iterative', chambolle_tv, 'fast-chambolle',
        #  'tab:blue', 's', 'dashed'),
        ('TV analysis ISTA-iterative', analysis_tv, 'ista',
         'tab:red', 's', 'dashed'),
        ('TV analysis FISTA-iterative', analysis_tv, 'fista',
         'tab:red', 's', 'dashed'),
    ]

    def _run_experiment(methods, x_train, x_test, L, lbda, all_n_layers):
        """ Experiment launcher. """
        print("=" * 80)

        l_train_loss, l_test_loss, l_train_reg, l_test_reg = [], [], [], []
        for name, func_bench, type_, _, _, _ in methods:
            print(f"[main script] running {name}")
            print("-" * 80)

            results = func_bench(x_train, x_test, A, D, L, lbda=lbda,
                                 type_=type_, all_n_layers=all_n_layers)
            train_loss, test_loss, train_reg, test_reg = results
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)
            l_train_reg.append(train_reg)
            l_test_reg.append(test_reg)

            print("=" * 80)

        return l_train_loss, l_test_loss, l_train_reg, l_test_reg

    run_experiment = Memory('__cache_dir__', verbose=0).cache(_run_experiment)

    results = run_experiment(methods, x_train, x_test, L, lbda, all_n_layers)
    l_train_loss, l_test_loss, l_train_reg, l_test_reg = results

    ###########################################################################
    # Plotting
    lw = 3
    eps_plots = 1.0e-15
    print("[Reference] computing minimum reference loss...")
    t0_ref_ = time.time()
    results = analysis_tv(x_train, x_test, A, D, L, lbda=lbda, type_='fista',
                          all_n_layers=[1000], verbose=0)
    ref_train_loss, ref_test_loss, _, _ = results
    min_train_loss, min_test_loss = ref_train_loss[-1], ref_test_loss[-1]
    print(f"[Reference] ({time.time() - t0_ref_:.3}s) "
          f"train-loss={min_train_loss:.6e} test-loss={min_test_loss:.6e}")

    fig, l_axis = plt.subplots(nrows=2, sharex=True, figsize=(15, 20),
                               num=f"[{__file__}] Loss functions")
    axis_train, axis_test = l_axis

    for method, train_loss in zip(methods, l_train_loss):
        name, _, _, color, marker, ls = method
        train_loss -= (min_train_loss - eps_plots)
        axis_train.loglog(ticks_layers + 1, train_loss, marker=marker,
                          color=color, ls=ls, lw=lw, ms=3*lw, label=name)
    axis_train.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                      borderaxespad=0.0, fontsize=15)
    axis_train.grid()
    axis_train.set_xlabel("Layers [-]", fontsize=15)
    axis_train.set_ylabel('$F(.) - F(z^*)$', fontsize=15)
    axis_train.set_title('Loss function comparison on training set',
                         fontsize=18)

    for method, test_loss in zip(methods, l_test_loss):
        name, _, _, color, marker, ls = method
        test_loss -= (min_test_loss - eps_plots)
        axis_test.loglog(ticks_layers + 1, test_loss, marker=marker,
                         color=color, ls=ls, lw=lw, ms=3*lw, label=name)
    axis_test.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                     borderaxespad=0.0, fontsize=15)
    axis_test.grid()
    axis_test.set_xlabel("Layers [-]", fontsize=15)
    axis_test.set_ylabel("$F(.) - F(z^*)$", fontsize=15)
    axis_test.set_title('Loss function comparison on testing set', fontsize=18)

    axis_test.set_xticks(ticks_layers + 1)
    axis_test.set_xticklabels(ticks_layers)

    fig.tight_layout()

    filename = os.path.join(ploting_dir, "loss_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

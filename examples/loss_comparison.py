""" Compare the convergence rate for the synthesis/analysis 1d TV-l1 problem.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from carpet.datasets import synthetic_1d_dataset
from carpet.checks import check_random_state
from utils import (synthesis_learned_algo, analysis_learned_algo,  # noqa: F401
                   analysis_learned_taut_string, synthesis_iter_algo,
                   analysis_primal_iter_algo, analysis_dual_iter_algo,
                   analysis_primal_dual_iter_algo)


def logspace_layers(n_layers=10, max_depth=50):
    """ Return n_layers, from 1 to max_depth of different number of layers to
    define networks """
    all_n_layers = np.logspace(0, np.log10(max_depth), n_layers).astype(int)
    return list(np.unique(all_n_layers))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Run comparison between the different learned algorithms '
        'to solve a TV regularized regression.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Use GPU <gpu> to run the computations. If it is '
                        'not set, use CPU computations.')
    parser.add_argument('--max-iter', type=int, default=300,
                        help='Max number of iterations to train the '
                        'learnable networks.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    args = parser.parse_args()

    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = 'cpu'

    print(__doc__)
    print('*' * 80)

    t0 = time.time()

    ploting_dir = 'outputs_plots'
    if not os.path.exists(ploting_dir):
        os.makedirs(ploting_dir)

    ###########################################################################
    # Define variables and data

    # Define variables
    n_samples = 200
    n_samples_testing = 100
    n_atoms = 8
    n_dim = 5
    s = 0.2
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=10, max_depth=40)
    ticks_layers = np.array([0] + all_n_layers)
    lbda = 1.0

    seed = np.random.randint(0, 1000)
    if args.seed is not None:
        seed = args.seed
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
    methods = [  # can be commented in one #
        ('Synthesis LISTA-Original', synthesis_learned_algo, 'origista',
         'tab:orange', '*', 'solid'),
        # ('Synthesis LISTA-Coupled', synthesis_learned_algo, 'coupledista',
        #  'tab:orange', '^', 'solid'),
        # ('Synthesis LISTA-Step', synthesis_learned_algo, 'stepista',
        #  'tab:orange', 'o', 'solid'),
        # ('Analysis Condat-Vu-Coupled', analysis_learned_algo,
        #  'coupledcondatvu', 'tab:green', '^', 'solid'),
        ('Analysis learned taut-string', analysis_learned_taut_string, None,
         'tab:red', '*', '-.'),
        ('Analysis TV-Original', analysis_learned_algo, 'origtv',
         'tab:red', '*', 'solid'),
        # ('Analysis Chamb-Original', analysis_learned_algo, 'origchambolle',
        #  'tab:blue', '*', 'solid'),
        # ('Analysis Chamb-Coupled', analysis_learned_algo, 'coupledchambolle',
        #  'tab:blue', '^', 'solid'),
        ('Synthesis FISTA-iterative', synthesis_iter_algo, 'ista',
         'tab:orange', 's', 'dashed'),
        ('Synthesis FISTA-iterative', synthesis_iter_algo, 'fista',
         'tab:orange', 's', 'dashed'),
        # ('Analysis Condat-Vu-iterative', analysis_primal_dual_iter_algo,
        #   None, 'tab:green', 's', 'dashed'),
        # ('Analysis Chamb-iterative', analysis_dual_iter_algo, 'chambolle',
        #  'tab:blue', 's', 'dashed'),
        # ('Analysis  Fast-Chamb-iterative', analysis_dual_iter_algo,
        #  'fast-chambolle', 'tab:blue', 's', 'dashed'),
        ('Analysis ISTA-iterative', analysis_primal_iter_algo, 'ista',
         'tab:red', 's', 'dashed'),
        ('Analysis FISTA-iterative', analysis_primal_iter_algo, 'fista',
         'tab:red', 's', 'dashed'),
    ]

    def run_experiment(methods, x_train, x_test, L, lbda, all_n_layers):
        """ Experiment launcher. """
        print("=" * 80)

        l_train_loss, l_test_loss, l_train_reg, l_test_reg = [], [], [], []
        for name, func_bench, type_, _, _, _ in methods:
            print(f"[main script] running {name}")
            print("-" * 80)

            results = func_bench(x_train, x_test, A, D, L, lbda=lbda,
                                 type_=type_, all_n_layers=all_n_layers,
                                 device=device, max_iter=args.max_iter)
            train_loss, test_loss, train_reg, test_reg = results
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)
            l_train_reg.append(train_reg)
            l_test_reg.append(test_reg)

            print("=" * 80)

        return l_train_loss, l_test_loss, l_train_reg, l_test_reg

    results = run_experiment(methods, x_train, x_test, L, lbda, all_n_layers)
    l_train_loss, l_test_loss, l_train_reg, l_test_reg = results

    ###########################################################################
    # Plotting
    lw = 4
    eps_plots = 1.0e-20
    print("[Reference] computing minimum reference loss...")
    t0_ref_ = time.time()
    results = analysis_primal_iter_algo(x_train, x_test, A, D, L, lbda=lbda,
                                        type_='fista', all_n_layers=[1000],
                                        verbose=0)
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
    title_ = f'Loss function comparison on training set (seed={seed})'
    axis_train.set_title(title_, fontsize=18)

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
    title_ = f'Loss function comparison on testing set (seed={seed})'
    axis_test.set_title(title_, fontsize=18)

    axis_test.set_xticks(ticks_layers + 1)
    axis_test.set_xticklabels(ticks_layers)

    fig.tight_layout()

    filename = os.path.join(ploting_dir, "loss_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

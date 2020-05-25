""" Compare the convergence rate for the synthesis/analysis 1d TV-l1 problem.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import os
import argparse
import time
from joblib import Memory
import matplotlib.pyplot as plt
import numpy as np
from carpet import LearnTVAlgo
from carpet.datasets import synthetic_1d_dataset
from carpet.checks import check_random_state
from carpet.metrics import compute_prox_tv_errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run comparison between the different learned algorithms '
        'to solve a TV regularized regression.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Use GPU <gpu> to run the computations. If it is '
                        'not set, use CPU computations.')
    parser.add_argument('--max-iter', type=int, default=300,
                        help='Max number of iterations to train the '
                        'learnable networks.')
    parser.add_argument('--n-layers', type=int, default=50,
                        help='Max number of layers to define the '
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

    ploting_dir = f'outputs_plots_sub_optim_comp_n_layers_{args.n_layers}'
    if not os.path.exists(ploting_dir):
        os.makedirs(ploting_dir)

    ###########################################################################
    # Define variables and data

    # Define variables
    n_samples = 100
    n_samples_testing = 50
    n_atoms = 8
    n_dim = 5
    s = 0.2
    snr = 0.0
    n_layers = args.n_layers
    lbda = 0.5

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
        ('Analysis TV-Original', 'origtv',
         dict(n_inner_layer=20), 'tab:orange', '', '--'),
        ('Analysis TV-Original', 'origtv',
         dict(n_inner_layer=200), 'tab:orange', '', 'solid'),
        ('Analysis untrained-TV-Original', 'untrained-origtv',
         dict(n_inner_layer=20), 'tab:blue', '', '--'),
        ('Analysis untrained-TV-Original', 'untrained-origtv',
         dict(n_inner_layer=200), 'tab:blue', '', 'solid'),
    ]

    memory = Memory('__cache_dir__', verbose=0)

    @memory.cache
    def run_experiment(methods, x_train, x_test, A, lbda, n_layers):
        """ Experiment launcher. """
        print("=" * 80)

        l_diff_loss = []
        for name, type_, kwargs, _, _, _ in methods:
            print(f"[main script] running {name}")
            print("-" * 80)

            algo_type = 'origtv' if ('untrained' in type_) else type_
            network = LearnTVAlgo(algo_type=algo_type, A=A, n_layers=n_layers,
                                  max_iter=args.max_iter, device=device,
                                  verbose=1, **kwargs)

            if 'untrained' not in type_:
                network.fit(x_train, lbda=lbda)

            diff_loss = compute_prox_tv_errors(network, x_test, lbda)
            l_diff_loss.append(diff_loss)

            print("=" * 80)

        return l_diff_loss

    l_diff_loss = run_experiment(methods, x_train, x_test, A, lbda, n_layers)

    ###########################################################################
    # Plotting
    lw = 4
    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(10, 5),
                           num=f"[{__file__}] Prox-TV sub-optimality")

    for method, diff_loss in zip(methods, l_diff_loss):
        name, _, _, color, marker, ls = method
        ax.semilogy(diff_loss, marker=marker, color=color, ls=ls, lw=lw,
                    ms=3*lw, label=name)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0,
              fontsize=12)
    ax.grid()
    ax.set_xlabel("Layers [-]", fontsize=12)
    ax.set_ylabel("$G(.) - G(z^*)$", fontsize=12)
    title_ = f'Prox-TV sub-optimality comparison\non testing set (seed={seed})'
    ax.set_title(title_, fontsize=15)

    plt.tight_layout()

    filename = os.path.join(ploting_dir, "sub_optimality_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

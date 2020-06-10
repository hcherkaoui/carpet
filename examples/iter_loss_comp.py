""" Compare the convergence rate of the iterative solvers for the synthesis /
analysis 1d TV-l1 problem.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import os
import shutil
import json
import time

import matplotlib as mpl
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amssymb}']
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 18

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from carpet.datasets import synthetic_1d_dataset  # noqa: E402
from carpet.checks import check_random_state  # noqa: E402
from utils import (synthesis_learned_algo,  # noqa: F401, E402
                   analysis_learned_algo, analysis_learned_taut_string,
                   synthesis_iter_algo, analysis_primal_iter_algo,
                   analysis_dual_iter_algo, analysis_primal_dual_iter_algo)


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
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Max number of iterations to train the '
                        'learnable networks.')
    parser.add_argument('--temp-reg', type=float, default=0.5,
                        help='Temporal regularisation parameter.')
    parser.add_argument('--plots-dir', type=str, default='outputs',
                        help='Outputs directory for plots.')
    parser.add_argument('--iter-mult', type=float, default='2.0',
                        help='Multiplicative coefficient to obtain the number'
                        ' of iteration.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    args = parser.parse_args()

    print(__doc__)
    print('*' * 80)

    t0 = time.time()

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    filename = os.path.join(args.plots_dir, 'command_line.json')
    print(f"Archiving '{filename}' under '{args.plots_dir}'")
    with open(filename, 'w') as jsonfile:
        json.dump(args._get_kwargs(), jsonfile)

    print("archiving '{0}' under '{1}'".format(__file__, args.plots_dir))
    shutil.copyfile(__file__, os.path.join(args.plots_dir, __file__))

    ###########################################################################
    # Define variables and data

    # Define variables
    n_samples = 1000 + 1  # training samples can't be 0
    n_samples_testing = n_samples - 1
    n_atoms = 40
    n_dim = 40
    s = 0.1
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=10, max_depth=args.max_iter)
    ticks_layers = np.array([0] + all_n_layers)

    seed = args.seed if args.seed is not None else np.random.randint(0, 1000)
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
        ('Synthesis primal PGD', synthesis_iter_algo, 'ista', dict(),
         'tab:blue', 's', 'dashed'),
        ('Synthesis primal APGD', synthesis_iter_algo, 'fista', dict(),
         'tab:blue', 's', 'solid'),
        ('Analysis primal PGD', analysis_primal_iter_algo, 'ista', dict(),
         'tab:orange', 's', 'dashed'),
        ('Analysis primal APGD', analysis_primal_iter_algo, 'fista', dict(),
         'tab:orange', 's', 'solid'),
        ('Analysis dual PGD', analysis_dual_iter_algo, 'ista', dict(),
         'tab:red', 's', 'dashed'),
        ('Analysis dual APGD', analysis_dual_iter_algo, 'fista', dict(),
         'tab:red', 's', 'solid'),
        ('Analysis primal-dual GD', analysis_primal_dual_iter_algo, 'fista',
         dict(), 'tab:olive', 's', 'solid'),
    ]

    def run_experiment(methods, x_train, x_test, L, lbda, all_n_layers):
        """ Experiment launcher. """
        print("=" * 80)

        l_train_loss, l_test_loss = [], []
        for name, func_bench, type_, net_kwargs, _, _, _ in methods:
            print(f"[main script] running {name}")
            print("-" * 80)

            results = func_bench(x_train, x_test, A, D, L, lbda=lbda,
                                 type_=type_, all_n_layers=all_n_layers,
                                 device='cpu', max_iter=int(args.max_iter),
                                 net_kwargs=net_kwargs)
            train_loss, test_loss = results
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)

            print("=" * 80)

        return l_train_loss, l_test_loss

    results = run_experiment(methods, x_train, x_test, L, args.temp_reg,
                             all_n_layers)
    l_train_loss, l_test_loss = results

    ###########################################################################
    # Plotting
    lw = 5
    eps_plots = 1.0e-20
    print("[Reference] computing minimum reference loss...")
    ref_n_layers = [int(args.iter_mult * args.max_iter)]
    t0_ref_ = time.time()
    results = analysis_primal_iter_algo(x_train, x_test, A, D, L,
                                        lbda=args.temp_reg, type_='fista',
                                        all_n_layers=ref_n_layers, verbose=0)
    ref_train_loss, ref_test_loss = results
    min_train_loss, min_test_loss = ref_train_loss[-1], ref_test_loss[-1]
    print(f"[Reference] ({time.time() - t0_ref_:.1f}s) "
          f"train-loss={min_train_loss:.6e} test-loss={min_test_loss:.6e}")

    fig, l_axis = plt.subplots(nrows=1, sharex=True, figsize=(8, 5),
                               num=f"[{__file__}] Loss functions")
    axis_test = l_axis

    for method, test_loss in zip(methods, l_test_loss):
        name, _, _, _, color, marker, ls = method
        test_loss -= (min_test_loss - eps_plots)
        mask = test_loss > 1e-8
        axis_test.loglog((ticks_layers + 1)[mask], test_loss[mask],
                         marker=marker, color=color, ls=ls, lw=lw, ms=2*lw,
                         label=name)

    axis_test.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), loc="lower left",
                     mode="expand", borderaxespad=0, ncol=2, fontsize=17)
    axis_test.grid()
    axis_test.set_xlabel("Iterations $t$")
    axis_test.set_ylabel(
        r'$\mathbb E \left[P_x(u^{(t)}) - P_x(u^{*}) \right]$'
        )
    axis_test.set_xticks(ticks_layers + 1)
    axis_test.set_xticklabels(ticks_layers)

    fig.tight_layout()

    filename = os.path.join(args.plots_dir, "loss_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

""" Compare the convergence rate for the synthesis/analysis 1d TV-l1 problem.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import os
import time
import pathlib
import numpy as np
import pandas as pd
from joblib import Memory
import matplotlib.pyplot as plt


from carpet.utils import init_vuz
from carpet.datasets import synthetic_1d_dataset
from carpet import ListaTV, LpgdTautString, CoupledIstaLASSO
from carpet.iterative_solver import IstaAnalysis, IstaSynthesis
from carpet.loss_gradient import analysis_primal_obj, tv_reg


OUTPUT_DIR = pathlib.Path('outputs_plots')
SCRIPT_NAME, _ = os.path.splitext(os.path.basename(__file__))


memory = Memory('__cache_dir__', verbose=0)


def logspace_layers(n_layers=10, max_depth=50):
    """ Return n_layers, from 1 to max_depth of different number of layers to
    define networks """
    all_n_layers = np.logspace(0, np.log10(max_depth), n_layers).astype(int)
    return list(np.unique(all_n_layers))


###########################################################################
# Main experiment runner

@memory.cache(ignore=['verbose', 'device'])
def run_one(x_train, x_test, A, D, L, lbda, network, all_n_layers,
            extra_args, meta={}, device=None, verbose=1):
    params = None
    log = []

    def record_loss(u_train, u_test, n_layers):
        log.append(dict(
            n_layers=n_layers, **meta,
            train_loss=analysis_primal_obj(u_train, A, D, x_train, lbda),
            test_loss=analysis_primal_obj(u_test, A, D, x_test, lbda),
            train_reg=tv_reg(u_train, D),
            test_reg=tv_reg(u_test, D)
        ))

    _, u0_train, _ = init_vuz(A, D, x_train, lbda)
    _, u0_test, _ = init_vuz(A, D, x_test, lbda)
    record_loss(u0_train, u0_test, n_layers=0)

    for i, n_layers in enumerate(all_n_layers):

        # declare network
        algo = network(A=A, n_layers=n_layers, initial_parameters=params,
                       **extra_args, device=device, verbose=verbose)

        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        u_train = algo.transform_to_u(x_train, lbda)
        u_test = algo.transform_to_u(x_test, lbda)
        record_loss(u_train, u_test, n_layers=n_layers)

        if verbose > 0:
            train_loss = log[i]['train_loss']
            test_loss = log[i]['test_loss']
            print(f"\r[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s train-loss={train_loss:.4e} "
                  f"test-loss={test_loss:.4e}")

    return log


def run_experiment(max_iter, max_iter_ref=1000, device=None, seed=None,
                   net_solver_type='greedy'):
    # Define variables
    n_samples_train = 1000
    n_samples_testing = 1000
    n_samples = n_samples_train + n_samples_testing
    n_atoms = 8
    n_dim = 5
    s = 0.2
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=10, max_depth=40)
    lbda = 1.0

    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f'Seed used = {seed}')

    # Store meta data of the problem
    meta_pb = dict(n_atoms=n_atoms, n_dim=n_dim, s=s, snr=snr, seed=seed,
                   n_samples_train=n_samples_train,
                   n_samples_testing=n_samples_testing)

    # Generate data
    x, _, z, L, D, A = synthetic_1d_dataset(
        n_atoms=n_atoms, n_dim=n_dim, n=n_samples, s=s, snr=snr, seed=seed
    )

    x_train = x[n_samples_testing:, :]
    x_test = x[:n_samples_testing, :]

    learning_parameters = dict(
        net_solver_type=net_solver_type, max_iter=max_iter
    )

    methods = {
        'lista_synthesis': {
            'label': 'Synthesis LISTA',
            'network': CoupledIstaLASSO,
            'extra_args': dict(**learning_parameters),
            'style': dict(color='tab:orange', marker='*', linestyle='-')
        },
        'lpgd_taut': {
            'label': 'Analysis LPGD - taut-string',
            'network': LpgdTautString,
            'extra_args': dict(**learning_parameters),
            'style': dict(color='tab:red', marker='*', linestyle='-.')
        },
        'lpgd_lista': {
            'label': 'Analysis LPGD - LISTA',
            'network': ListaTV,
            'extra_args': dict(n_inner_layers=100, **learning_parameters),
            'style': dict(color='tab:red', marker='*', linestyle='-')
        },
        'ista_synthesis': {
            'label': 'Synthesis ISTA',
            'network': IstaSynthesis,
            'extra_args': dict(momentum=None),
            'style': dict(color='tab:orange', marker='s', linestyle='--')
        },
        'fista_synthesis': {
            'label': 'Synthesis FISTA',
            'network': IstaSynthesis,
            'extra_args': dict(momentum='fista'),
            'style': dict(color='tab:orange', marker='*', linestyle='--')
        },
        'ista_analysis': {
            'label': 'Analysis ISTA',
            'network': IstaAnalysis,
            'extra_args': dict(momentum=None),
            'style': dict(color='tab:red', marker='s', linestyle='--')
        },
        'fista_analysis': {
            'label': 'Analysis FISTA',
            'network': IstaAnalysis,
            'extra_args': dict(momentum='fista'),
            'style': dict(color='tab:red', marker='*', linestyle='--')
        }
    }

    # launch all experiments
    print("=" * 80)
    t0 = time.time()
    log = []
    for m in methods.values():
        print(f"[main script] running {m['label']}")
        print("-" * 80)

        meta = meta_pb.copy()
        meta['label'] = m['label']
        meta['style'] = m['style']
        results = run_one(x_train, x_test, A, D, L, lbda=lbda,
                          network=m['network'], extra_args=m['extra_args'],
                          all_n_layers=all_n_layers, device=device,
                          meta=meta)
        log.extend(results)

    # Compute reference cost
    m = methods['fista_analysis']
    meta = meta_pb.copy()
    meta['label'] = 'reference'
    meta['style'] = None
    results = run_one(x_train, x_test, A, D, L, lbda=lbda,
                      network=m['network'], extra_args=m['extra_args'],
                      all_n_layers=[max_iter_ref], device=device,
                      meta=meta)
    log.extend(results[-1:])

    df = pd.DataFrame(log)
    df.to_pickle(OUTPUT_DIR / f'{SCRIPT_NAME}.pkl')

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("=" * 80)
    print("Script runs in: {}".format(delta_t))


###########################################################################
# Plotting

def plot_results(filename=None):

    if filename is None:
        filename = OUTPUT_DIR / f'{SCRIPT_NAME}.pkl'

    df = pd.read_pickle(filename)

    lw = 4
    eps_plots = 1.0e-20

    ref = df.query("label == 'reference'")
    df = df.query("label != 'reference'")
    assert ref.shape[0] == 1, "There should be only one reference"
    ref = ref.iloc[0]
    seed = df.iloc[0]['seed']

    fig, l_axis = plt.subplots(nrows=2, sharex=True, figsize=(15, 20),
                               num=f"[{SCRIPT_NAME}] Loss functions")
    axis_train, axis_test = l_axis

    c_star = ref.train_loss - eps_plots
    c_star_test = ref.test_loss - eps_plots
    for method in df['label'].unique():
        this_loss = df.query("label == @method")
        style = this_loss.iloc[0]['style']
        ticks_layers = this_loss['n_layers']
        train_loss = this_loss.train_loss - c_star
        axis_train.loglog(ticks_layers + 1, train_loss, **style, lw=lw,
                          ms=3*lw, label=method)
        test_loss = this_loss.test_loss - c_star_test
        axis_test.loglog(ticks_layers + 1, test_loss, **style, lw=lw,
                         ms=3*lw, label=method)

    # Formatting train loss
    axis_train.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                      borderaxespad=0.0, fontsize=15)
    axis_train.grid()
    axis_train.set_xlabel("Layers [-]", fontsize=15)
    axis_train.set_ylabel('$F(.) - F(z^*)$', fontsize=15)
    title_ = f'Loss function comparison on training set (seed={seed})'
    axis_train.set_title(title_, fontsize=18)

    # Formatting test loss
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

    filename = OUTPUT_DIR / f"{SCRIPT_NAME}.pdf"
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    plt.show()


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
    parser.add_argument('--plot', action='store_true',
                        help='Only plot the results of a previous run.')
    args = parser.parse_args()

    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = 'cpu'

    print(__doc__)
    print('*' * 80)

    # Make sure the output folder exists
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    if not args.plot:
        run_experiment(max_iter=args.max_iter, device=device, seed=args.seed,
                       net_solver_type='recursive')
    plot_results()

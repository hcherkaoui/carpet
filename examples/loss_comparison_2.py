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
from datetime import datetime
import matplotlib.pyplot as plt

from joblib import Memory, Parallel, delayed


from carpet.datasets import synthetic_1d_dataset
from carpet.metrics import compute_prox_tv_errors
from carpet.loss_gradient import analysis_primal_obj, tv_reg
from carpet import ListaTV, LpgdTautString, CoupledIstaLASSO  # noqa: F401
from carpet.iterative_solver import IstaAnalysis, IstaSynthesis


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
def run_one(x_train, x_test, A, D, L, lmbd, network, all_n_layers, key,
            extra_args, meta={}, device=None, verbose=1):
    params = None
    log = []
    print(f"[main script] running {key}")
    print("-" * 80)

    def record_loss(n_layers, algo):
        u_train = algo.transform_to_u(x_train, lmbd)
        u_test = algo.transform_to_u(x_test, lmbd)

        if isinstance(algo, ListaTV):
            prox_tv_loss_train = compute_prox_tv_errors(algo, x_train, lmbd)
            prox_tv_loss_test = compute_prox_tv_errors(algo, x_test, lmbd)
        else:
            prox_tv_loss_train = prox_tv_loss_test = None
        log.append(dict(
            key=key, **meta, lmbd=lmbd, extra_args=extra_args,
            n_layers=n_layers,
            train_loss=analysis_primal_obj(u_train, A, D, x_train, lmbd),
            test_loss=analysis_primal_obj(u_test, A, D, x_test, lmbd),
            train_reg=tv_reg(u_train, D),
            test_reg=tv_reg(u_test, D),
            prox_tv_loss_train=prox_tv_loss_train,
            prox_tv_loss_test=prox_tv_loss_test
        ))

    algo = network(A=A, n_layers=0)
    record_loss(n_layers=0, algo=algo)

    for i, n_layers in enumerate(all_n_layers):

        # declare network
        algo = network(A=A, n_layers=n_layers, initial_parameters=params,
                       **extra_args, device=device, verbose=verbose)

        t0_ = time.time()
        algo.fit(x_train, lmbd)
        delta_ = time.time() - t0_

        # save parameters
        params = algo.export_parameters()

        # get train and test error
        record_loss(n_layers=n_layers, algo=algo)

        if verbose > 0:
            train_loss = log[i+1]['train_loss']
            test_loss = log[i+1]['test_loss']
            print(f"\r[{algo.name}|layers#{n_layers:3d}] model fitted "
                  f"{delta_:4.1f}s train-loss={train_loss:.4e} "
                  f"test-loss={test_loss:.4e}")

    return log


def run_experiment(max_iter, max_iter_ref=1000, lmbd=.1, seed=None,
                   net_solver_type='recursive', n_jobs=1, device=None):
    # Define variables
    n_samples_train = 1000
    n_samples_testing = 1000
    n_samples = n_samples_train + n_samples_testing
    n_atoms = 8
    n_dim = 5
    s = 0.2
    snr = 0.0

    # Layers that are sampled
    all_n_layers = logspace_layers(n_layers=10, max_depth=40)

    timestamp = datetime.now()

    print(__doc__)
    print('*' * 80)
    print(f"Script started on: {timestamp.strftime('%Y/%m/%d %Hh%M')}")

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
        },
        # reference cost, use all_n_layers to override the computations
        'reference': {
            'label': 'Analysis FISTA',
            'network': IstaAnalysis,
            'extra_args': dict(momentum='fista'),
            'style': dict(color='tab:red', marker='*', linestyle='--'),
            'all_n_layers': [max_iter_ref]
        }
    }

    # for i, learn_prox in enumerate(['none', 'global', 'per-layer']):
    for i, learn_prox in enumerate(['none', 'per-layer']):
        # for n_inner_layer, marker in [(10, '*'), (50, 's'), (100, 'h'),
        #                               (300, 'o'), (500, '>')]:
        for n_inner_layer, marker in [(50, 's'), (20, '*')]:
            methods[f'lpgd_lista_{learn_prox}_{n_inner_layer}'] = {
                'label': f'LPGD - LISTA[{learn_prox}-{n_inner_layer}]',
                'network': ListaTV,
                'extra_args': dict(n_inner_layers=n_inner_layer,
                                   learn_prox=learn_prox,
                                   **learning_parameters),
                'style': dict(color=f'C{i}', marker=marker, linestyle='-')
            }

    # launch all experiments
    print("=" * 80)
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(x_train, x_test, A, D, L, lmbd=lmbd, key=k,
                         network=m['network'], extra_args=m['extra_args'],
                         all_n_layers=m.get('all_n_layers', all_n_layers),
                         device=device, meta=meta_pb)
        for k, m in methods.items()
    )

    # concatenate all results as a big list. Also update style and label
    # here to avoid recomputing the results when changing the style only.
    log = []
    for records in results:
        for rec in records:
            k = rec['key']
            rec.update(style=methods[k]['style'], label=methods[k]['label'])
            log.append(rec)

    # Save the computations in a pickle file
    df = pd.DataFrame(log)
    t_tag = timestamp.strftime('%Y-%m-%d_%Hh%M')
    tag = f'{t_tag}_{lmbd}_{seed}'
    df.to_pickle(OUTPUT_DIR / f'{SCRIPT_NAME}_{tag}.pkl')

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("=" * 80)
    print("Script runs in: {}".format(delta_t))


###########################################################################
# Plotting

def plot_results(filename=None):

    if filename is None:
        all_files = sorted(OUTPUT_DIR.glob('*.pkl'))
        filename = all_files[-1]

    df = pd.read_pickle(filename)

    lw = 4
    eps_plots = 1.0e-20

    ref = df.query("key == 'reference'")
    df = df.query("key != 'reference'")
    ref = ref.loc[ref['n_layers'].idxmax()]
    seed = df.iloc[0]['seed']
    lmbd = df.iloc[0]['lmbd']

    fig, l_axis = plt.subplots(nrows=2, sharex=True, figsize=(15, 20),
                               num=f"[{SCRIPT_NAME}] Loss functions")
    axis_train, axis_test = l_axis

    c_star = ref.train_loss - eps_plots
    c_star_test = ref.test_loss - eps_plots
    for method in df['label'].unique():
        this_loss = df.query("label == @method")
        style = this_loss.iloc[0]['style']
        if ':' in style['color']:
            style['color'] = style['color'].split(':')[1]
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
    title_ = (f'Loss function comparison on training set (seed={seed},'
              fr' $\lambda = {lmbd}$)')
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

    filename = filename.with_suffix('.pdf')
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    fig = plt.figure()
    handles = {'Trained': '', 'Untrained': ''}
    for i, nl in enumerate([20, 50]):
        handles[f'{nl} inner layers'] = plt.Line2D([], [], color=f'C{i}')
        for ls, learn, label in [('--', 'none', 'Untrained'),
                                 ('-', 'per-layer', 'Trained')]:
            handles[label] = plt.Line2D([], [], color='k', ls=ls)
            curve = []
            method = df.query(f"key == 'lpgd_lista_{learn}_{nl}'")
            for _, v in method.iterrows():
                prox_tv_loss = v[f'prox_tv_loss_test']
                if len(prox_tv_loss) == 0:
                    continue
                curve.append((v['n_layers'], prox_tv_loss[-1]))
                layers = np.arange(len(prox_tv_loss)) + 1
                plt.plot(layers, prox_tv_loss, color=f'C{i}', ls=ls, alpha=.1)
            curve = np.array(curve).T
            plt.loglog(curve[0], curve[1], label=v['key'],
                       color=f'C{i}', ls=ls)
    plt.xlabel('Layers [-]')
    plt.ylabel('$P(z^{(t)}) - P(z^*)$')
    plt.grid()
    plt.xlim(.9, 40)
    plt.legend(handles.values(), handles.keys(), ncol=2,
               loc='lower left', bbox_to_anchor=(-0.08, 1, 1, .05))
    filename = filename.with_name(
        filename.name.replace('.pdf', '_comparison_prox_tv.pdf')
    )
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
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Use <n-jobs> workers to run the computations.')
    parser.add_argument('--max-iter', type=int, default=300,
                        help='Max number of iterations to train the '
                        'learnable networks.')
    parser.add_argument('--lmbd', type=float, default=None,
                        help='Set the regularisation parameter for the '
                        'experiment.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--solver', type=str, default='recursive',
                        help='Set the solver for training the networks.')
    parser.add_argument('--plot', type=str, default=None,
                        help='Only plot the results of a previous run.')
    args = parser.parse_args()

    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = 'cpu'

    # Make sure the output folder exists
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    if args.plot is None:
        run_experiment(max_iter=args.max_iter, lmbd=args.lmbd, seed=args.seed,
                       net_solver_type=args.solver, n_jobs=args.n_jobs,
                       device=device)
        plot_results()
    else:
        plot_results(pathlib.Path(args.plot))

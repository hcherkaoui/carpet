""" Compare the convergence rate for the synthesis 1d TV-l1 problem. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from carpet.datasets import synthetic_1d_dataset
from carpet.synthesis_loss_gradient import obj
from carpet.lista import Lista
from carpet.utils import logspace_layers
from utils import ista_like_synth_tv, lista_like_synth_tv


if __name__ == '__main__':

    print(__doc__)
    print('*' * 80)

    t0 = time.time()

    ploting_dir = 'outputs_plots'
    if not os.path.exists(ploting_dir):
        os.makedirs(ploting_dir)

    # Define variables
    n_samples = 3000
    n_samples_testing = 1000
    m = 20
    s = 0.2
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=10, max_depth=100)
    ticks_layers = np.array([0] + all_n_layers)
    lbda = 0.5

    seed = np.random.randint(0, 1000)
    print(f'Seed used = {seed}')  # noqa: E999

    # Generate data
    D = np.triu(np.ones((m, m)))
    x, _, _ = synthetic_1d_dataset(D=D, n=n_samples, s=s, snr=snr, seed=seed)

    x_train = x[n_samples_testing:, :]
    x_test = x[:n_samples_testing, :]

    l_train_loss, l_test_loss = [], []
    names = ['ISTA-neural-net', 'LISTA-original', 'LISTA-coupled',
             'LISTA-step', 'ISTA-iterative', 'FISTA-iterative']
    funcs_bench = [lista_like_synth_tv, lista_like_synth_tv,
                   lista_like_synth_tv, lista_like_synth_tv,
                   ista_like_synth_tv, ista_like_synth_tv]
    l_type_ = ['ista', 'lista', 'coupled', 'step', 'ista', 'fista']
    for name, func_bench, type_ in zip(names, funcs_bench, l_type_):
        print(f"[main script] running {name}")
        print("-" * 80)
        # get the loss function evolution for a given 'algorithm'
        train_loss, test_loss = func_bench(x_train, x_test, D, lbda,
                                           type_=type_,
                                           all_n_layers=all_n_layers)
        l_train_loss.append(train_loss)
        l_test_loss.append(test_loss)
        print("=" * 80)

    # Plotting processing
    lw = 6
    eps_plots = 1.0e-10
    z_star_train = Lista(D=D, n_layers=10000).transform(x_train, lbda)
    min_train_loss = obj(z_star_train, D, x_train, lbda)
    z_star_test = Lista(D=D, n_layers=10000).transform(x_test, lbda)
    min_test_loss = obj(z_star_test, D, x_test, lbda)

    # Plotting train loss function
    plt.figure(f"[{__file__}] Train loss function", figsize=(6, 4))
    for name, train_loss in zip(names, l_train_loss):
        ls = '--' if name == 'ISTA-iterative' else '-'
        plt.loglog(ticks_layers + 1, train_loss - (min_train_loss - eps_plots),
                   ls=ls, lw=lw, label=name)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0,
               fontsize=12)
    plt.grid()
    plt.xticks(ticks_layers + 1, ticks_layers)
    plt.xlabel('Layers [-]', fontsize=15)
    plt.ylabel('$F(.) - F(z^*)$', fontsize=15)
    plt.title('Loss function evolution\non training set', fontsize=15)
    plt.tight_layout()
    filename = os.path.join(ploting_dir, "train_synthesis_loss.pdf")
    print("Saving plot at '{}'".format(filename))
    plt.savefig(filename, dpi=300)

    # Plotting test loss function
    plt.figure(f"[{__file__}] Test loss function", figsize=(6, 4))
    for name, test_loss in zip(names, l_test_loss):
        ls = '--' if name == 'ISTA-iterative' else '-'
        plt.loglog(ticks_layers + 1, test_loss - (min_test_loss - eps_plots),
                   ls=ls, lw=lw, label=name)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0,
               fontsize=12)
    plt.grid()
    plt.xticks(ticks_layers + 1, ticks_layers)
    plt.xlabel("Layers [-]", fontsize=15)
    plt.ylabel("$F(.) - F(z^*)$", fontsize=15)
    plt.title('Loss function evolution\non testing set', fontsize=15)
    plt.tight_layout()
    filename = os.path.join(ploting_dir, "test_synthesis_loss.pdf")
    print("Saving plot at '{}'".format(filename))
    plt.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

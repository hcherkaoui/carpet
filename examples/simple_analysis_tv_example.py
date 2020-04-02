""" Compare the convergence rate for the analysis 1d TV-l1 problem. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from prox_tv import tv1_1d
from carpet.datasets import synthetic_1d_dataset
from carpet.analysis_loss_gradient import obj
from carpet.utils import logspace_layers
from utils import ista_like_analy_tv, lista_like_analy_tv


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
    L = np.triu(np.ones((m, m)))
    D = (np.eye(m, k=-1) - np.eye(m, k=0))[:, :-1]
    x, _, _ = synthetic_1d_dataset(D=L, n=n_samples, s=s, snr=snr, seed=seed)

    x_train = x[n_samples_testing:, :]
    x_test = x[:n_samples_testing, :]

    l_train_loss, l_test_loss = [], []
    names = ['LTV-step', 'Sub-gradient']
    funcs_bench = [lista_like_analy_tv, ista_like_analy_tv]
    l_type_ = ['step', 'ista']
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

    z_hat_train_star = np.c_[[tv1_1d(x_train_, lbda) for x_train_ in x_train]]
    min_train_loss = obj(z_hat_train_star, D, x_train, lbda)
    z_hat_test_star = np.c_[[tv1_1d(x_test_, lbda) for x_test_ in x_test]]
    min_test_loss = obj(z_hat_test_star, D, x_test, lbda)

    # Plotting train loss function
    plt.figure(f"[{__file__}] Train loss function", figsize=(6, 4))
    for name, train_loss in zip(names, l_train_loss):
        ls = '--' if name == 'Sub-gradient' else '-'
        plt.loglog(ticks_layers, train_loss - (min_train_loss - eps_plots),
                   ls=ls, lw=lw, label=name)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0,
               fontsize=12)
    plt.grid()
    plt.xticks(ticks_layers + 1, ticks_layers)
    plt.xlabel('Layers [-]', fontsize=15)
    plt.ylabel('$F(.) - F(z^*)$', fontsize=15)
    plt.title('Loss function evolution\non training set', fontsize=15)
    plt.tight_layout()
    filename = os.path.join(ploting_dir, "train_analysis_loss.pdf")
    print("Saving plot at '{}'".format(filename))
    plt.savefig(filename, dpi=300)

    # Plotting test loss function
    plt.figure(f"[{__file__}] Test loss function", figsize=(6, 4))
    for name, test_loss in zip(names, l_test_loss):
        ls = '--' if name == 'Sub-gradient' else '-'
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
    filename = os.path.join(ploting_dir, "test_analysis_loss.pdf")
    print("Saving plot at '{}'".format(filename))
    plt.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

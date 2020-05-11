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
from carpet.loss_gradient import l1_reg
from carpet import LearnTVAlgo


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
    n_samples = 100
    n_samples_testing = 50
    n_atoms = 10
    n_dim = 5
    s = 0.2
    snr = 0.0
    n_layers = 50
    lbdas = np.linspace(0.0, 1.5, 10)

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
    l_train_reg, l_test_reg = [], []
    for lbda in lbdas:
        algo = LearnTVAlgo(algo_type='origista', A=A, n_layers=n_layers,
                           max_iter=500, device='cpu', verbose=0)
        t0_ = time.time()
        algo.fit(x_train, lbda=lbda)
        delta_ = time.time() - t0_
        z_train = algo.transform(x_train, lbda, output_layer=n_layers)
        z_test = algo.transform(x_test, lbda, output_layer=n_layers)
        train_reg = l1_reg(z_train)
        test_reg = l1_reg(z_test)
        l_train_reg.append(train_reg)
        l_test_reg.append(test_reg)
        print(f"[{algo.name}|lbda#{lbda:.3f}] model fitted "
              f"{delta_:3.1f}s train-reg={train_reg:.3e} "
              f"test-reg={test_reg:.3e}")

    ###########################################################################
    # Plotting
    lw = 4

    plt.figure(f"[{__file__}] Reg evolution", figsize=(6, 3))
    plt.plot(lbdas, l_train_reg, ls='-', lw=4, label='Training set')
    plt.plot(lbdas, l_test_reg, ls='--', lw=4, label='Testing set')
    plt.axvline(1.0, ls='--', lw=4, color='k')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                      borderaxespad=0.0, fontsize=15)
    plt.grid()
    plt.xlabel("Reg. ratio", fontsize=15)
    plt.ylabel('Mean reg. term', fontsize=15)
    plt.title('Regularization evolution', fontsize=18)

    plt.tight_layout()

    filename = os.path.join(ploting_dir, "reg_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    plt.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

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
from carpet.checks import check_random_state
from carpet.loss_gradient import l1_reg


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
    n_atoms = 20
    s = 0.2
    snr = 0.0
    lbdas = np.linspace(0.0, 2.0, 20)

    seed = np.random.randint(0, 1000)
    rng = check_random_state(seed)
    print(f'Seed used = {seed}')  # noqa: E999

    # Generate data
    results = synthetic_1d_dataset(A=np.eye(n_atoms), n=n_samples, s=s, snr=snr,
                                   seed=seed)
    x, _, _, _, _, _ = results


    ###########################################################################
    # Main experiment
    l_u_reg = []
    for lbda in lbdas:
        u = np.c_[[tv1_1d(x_, lbda) for x_ in x]]
        l_u_reg.append(np.mean(u))
        print(f"[Analysis|lbda#{lbda:.3f}] reg={np.mean(u):.3e} ")

    ###########################################################################
    # Plotting
    lw = 3
    l_u_reg = np.array(l_u_reg)
    plt.figure(f"[{__file__}] Reg evolution", figsize=(6, 3))
    plt.plot(lbdas, l_u_reg, ls='-', lw=lw)
    plt.axvline(1.0, ls='--', lw=lw, color='k')
    plt.axhline(np.mean(x), ls='--', lw=lw, color='k')
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

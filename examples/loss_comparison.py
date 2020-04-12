# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from prox_tv import tv1_1d
from carpet.datasets import synthetic_1d_dataset
from carpet.loss_gradient import analysis_obj
from carpet.utils import logspace_layers
from utils import (lasso_like_tv, learned_lasso_like_tv, chambolle_tv,
                   learned_chambolle_tv)


if __name__ == '__main__':

    print(__doc__)
    print('*' * 80)

    t0 = time.time()

    ploting_dir = 'outputs_plots'
    if not os.path.exists(ploting_dir):
        os.makedirs(ploting_dir)

    # Define variables
    n_samples = 1000
    n_samples_testing = 100
    m = 10
    s = 0.2
    snr = 0.0
    all_n_layers = logspace_layers(n_layers=20, max_depth=100)
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
    names = [
             'LISTA original',
             'LISTA coupled',
             'LISTA step',
             'ISTA iterative',
             'FISTA-iterative',
             'learned-TV Chambolle-Original',
             'learned-TV Chambolle-Coupled',
             'learned-TV Chambolle-Step',
             'Chambolle iterative',
             'Fast Chambolle iterative',
             ]
    funcs_bench = [
                   learned_lasso_like_tv,
                   learned_lasso_like_tv,
                   learned_lasso_like_tv,
                   lasso_like_tv,
                   lasso_like_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                   learned_chambolle_tv,
                   chambolle_tv,
                   chambolle_tv,
                   ]
    l_type_ = [
               'lista',
               'coupled',
               'step',
               'ista',
               'fista',
               'lchambolle',
               'coupledchambolle',
               'stepchambolle',
               'chambolle',
               'fast-chambolle',
               ]

    for name, func_bench, type_ in zip(names, funcs_bench, l_type_):
        print(f"[main script] running {name}")
        print("-" * 80)
        # get the loss function evolution for a given 'algorithm'
        train_loss, test_loss = func_bench(x_train, x_test, L, lbda,
                                           type_=type_,
                                           all_n_layers=all_n_layers)
        l_train_loss.append(train_loss)
        l_test_loss.append(test_loss)
        print("=" * 80)

    # Plotting processing
    lw = 5
    eps_plots = 1.0e-10
    z_hat_train_star = np.c_[[tv1_1d(x_train_, lbda) for x_train_ in x_train]]
    z_hat_test_star = np.c_[[tv1_1d(x_test_, lbda) for x_test_ in x_test]]
    min_train_loss = analysis_obj(z_hat_train_star, D, x_train, lbda)
    min_test_loss = analysis_obj(z_hat_test_star, D, x_test, lbda)

    # Plotting loss function
    fig, l_axis = plt.subplots(nrows=2, sharex=True, figsize=(9, 9),
                               num=f"[{__file__}] Loss functions")
    axis_train, axis_test = l_axis

    for name, train_loss in zip(names, l_train_loss):
        ls = '--' if 'Chambolle' in name else '-'
        train_loss -= (min_train_loss - eps_plots)
        axis_train.loglog(ticks_layers + 1, train_loss, ls=ls, lw=lw,
                          label=name)
    axis_train.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                      borderaxespad=0.0, fontsize=12)
    axis_train.grid()
    axis_train.set_ylabel('$F(.) - F(z^*)$', fontsize=12)
    axis_train.set_title('Loss function comparison on training set',
                         fontsize=12)

    for name, test_loss in zip(names, l_test_loss):
        ls = '--' if 'Chambolle' in name else '-'
        test_loss -= (min_test_loss - eps_plots)
        axis_test.loglog(ticks_layers + 1, test_loss, ls=ls, lw=lw, label=name)
    axis_test.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                     borderaxespad=0.0, fontsize=12)
    axis_test.grid()
    axis_test.set_xticks(ticks_layers + 1)
    axis_test.set_xticklabels(ticks_layers)
    axis_test.set_xlabel("Layers [-]", fontsize=12)
    axis_test.set_ylabel("$F(.) - F(z^*)$", fontsize=12)
    axis_test.set_title('Loss function comparison on testing set', fontsize=12)

    fig.tight_layout()
    filename = os.path.join(ploting_dir, "loss_comparison.pdf")
    print("Saving plot at '{}'".format(filename))
    fig.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

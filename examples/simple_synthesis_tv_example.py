""" Compare the convergence rate for the synthesis 1d TV-l1 problem. """
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from carpet.datasets import synthetic_1d_dataset
# from adopty.datasets import make_coding
from carpet.synthesis_loss_gradient import obj
from carpet.lista import Lista
from utils import ista_like_tv, lista_like_tv


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
    m = 5
    s = 0.3
    snr = 0.0
    n_layers = 10
    lbda_ratio = 0.2

    seed = np.random.randint(0, 1000)
    print(f"Seed used = {seed}")

    # Generate data
    D = np.triu(np.ones((m, m)))
    dataset, _, _ = synthetic_1d_dataset(n=n_samples, m=m, s=s, snr=snr,
                                         seed=seed)
    # dataset, D, _ = make_coding(n_samples=n_samples, n_atoms=m, n_dim=m,
    #                             random_state=seed)
    dataset /= n_samples  # normalized it

    training_dataset = dataset[n_samples_testing:, :]
    testing_dataset = dataset[:n_samples_testing, :]

    # Fix properly the regularization
    Dtx = testing_dataset.dot(D.T)
    lbda_max = np.max(Dtx)
    lbda = lbda_ratio * lbda_max

    l_train_loss, l_test_loss = [], []
    names = ['ISTA-neural-net', 'LISTA-original', 'LISTA-coupled',
             'LISTA-step', 'ISTA', 'FISTA', 'rs-FISTA', 'sub-gradient']
    funcs_bench = [lista_like_tv, lista_like_tv, lista_like_tv, lista_like_tv,
                   ista_like_tv, ista_like_tv, ista_like_tv, ista_like_tv]
    l_type_ = ['ista', 'lista', 'coupled', 'step', 'ista', 'fista', 'rsfista',
               'sub-gradient']

    for name, func_bench, type_ in zip(names, funcs_bench, l_type_):
        print(f"[main script] running {name}")
        print("-" * 80)
        # get the loss function evolution for a given 'algorithm'
        train_loss, test_loss = func_bench(training_dataset, testing_dataset,
                                           D, lbda, type_=type_,
                                           n_layers=n_layers)
        if train_loss is not None:
            l_train_loss.append(train_loss)
        l_test_loss.append(test_loss)
        print("=" * 80)

    # Plotting processing
    eps_plot = 1.0e-10
    lw = 5
    z_star_train = Lista(D=D, n_layers=5000).transform(training_dataset, lbda)
    min_train_loss = obj(z_star_train, D, training_dataset, lbda)
    z_star_test = Lista(D=D, n_layers=5000).transform(testing_dataset, lbda)
    min_test_loss = obj(z_star_test, D, testing_dataset, lbda)

    # Plotting train loss function
    names = ['ISTA-neural-net', 'LISTA-original', 'LISTA-coupled',
             'LISTA-step']
    plt.figure("Train loss function", figsize=(6, 4))
    for name, train_loss in zip(names, l_train_loss):
        plt.semilogy(train_loss - (min_train_loss - eps_plot), lw=lw,
                     label=name)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0,
               fontsize=12)
    plt.grid()
    plt.xticks(np.linspace(0, n_layers, 3), fontsize=15)
    plt.xlabel('Layers [-]', fontsize=15)
    plt.ylabel('F(.) - F(.)*', fontsize=15)
    plt.title('Loss function evolution\non training set', fontsize=15)
    plt.tight_layout()
    filename = os.path.join(ploting_dir, "train_loss_.pdf")
    print("Saving plot at '{}'".format(filename))
    plt.savefig(filename, dpi=300)

    # Plotting test loss function
    names = ['ISTA-neural-net', 'LISTA-original', 'LISTA-coupled',
             'LISTA-step', 'ISTA-iterative', 'FISTA-iterative',
             'restarting-FISTA-iterative', 'sub-gradient-iterative']
    plt.figure("Test loss function", figsize=(6, 4))
    for name, test_loss in zip(names, l_test_loss):
        ls = '--' if name ==  'ISTA-iterative' else '-'
        plt.semilogy(test_loss - (min_test_loss - eps_plot), ls=ls, lw=lw,
                     label=name)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0,
               fontsize=12)
    plt.grid()
    plt.xticks(np.linspace(0, n_layers, 3), fontsize=15)
    plt.xlabel("Layers [-]", fontsize=15)
    plt.ylabel("F(.) - F(.)*", fontsize=15)
    plt.title('Loss function evolution\non testing set', fontsize=15)
    plt.tight_layout()
    filename = os.path.join(ploting_dir, "test_loss_.pdf")
    print("Saving plot at '{}'".format(filename))
    plt.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))

    plt.show()

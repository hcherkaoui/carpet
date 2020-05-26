import numpy as np


from carpet.iterative_solver import IstaAnalysis
from carpet.datasets import synthetic_1d_dataset


def test_lbda_max():

    # Define variables
    n_dim = 5
    n_atoms = 8
    n_samples = 200
    s = 0.2
    snr = 0.0
    seed = 637

    # Generate data
    x, *_, A = synthetic_1d_dataset(n_atoms=n_atoms, n_dim=n_dim, n=n_samples,
                                    s=s, snr=snr, seed=seed)

    ista = IstaAnalysis(A, n_layers=1000, momentum='fista')

    # make sure the dataset is correctly scaled with lbda_max = 1
    z = ista.transform(x, lbda=1)
    assert all((np.diff(z, axis=1) != 0).sum(axis=1) == 0)

    z = ista.transform(x, lbda=1-1e-9)
    assert all((np.diff(z, axis=1) != 0).sum(axis=1) != 0)

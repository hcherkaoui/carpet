""" Utilities to generate a synthetic 1d data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from joblib import Parallel, delayed
from .checks import check_random_state


def add_gaussian_noise(signal, snr, random_state=None):
    """ Add a Gaussian noise to inout signal to output a noisy signal with the
    targeted SNR.

    Parameters
    ----------
    signal : array, the given signal on which add a Guassian noise.
    snr : float, the expected SNR for the output signal.
    random_state :  int or None (default=None),
        Whether to impose a seed on the random generation or not (for
        reproductability).

    Return
    ------
    noisy_signal : array, the noisy produced signal.
    noise : array, the additif produced noise.
    """
    # draw the noise
    rng = check_random_state(random_state)
    s_shape = signal.shape
    noise = rng.randn(*s_shape)
    # adjuste the standard deviation of the noise
    true_snr_num = np.linalg.norm(signal)
    true_snr_deno = np.linalg.norm(noise)
    true_snr = true_snr_num / (true_snr_deno + np.finfo(np.float).eps)
    std_dev = (1.0 / np.sqrt(10**(snr/10.0))) * true_snr
    noise = std_dev * noise
    noisy_signal = signal + noise
    return noisy_signal, noise


def _generate_dirac(m=100, s=0.1, rng=None):
    """ Generate a Dirac signal. """
    m_s = int(m*s)
    assert m_s != 0, "Generated zero z, please reduce sparsity"
    idx_non_zero = rng.randint(0, m, m_s)
    z = np.zeros(m)
    z[idx_non_zero] = rng.randn(m_s)
    return z


def _generate_1d_signal(A, L, s=0.1, snr=1.0, rng=None):
    """ Generate one 1d synthetic signal. """
    m = L.shape[0]
    z = _generate_dirac(m=m, s=s, rng=rng)
    u = z.dot(L)
    x, _ = add_gaussian_noise(signal=u.dot(A), snr=snr, random_state=rng)
    return x[None, :], u[None, :], z[None, :]


def synthetic_1d_dataset(n_atoms=10, n_dim=20, A=None, n=1000, s=0.1, snr=1.0,
                         seed=None, n_jobs=1):
    """ Generate n samples of 1d synthetic signal.

    Parameters
    ----------
    n_atoms : int, (default=10), number of atoms
    n_dim : int, (default=20), length of the obsersed signals
    A : np.ndarray, (default=None), shape=(n_atoms, n_dim)
        manually set observation matrix, if set to None, the function use
        generate a random Gaussian matrix of dimension (n_atoms, n_dim)
    n : int,
        number of 1d signal to generate
    s : float,
        sparsity of the derivative of the signal generated
    snr : float,
        SNR of the signal generated
    seed : int,
        random-seed used to initialize the random-instance
    n_jobs : int, (default=1),
        number of CPU to use

    Return
    ------
    x : numpy array, shape (n, n_dim), noisy observation of the signal
                 generated
    u : numpy array, shape (n, n_atoms), signal
    z : numpy array, shape (n, n_atoms - 1), source signal
    """
    rng = check_random_state(seed)

    # generate observation operator if needed
    if A is None:
        A = rng.randn(n_atoms, n_dim)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
    else:
        n_atoms = A.shape[0]

    # generate samples
    L = np.triu(np.ones((n_atoms, n_atoms)))
    D = (np.eye(n_atoms, k=-1) - np.eye(n_atoms, k=0))[:, :-1]
    params = dict(A=A, L=L, s=s, snr=snr, rng=rng)
    results = Parallel(n_jobs=n_jobs)(delayed(_generate_1d_signal)(**params)
                                      for _ in range(n))

    # stack samples
    x, u, z = [], [], []
    for x_, u_, z_ in results:
        x.append(x_.ravel())
        u.append(u_.ravel())
        z.append(z_.ravel())
    x, u, z = np.c_[x], np.c_[u], np.c_[z]

    # lbda_max = 1.0 for each sample
    S = A.sum(axis=0)
    c = (x.dot(S) / (S ** 2).sum())[:, None] * np.ones(z.shape)
    lmbd_max = np.abs((x - c.dot(A)).dot(A.T).dot(L.T))
    lmbd_max = lmbd_max.max(axis=1, keepdims=True)
    x /= lmbd_max

    return x, u, z, L, D, A

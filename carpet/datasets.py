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


def _generate_1d_signal(D, s=0.1, snr=1.0, rng=None):
    """ Generate one 1d synthetic signal. """
    z = _generate_dirac(m=D.shape[0], s=s, rng=rng)
    Dz = z.dot(D)
    x, _ = add_gaussian_noise(signal=Dz, snr=snr, random_state=rng)
    return x[None, :], Dz[None, :], z[None, :]


def synthetic_1d_dataset(D, n=1000, s=0.1, snr=1.0, seed=None, n_jobs=1):
    """ Generate n samples of 1d synthetic signal.

    Parameters
    ----------
    D : np.array, shape=(m1, m2), dictionary
    n : int, number of 1d signal to generate
    s : float, sparsity of the derivative of the signal generated
    snr : float, SNR of the signal generated
    seed : int, random-seed used to initialize the random-instance
    n_jobs : int, (default=1), number of CPU to use

    Return
    ------
    x : numpy array, shape (n, m), noisy observation of the signal
                 generated
    Dz : numpy array, shape (n, m), signal
    z : numpy array, shape (n, m), source signal
    """
    rng = check_random_state(seed)

    # generate samples
    params = dict(D=D, s=s, snr=snr, rng=rng)
    results = Parallel(n_jobs=n_jobs)(delayed(_generate_1d_signal)(**params)
                                      for _ in range(n))

    # stack samples
    x, Dz, z = [], [], []
    for x_, Dz_, z_ in results:
        x.append(x_.ravel())
        Dz.append(Dz_.ravel())
        z.append(z_.ravel())
    x, Dz, z = np.c_[x], np.c_[Dz], np.c_[z]

    # lbda_max = 1.0 for each sample
    x /= np.max(np.abs(x.dot(D.T)), axis=1, keepdims=True)

    return x, Dz, z

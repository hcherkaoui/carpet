""" Utilities to generate a synthetic 1d data. """
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


def _synthetic_1d_signal(m=100, s=0.1, snr=1.0, rng=None):
    """ Generate one 1d synthetic signal. """
    m_s = int(m*s)
    idx_non_zero = rng.randint(0, m, m_s)
    z = np.zeros(m)
    z[idx_non_zero] = rng.randn(m_s)
    Lz = np.cumsum(z)
    observed_Lz, _ = add_gaussian_noise(signal=Lz, snr=snr, random_state=rng)
    return observed_Lz[None, :], Lz[None, :], z[None, :]


def synthetic_1d_dataset(n=1000, m=100, s=0.1, snr=1.0, seed=None, n_jobs=1):
    """ Generate n samples of 1d synthetic signal.

    Parameters
    ----------
    n : int, number of 1d signal to generate
    m : int, length of the signal generated
    s : float, sparsity of the derivative of the signal generated
    snr : float, SNR of the signal generated
    seed : int, random-seed used to initialize the random-instance
    n_jobs : int, number of CPU to use

    Return
    ------
    observed_LZ : numpy array, shape (n, m), noisy observation of the signal
                 generated
    LZ : numpy array, shape (n, m), signal generated
    Z : numpy array, shape (n, m), derivative of the signal generated
    """
    rng = check_random_state(seed)

    params = dict(m=m, s=s, snr=snr, rng=rng)
    results = Parallel(n_jobs=n_jobs)(delayed(_synthetic_1d_signal)(**params)
                                      for _ in range(n))

    observed_LZ, LZ, Z = [], [], []
    for observed_Lz, Lz, z in results:
        observed_LZ.append(observed_Lz.ravel())
        LZ.append(Lz.ravel())
        Z.append(z.ravel())

    return np.c_[observed_LZ], np.c_[LZ], np.c_[Z]

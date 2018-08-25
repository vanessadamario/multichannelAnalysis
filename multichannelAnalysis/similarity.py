import numpy as np


def amp_correlation(x):
    """
    Correlation coefficients for all channels
    Parameters:
        x, matrix of dimension (# recordings, # time points)
        we must give a real input np.abs(wavelet_coefficients)
    Returns:
        correlation coefficients
    """
    return np.corrcoef(x)


def plv(x):
    """
    Compute the phase locking value, making a pairwise comparison of all the recordings.
    Parameter:
        x, matrix of dimension (# recordings, # time points)
    Return:
        phase locking value
    """
    n, samples = x.shape
    phase = np.arctan2(np.imag(x), np.real(x));
    return np.abs(np.exp(1j * phase).dot(np.exp(-1j * phase.T))) / samples

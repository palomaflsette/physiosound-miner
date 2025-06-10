import numpy as np
import scipy.stats


def compute_spectral_centroid(signal, sr):
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1/sr)
    if np.sum(spectrum) == 0:
        return 0.0
    return np.sum(freqs * spectrum) / np.sum(spectrum)


def compute_zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)


def compute_skewness(signal):
    return scipy.stats.skew(signal)


def compute_kurtosis(signal):
    return scipy.stats.kurtosis(signal)


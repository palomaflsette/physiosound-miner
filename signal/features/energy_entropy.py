import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy


def compute_rms_energy(signal):
    return np.sqrt(np.mean(signal**2))


def compute_spectral_entropy(signal, sr, nperseg=256):
    freqs, psd = welch(signal, fs=sr, nperseg=nperseg)
    psd_norm = psd / np.sum(psd)
    return scipy_entropy(psd_norm)


def compute_band_energy_ratios(signal, sr, bands=((0, 150), (150, 400))):
    freqs, psd = welch(signal, fs=sr)
    total_energy = np.sum(psd)
    ratios = []
    for low, high in bands:
        band_energy = np.sum(psd[(freqs >= low) & (freqs < high)])
        ratios.append(band_energy / total_energy if total_energy > 0 else 0)
    return ratios


def compute_sample_entropy(signal, m=2, r=None):
    """
    Calcula a Sample Entropy (SampEn) de um sinal unidimensional.
    
    Args:
        signal (np.ndarray): Sinal 1D.
        m (int): Tamanho do padrão (embedding dimension).
        r (float): Tolerância. Se None, será 0.2 * std(signal).

    Returns:
        float: Sample Entropy (quanto maior, mais irregular)
    """
    if len(signal) < m + 2:
        return np.nan  # não tem dados suficientes
    
    if r is None:
        r = 0.2 * np.std(signal)

    N = len(signal)
    
    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)

    try:
        return -np.log(_phi(m + 1) / _phi(m))
    except (ZeroDivisionError, FloatingPointError):
        return np.nan

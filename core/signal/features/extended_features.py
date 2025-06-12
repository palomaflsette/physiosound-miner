from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert
from scipy.fft import fft
import scipy.stats
import numpy as np
import scipy.signal
import librosa
from scipy.stats import kurtosis, skew


def zero_crossing_rate(signal):
    return np.mean(librosa.feature.zero_crossing_rate(signal))


def signal_envelope_stats(signal):
    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    return np.mean(envelope), np.std(envelope)


def spectral_bandwidth(signal, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))


def spectral_rolloff(signal, sr):
    return np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.85))


def spectral_flux(signal, sr, hop_length=512, n_fft=1024):

    stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
    return np.mean(flux)


def energy_subbands(signal, sr, bands=[(0, 150), (150, 400), (400, 1000)]):
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    power = np.abs(fft)**2
    subband_energies = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        subband_energies.append(np.sum(power[mask]))
    return subband_energies


def lpc_coeffs(signal, order=10):
    return librosa.lpc(signal, order=order)


def autocorrelation_lags(signal, lags=[1, 2, 3]):
    result = []
    norm_signal = signal - np.mean(signal)
    for lag in lags:
        if lag < len(signal):
            corr = np.corrcoef(norm_signal[:-lag], norm_signal[lag:])[0, 1]
        else:
            corr = np.nan
        result.append(corr)
    return result


def extended_statistical_features(signal):
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "kurtosis": kurtosis(signal),
        "skewness": skew(signal)
    }


def compute_zero_crossing_rate(signal: np.ndarray) -> float:
    return ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)


def compute_envelope_statistics(signal: np.ndarray, fs:int) -> dict[str, float]:
    envelope = np.abs(hilbert(signal))
    return {
        'envelope_mean': np.mean(envelope),
        'envelope_std': np.std(envelope),
        'envelope_max': np.max(envelope),
        'envelope_min': np.min(envelope),
    }


def compute_spectral_bandwidth(signal: np.ndarray, sr: int) -> float:
     return librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=1024).mean()


def compute_spectral_rolloff(signal: np.ndarray, sr: int, roll_percent: float = 0.85) -> float:
    return librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=roll_percent, n_fft=1024).mean()


def compute_spectral_flux(signal: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256) -> float:
     
     S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
     flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
     return np.mean(flux)



def compute_subband_energies(signal: np.ndarray, sr: int) -> dict:
    fft_vals = np.abs(fft(signal))[:len(signal) // 2]
    freqs = np.fft.fftfreq(len(signal), d=1/sr)[:len(signal) // 2]

    bands = [(0, 50), (50, 100), (100, 200), (200, 400)]
    energies = {}
    for i, (low, high) in enumerate(bands):
        mask = (freqs >= low) & (freqs < high)
        energies[f'subband_energy_{low}_{high}Hz'] = np.sum(
            fft_vals[mask] ** 2)
    return energies


def compute_lpc_coefficients(signal: np.ndarray, order: int = 10) -> dict:
     try:
        coeffs = librosa.lpc(signal, order=order)
        return {f'lpc_{i}': coeff for i, coeff in enumerate(coeffs)}
     except Exception:
        return {f'lpc_{i}': 0 for i in range(order + 1)}



def compute_autocorrelation_lags(signal: np.ndarray, lags: int = 3) -> dict:
    result = np.correlate(signal, signal, mode='full')
    result = result[result.size // 2:]
    return {f'autocorr_lag_{i+1}': result[i+1] if i+1 < len(result) else 0 for i in range(lags)}

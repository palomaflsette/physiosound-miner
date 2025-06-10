import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from typing import Tuple, List
import librosa



def apply_fft(signal: np.ndarray, fs: int, n_fft: int = 1024) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply FFT to a time-domain signal using fixed n_fft size.

    Parameters:
        signal (np.ndarray): Time-domain signal.
        fs (int): Sampling rate in Hz.
        n_fft (int): FFT window size (default = 512)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - freqs: Frequency bins in Hz.
            - magnitudes: Normalized magnitude of the FFT.
            - fft_result: Complex FFT coefficients.
    """
    if len(signal) < n_fft:
        # Zero-padding à direita
        padded_signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')
    else:
        #padded_signal = signal[:n_fft]  # recorte, se maior
        padded_signal = signal
    
    fft_result = np.fft.fft(padded_signal, n=n_fft)
    magnitudes = np.abs(fft_result[:n_fft // 2]) * 2 / n_fft
    freqs = np.fft.fftfreq(n_fft, d=1 / fs)[:n_fft // 2]
    return freqs, magnitudes, fft_result


def apply_stft_with_hamming(signal: np.ndarray, fs: int, window_size: int = 2048, hop_size: int = 512):
    """
    Apply Short-Time Fourier Transform (STFT) using a Hamming window.

    Parameters:
        signal (np.ndarray): Input signal.
        fs (int): Sampling rate in Hz.
        window_size (int): Size of the FFT window. Default is 2048.
        hop_size (int): Step size between successive windows. Default is 512.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - freqs: Array of frequency bins in Hz.
            - magnitude: Magnitude spectrogram (time x frequency).
    """
    window = np.hamming(window_size)
    stft_result = librosa.stft(
        signal, n_fft=window_size, hop_length=hop_size, window=window)
    magnitude = np.abs(stft_result)
    freqs = librosa.fft_frequencies(sr=fs, n_fft=window_size)
    return freqs, magnitude

def get_dominant_frequencies(freqs: np.ndarray, magnitudes: np.ndarray, threshold: float = 0.01) -> List[float]:
    """
    Select dominant frequencies based on a relative magnitude threshold.

    Parameters:
        freqs (np.ndarray): Array of frequency bins (Hz).
        magnitudes (np.ndarray): Corresponding magnitude values.
        threshold (float): Relative threshold (0–1) for selecting dominant peaks.
                           Frequencies with magnitude above threshold * max are returned.

    Returns:
        List[float]: List of dominant frequencies in Hz.
    """
    max_magnitude = np.max(magnitudes)
    indices = np.where(magnitudes > threshold * max_magnitude)[0]
    return freqs[indices].tolist()





# Exemplo de uso
# PATH = '../../data/raw/waves-ipanema-beach.wav'
# fs, signal = load_audio(PATH)
# freqs, magnitudes, _ = apply_fft(signal, fs)
# dominantes = get_dominant_frequencies(freqs, magnitudes, threshold=0.3)


# plot_signal_in_time(PATH)

# plot_time_components(signal, fs, dominantes[:2])
# plot_frequency_components(signal, fs, dominantes[:7])

# for freq in dominantes[:7]:
#     plot_winding(signal, fs, freq, duration=0.25)
    
# features_list = []
# for freq in dominantes[:5]:  # pega até 5 freq. dominantes
#     feats = extract_winding_features(signal, fs, freq)
#     features_list.append(feats)

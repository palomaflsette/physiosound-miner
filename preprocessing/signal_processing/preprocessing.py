from scipy.signal import butter, filtfilt, decimate
import numpy as np
from typing import Tuple



def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize the signal to the range [-1, 1].

    Parameters:
        signal (np.ndarray): Raw audio signal.

    Returns:
        np.ndarray: Normalized signal with values between -1 and 1.
    """
    return signal / np.max(np.abs(signal))


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float = 20.0, highcut: float = 800.0, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the signal.

    Parameters:
        signal (np.ndarray): Input audio signal.
        fs (int): Sampling rate in Hz.
        lowcut (float): Lower cutoff frequency in Hz. Default is 20.0 Hz.
        highcut (float): Upper cutoff frequency in Hz. Default is 2000.0 Hz.
        order (int): Order of the filter. Default is 4.

    Returns:
        np.ndarray: Filtered audio signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < high < 1):
        print(
            f"Aviso: corte inválido. fs={fs}, lowcut={lowcut}, highcut={highcut}")
        return signal  # retorna sem filtrar
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def binomial_filter(signal: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Smooth the signal using a simple binomial (moving average-like) filter.

    Parameters:
        signal (np.ndarray): Input audio signal.
        iterations (int): Number of times the filter is applied. Default is 1.

    Returns:
        np.ndarray: Smoothed signal.
        
    """
    kernel = np.array([1, 2, 1]) / 4.0
    for _ in range(iterations):
        signal = np.convolve(signal, kernel, mode='same')
    return signal


def kalman_filter(signal: np.ndarray) -> np.ndarray:
    """
    Apply a simple 1D Kalman filter to smooth the signal.

    Returns:
        Smoothed signal (same shape as input)
    """
    n = len(signal)
    x = np.zeros(n)  # estimated signal
    P = np.zeros(n)  # estimated error
    x[0] = signal[0]
    P[0] = 1.0

    Q = 1e-5  # process variance (tune this)
    R = 0.03  # measurement noise (tune this)

    for k in range(1, n):
        # Prediction
        x_pred = x[k-1]
        P_pred = P[k-1] + Q

        # Update
        K = P_pred / (P_pred + R)
        x[k] = x_pred + K * (signal[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x


def preprocess_signal(
    signal: np.ndarray, 
    fs: int, 
    apply_filter: bool = True, 
    apply_smoothing: bool = True,
    use_kalman: bool = True
    ) -> np.ndarray:
    """
    Execute the complete preprocessing pipeline:
    - Normalization
    - Bandpass filter (Butterworth)
    - Smoothing: Binomial or Kalman

    Parameters:
        signal (np.ndarray): Raw input signal.
        fs (int): Sampling rate in Hz.
        apply_filter (bool): Whether to apply bandpass filter. Default is True.
        apply_smoothing (bool): Whether to apply smoothing. Default is False.
        use_kalman (bool): Whether to apply Kalman filter. Default is True.

    Returns:
        np.ndarray: Preprocessed signal ready for feature extraction.
    """
    signal = normalize_signal(signal)
    if apply_filter:
        signal = bandpass_filter(signal, fs)
    if apply_smoothing:
        if use_kalman:
            signal = kalman_filter(signal)
        else:
            signal = binomial_filter(signal)
    return signal


def prepare_signal(
    signal: np.ndarray,
    fs: int,
    duration: float = 5.0,
    downsample: int = 10,
    use_kalman: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Aplica pré-processamento e redução de taxa de amostragem ao sinal.

    Parâmetros:
        signal (np.ndarray): Sinal de entrada (1D, domínio do tempo).
        fs (int): Taxa de amostragem original (Hz).
        duration (float): Duração máxima em segundos a ser considerada do sinal (default: 5.0).
        downsample (int): Fator de decimação. Ex: 10 reduz de 44100 Hz para 4410 Hz (default: 10).
        use_kalman (bool): Se True, aplica filtro de Kalman; caso contrário, filtro binomial.

    Retorna:
        Tuple[np.ndarray, int]: Tupla com o sinal pré-processado e a nova taxa de amostragem.
    """
    signal = preprocess_signal(signal, fs, use_kalman=use_kalman)

    max_samples = int(fs * duration)
    signal = signal[:max_samples]
    if len(signal) == 0:
        raise ValueError("O sinal está vazio após o pré-processamento.")
    
    signal = decimate(signal, q=downsample)
    fs_new = fs // downsample

    return signal, fs_new

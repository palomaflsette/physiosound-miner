import numpy as np
from typing import List


def segment_signal(signal: np.ndarray, fs: int, window_duration_sec: float = 2.0, overlap: float = 0.5) -> List[np.ndarray]:
    """
    Segments a signal into overlapping windows of fixed duration.

    Parameters:
        signal (np.ndarray): The input signal array.
        fs (int): Sampling frequency in Hz.
        window_duration_sec (float): Duration of each window in seconds. Default is 2.0.
        overlap (float): Fraction of overlap between consecutive windows (0 to <1). Default is 0.5.

    Returns:
        List[np.ndarray]: A list of signal segments/windows.
    """
    window_size = int(window_duration_sec * fs)
    hop_size = int(window_size * (1 - overlap))
    segments = []

    for start in range(0, len(signal) - window_size + 1, hop_size):
        end = start + window_size
        segment = signal[start:end]
        segments.append(segment)

    return segments
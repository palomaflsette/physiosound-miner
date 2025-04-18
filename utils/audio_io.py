import numpy as np
import scipy.io.wavfile as wav
from typing import Tuple


def load_audio(file_path: str) -> Tuple[int, np.ndarray]:
    """
    Load a .wav audio file.

    Parameters:
        file_path (str): Path to the .wav file.

    Returns:
        Tuple[int, np.ndarray]:
            - fs: Sampling rate (Hz) - number of samples per second. According to the Nyquist theorem, the highest frequency we can detect is fs/2
            - signal: Mono audio signal as NumPy array. vector of sound signal samples is the digital signal, that is, the numerical representation of the sound in the time domain
    """
    fs, signal = wav.read(file_path)
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)
    return fs, signal

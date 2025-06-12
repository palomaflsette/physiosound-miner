import librosa
import numpy as np
from typing import List


def extract_mfcc_features(signal: np.ndarray, fs: int, n_mfcc: int = 13, n_fft: int = 1024, hop_length: int = 512) -> List[float]:
    """
    Extrai coeficientes MFCC de um sinal de áudio mono.

    Parameters:
        signal (np.ndarray): Sinal de entrada (1D).
        fs (int): Taxa de amostragem.
        n_mfcc (int): Número de coeficientes MFCC a extrair.

    Returns:
        List[float]: Lista com os coeficientes médios de MFCC extraídos.
    """
    adjusted_n_fft = min(n_fft, len(signal))  # Evita warnings

    mfccs = librosa.feature.mfcc(
        y=signal.astype(float),
        sr=fs,
        n_mfcc=n_mfcc,
        n_fft=adjusted_n_fft,
        hop_length=hop_length
    )
    return mfccs.mean(axis=1).tolist()

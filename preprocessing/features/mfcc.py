import librosa
import numpy as np
from typing import List


def extract_mfcc_features(signal: np.ndarray, fs: int, n_mfcc: int = 13) -> List[float]:
    """
    Extrai coeficientes MFCC de um sinal de áudio mono.

    Parameters:
        signal (np.ndarray): Sinal de entrada (1D).
        fs (int): Taxa de amostragem.
        n_mfcc (int): Número de coeficientes MFCC a extrair.

    Returns:
        List[float]: Lista com os coeficientes médios de MFCC extraídos.
    """
    mfccs = librosa.feature.mfcc(y=signal.astype(float), sr=fs, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1).tolist()

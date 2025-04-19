import sys
import os
sys.path.append(os.path.abspath("../../"))
import numpy as np
from typing import List, Dict
from preprocessing.signal_processing.wavelet import apply_wavelet_transform

def extract_wavelet_features(signal: np.ndarray, wavelet: str = "db4", level: int = None) -> Dict[str, float]:
    """
    Extrai características estatísticas dos coeficientes da transformada wavelet.

    Parameters:
        signal (np.ndarray): Sinal de entrada (1D).
        wavelet (str): Tipo de wavelet (default: "db4").
        level (int): Nível de decomposição (default: máximo possível).

    Returns:
        Dict[str, float]: Dicionário com características por nível.
    """
    coeffs, _ = apply_wavelet_transform(signal, wavelet, level)

    features = {}
    for i, c in enumerate(coeffs):
        prefix = f"L{i}" if i == 0 else f"D{i}"
        c = np.asarray(c)

        features[f"{prefix}_mean"] = float(np.mean(c))
        features[f"{prefix}_std"] = float(np.std(c))
        features[f"{prefix}_energy"] = float(np.sum(c ** 2))

    return features
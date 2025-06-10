"""
Aplicar transformada wavelet discreta (ex: Daubechies) → ideal para capturar detalhes finos.

Podemos usar PyWavelets (pywt) para facilitar.

Gera um vetor multiescalar que pode alimentar a rede neural.

"""

import pywt
import numpy as np
from typing import Tuple, List


def apply_wavelet_transform(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = None
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Aplica a transformada wavelet discreta (DWT) no sinal de entrada.

    Parameters:
        signal (np.ndarray): Sinal de entrada (1D).
        wavelet (str): Nome da wavelet a ser utilizada (ex: 'db4', 'sym5').
        level (int): Nível de decomposição. Se None, usa o máximo possível para o tamanho do sinal.

    Returns:
        Tuple[List[np.ndarray], List[int]]: Coeficientes por nível e suas respectivas dimensões originais.
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    shapes = [len(c) for c in coeffs]
    return coeffs, shapes

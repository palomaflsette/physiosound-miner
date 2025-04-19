from typing import Tuple, List, Callable, Dict
import pandas as pd
import numpy as np
from preprocessing.segment_signal import segment_signal


def extract_features_from_segmented_signal(
    signal: np.ndarray,
    fs: int,
    file_id: str,
    extract_its_fn: Callable[[np.ndarray, int, List[float], float], List[Dict]],
    fft_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    get_dom_freqs_fn: Callable[[np.ndarray, np.ndarray, float], List[float]],
    extract_mfcc_fn: Callable[[np.ndarray, int], List[float]],
    extract_wavelet_fn: Callable[[np.ndarray], Dict[str, float]],
    window_sec: float = 2.0,
    overlap: float = 0.5,
    threshold: float = 0.2,
    duration: float = 1.0,
    n_mfcc: int = 13
) -> pd.DataFrame:
    """
    Extrai ITS, MFCCs e Wavelet features de um sinal segmentado, combinando os vetores em um único DataFrame.

    Returns:
        pd.DataFrame: DataFrame contendo vetores ITS+MFCC+Wavelet com metadados.
    """
    windows = segment_signal(signal, fs, window_sec, overlap)
    all_features = []

    for i, window in enumerate(windows):
        freqs, mags, _ = fft_fn(window, fs)
        dominantes = get_dom_freqs_fn(freqs, mags, threshold)

        # ITS
        its_list = extract_its_fn(window, fs, dominantes, duration)

        # MFCC (único vetor por janela)
        mfccs = extract_mfcc_fn(window, fs, n_mfcc)

        # Wavelet features
        wavelet_feats = extract_wavelet_fn(window)

        for feat in its_list:
            feat["file_id"] = file_id
            feat["window_id"] = i

            # Adiciona MFCCs numerados: mfcc_0, mfcc_1, ...
            for j, val in enumerate(mfccs):
                feat[f"mfcc_{j}"] = val

            # Adiciona descritores da wavelet
            for k, val in wavelet_feats.items():
                feat[f"wave_{k}"] = val

            all_features.append(feat)

    return pd.DataFrame(all_features)

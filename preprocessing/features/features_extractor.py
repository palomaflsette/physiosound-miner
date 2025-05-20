from typing import Tuple, List, Callable, Dict
import pandas as pd
import numpy as np
from preprocessing.segment_signal import segment_signal


def extract_features_from_segmented_signal(
    signal: np.ndarray,
    fs: int,
    file_id: str,
    extract_its_fn: Callable[[np.ndarray, int, List[float], float], List[Dict]] = None,
    fft_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    get_dom_freqs_fn: Callable[[np.ndarray, np.ndarray, float], List[float]] = None,
    extract_mfcc_fn: Callable[[np.ndarray, int], List[float]] = None,
    extract_wavelet_fn: Callable[[np.ndarray], Dict[str, float]] = None,
    extract_rqa_fn: Callable[[np.ndarray, int, str, int], Dict[str, float]] = None,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    threshold: float = 0.2,
    duration: float = 1.0,
    n_mfcc: int = 13
) -> pd.DataFrame:
    """
    Extrai ITS, MFCCs, Wavelet e opcionalmente RQA de um sinal segmentado,
    combinando os vetores em um único DataFrame por janela.

    Returns:
        pd.DataFrame: DataFrame contendo vetores ITS+MFCC+Wavelet+RQA com metadados.
    """
    windows = segment_signal(signal, fs, window_sec, overlap)
    all_features = []

    for i, window in enumerate(windows):
        freqs, mags, _ = fft_fn(window, fs)
        dominantes = get_dom_freqs_fn(freqs, mags, threshold)

        # ITS (para cada frequência dominante de cada janela)
        its_list = extract_its_fn(window, fs, dominantes, duration)

        # MFCC (único vetor por janela)
        mfccs = extract_mfcc_fn(window, fs, n_mfcc)

        # Wavelet features (único por janela)
        wavelet_feats = extract_wavelet_fn(window)

        # RQA (único por janela)
        rqa_feats = extract_rqa_fn(window, fs, file_id, i) if extract_rqa_fn else {}

        for feat in its_list:  # aqui é como se eu estivesse percorrendo as linhas do DataFrame
            feat["file_id"] = file_id
            feat["window_id"] = i

            # Adiciona MFCCs numerados, será o mesmo para cada janela
            for j, val in enumerate(mfccs):
                feat[f"mfcc_{j}"] = val

            # Adiciona descritores da wavelet, será o mesmo para cada janela
            for k, val in wavelet_feats.items():
                feat[f"wave_{k}"] = val

            # Adiciona RQA, será o mesmo para cada janela
            for k, val in rqa_feats.items():
                feat[f"rqa_{k}"] = val

            all_features.append(feat)

    return pd.DataFrame(all_features)

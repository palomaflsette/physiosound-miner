from typing import Tuple, List, Callable, Dict
import pandas as pd
import numpy as np
from segment_signal import segment_signal
from features.energy_entropy import (
    compute_rms_energy,
    compute_spectral_entropy,
    compute_band_energy_ratios,
    compute_sample_entropy
)
from features.temporal_shape import (
    compute_spectral_centroid,
    compute_zero_crossing_rate,
    compute_skewness,
    compute_kurtosis
)



def extract_features_from_segmented_signal(
    signal: np.ndarray,
    fs: int,
    file_id: str,
    extract_its_fn: Callable[[np.ndarray, int,
                              List[float], float], List[Dict]] = None,
    fft_fn: Callable[[np.ndarray, int],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    get_dom_freqs_fn: Callable[[np.ndarray,
                                np.ndarray, float], List[float]] = None,
    extract_mfcc_fn: Callable[[np.ndarray, int], List[float]] = None,
    extract_wavelet_fn: Callable[[np.ndarray], Dict[str, float]] = None,
    extract_rqa_fn: Callable[[np.ndarray, int,
                              str, int], Dict[str, float]] = None,
    window_duration_sec: float = 2.0,
    overlap: float = 0.5,
    threshold: float = 0.2,
    winding_duration: float = 5.0,
    n_mfcc: int = 13
) -> pd.DataFrame:
    """
    Extrai ITS, MFCCs, Wavelet, RQA e outras features espectrais e energéticas
    de um sinal segmentado, retornando um DataFrame por janela.
    """
    windows = segment_signal(signal, fs, window_duration_sec, overlap)
    all_features = []

    for i, window in enumerate(windows):
        freqs, mags, _ = fft_fn(window, fs)
        dominantes = get_dom_freqs_fn(freqs, mags, threshold)

        its_list = extract_its_fn(window, fs, dominantes, winding_duration)

        # MFCCs
        mfccs = extract_mfcc_fn(window, fs, n_mfcc) if extract_mfcc_fn else []

        # Wavelet features
        wavelet_feats = extract_wavelet_fn(
            window) if extract_wavelet_fn else {}

        # RQA
        rqa_feats = extract_rqa_fn(
            window, fs, file_id, i) if extract_rqa_fn else {}

        # Extração das features de forma temporal/espectral
        centroid = compute_spectral_centroid(window, fs)
        zcr = compute_zero_crossing_rate(window)
        skew = compute_skewness(window)
        kurt = compute_kurtosis(window)

        # Energy and Entropy features
        rms = compute_rms_energy(window)
        spec_entropy = compute_spectral_entropy(window, fs)
        band_ratios = compute_band_energy_ratios(window, fs)
        samp_entropy = compute_sample_entropy(window)

        for feat in its_list:
            feat["file_id"] = file_id
            feat["window_id"] = i

            # MFCCs
            for j, val in enumerate(mfccs):
                feat[f"mfcc_{j}"] = val

            # Wavelet
            for k, val in wavelet_feats.items():
                feat[f"wave_{k}"] = val

            # RQA
            for k, val in rqa_feats.items():
                feat[f"rqa_{k}"] = val

            # Energy & Entropy
            feat["rms_energy"] = rms
            feat["spectral_entropy"] = spec_entropy
            feat["band_energy_0_150Hz"] = band_ratios[0]
            feat["band_energy_150_400Hz"] = band_ratios[1]
            feat["sample_entropy"] = samp_entropy
            
            feat["spectral_centroid"] = centroid
            feat["zero_crossing_rate"] = zcr
            feat["skewness"] = skew
            feat["kurtosis"] = kurt


            all_features.append(feat)

    return pd.DataFrame(all_features)

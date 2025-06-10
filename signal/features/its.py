from typing import Tuple, List, Callable, Dict
import pandas as pd
import numpy as np
from segment_signal import segment_signal
from features.winding import extract_winding_features_extended


def extract_its(signal: np.ndarray, fs: int, freqs: List[float], winding_duration: float = 1.0) -> List[dict]:
    """
    Extract the extended topological index (ITS) for multiple frequencies of the same signal.

    Parameters:
        signal (np.ndarray): Input time-domain signal.
        fs (int): Sampling rate in Hz.
        freqs (List[float]): List of target frequencies to extract descriptors from.
        winding_duration (float): Time duration to consider from the signal for each winding.

    Returns:
        List[dict]: A list of ITS vectors (one per frequency).
    """
    return [extract_winding_features_extended(signal, fs, f, winding_duration) for f in freqs]




def extract_its_from_segmented_signal(
        signal: np.ndarray,
        fs: int,
        file_id: str,
        extract_its_fn: Callable[[np.ndarray, int, List[float], float], List[Dict]],
        fft_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        get_dom_freqs_fn: Callable[[np.ndarray, np.ndarray, float], List[float]],
        window_duration_sec: float = 2.0,
        overlap: float = 0.5,
        threshold: float = 0.2,
        winding_duration: float = 1.0,) -> pd.DataFrame:
    """
    Extracts ITS features from a full signal using window segmentation.

    Parameters:
        signal (np.ndarray): Input signal.
        fs (int): Sampling frequency in Hz.
        file_id (str): Identifier for the original file.
        extract_its_fn (Callable): Function to extract ITS features.
        fft_fn (Callable): Function to compute FFT.
        get_dom_freqs_fn (Callable): Function to obtain dominant frequencies.
        window_duration_sec (float): Duration of each window in seconds.
        overlap (float): Overlap fraction between windows.
        threshold (float): Magnitude threshold for dominant frequencies.
        duration (float): Duration (in seconds) considered inside each segment for ITS.

    Returns:
        pd.DataFrame: DataFrame containing ITS vectors with metadata.
    """
    windows = segment_signal(signal, fs, window_duration_sec, overlap)
    all_its = []

    for i, window in enumerate(windows):
        freqs, mags, _ = fft_fn(window, fs)
        dominantes = get_dom_freqs_fn(freqs, mags, threshold)
        its_list = extract_its_fn(window, fs, dominantes, winding_duration)

        for feat in its_list:
            feat["file_id"] = file_id
            feat["window_id"] = i
            all_its.append(feat)

    return pd.DataFrame(all_its)


def save_its_to_csv(features: List[dict], out_path: str) -> None:
    """
    Save a list of ITS feature dictionaries to a CSV file.

    Parameters:
        features (List[dict]): List of ITS feature dictionaries.
        out_path (str): Output path for the CSV file.

    Returns:
        None
    """
    df = pd.DataFrame(features)
    df.to_csv(out_path, index=False)

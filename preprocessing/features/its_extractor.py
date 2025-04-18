import pandas as pd
import numpy as np
from typing import List
from preprocessing.features.winding import extract_winding_features_extended


def extract_its(signal: np.ndarray, fs: int, freqs: List[float], duration: float = 1.0) -> List[dict]:
    """
    Extract the extended topological index (ITS) for multiple frequencies of the same signal.

    Parameters:
        signal (np.ndarray): Input time-domain signal.
        fs (int): Sampling rate in Hz.
        freqs (List[float]): List of target frequencies to extract descriptors from.
        duration (float): Time duration to consider from the signal for each winding.

    Returns:
        List[dict]: A list of ITS vectors (one per frequency).
    """
    return [extract_winding_features_extended(signal, fs, f, duration) for f in freqs]


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

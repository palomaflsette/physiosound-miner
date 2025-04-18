from scipy.spatial import cKDTree
from typing import Dict
from scipy.stats import entropy
import numpy as np


def extract_winding_features(signal: np.ndarray, fs: int, freq: float, duration: float = 1.0) -> dict:
    """
    Extracts numerical features from the winding curve generated for a given frequency.

    Parameters:
        signal (np.ndarray): Time-domain signal array.
        fs (int): Sampling rate in Hz.
        freq (float): Target frequency in Hz to generate the winding.
        duration (float): Time window in seconds to consider from the signal. Default is 1.0s.

    Returns:
        dict: Dictionary containing numerical descriptors of the winding, including:
            - freq (float): The analyzed frequency.
            - centro_x (float): X-coordinate of the winding centroid.
            - centro_y (float): Y-coordinate of the winding centroid.
            - raio_medio (float): Average distance from center to points.
            - raio_std (float): Standard deviation of radius.
            - raio_max (float): Maximum radius.
            - raio_min (float): Minimum radius.
            - simetria_x (float): Symmetry along X-axis.
            - simetria_y (float): Symmetry along Y-axis.
            - densidade_nucleo (float): Percentage of points close to the center.
    """
    t = np.arange(len(signal)) / fs
    n_samples = int(fs * duration)
    t = t[:n_samples]
    signal = signal[:n_samples]

    # Generate winding
    winding = signal * np.exp(-2j * np.pi * freq * t)
    x = winding.real
    y = winding.imag

    # Centroid and radius
    cx, cy = np.mean(x), np.mean(y)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    features = {
        'freq': freq,
        'centro_x': cx,
        'centro_y': cy,
        'raio_medio': np.mean(r),
        'raio_std': np.std(r),
        'raio_max': np.max(r),
        'raio_min': np.min(r),
        'simetria_x': np.mean(np.abs(x + x[::-1])),  # Reflective symmetry X
        'simetria_y': np.mean(np.abs(y - y[::-1])),  # Reflective symmetry Y
        # Ratio of points near center
        'densidade_nucleo': np.mean(r < (0.2 * np.max(r)))
    }

    return features


def extract_winding_features_extended(signal: np.ndarray, fs: int, freq: float, duration: float = 1.0) -> Dict[str, float]:
    """
    Extracts extended topological and geometric descriptors from the winding curve 
    generated for a given frequency.

    Parameters:
        signal (np.ndarray): Time-domain signal array.
        fs (int): Sampling rate in Hz.
        freq (float): Target frequency in Hz.
        duration (float): Time duration (in seconds) to consider. Default is 1.0s.

    Returns:
        dict: Dictionary containing extended winding descriptors.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signal = signal[:n_samples]

    # Generate winding
    winding = signal * np.exp(-2j * np.pi * freq * t)
    x = winding.real
    y = winding.imag

    # Centroid and radius
    cx, cy = np.mean(x), np.mean(y)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Curve length
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    curve_length = np.sum(segment_lengths)

    # Direction change (angle between consecutive vectors)
    angles = np.arctan2(dy, dx)
    d_theta = np.diff(angles)
    mean_angle_change = np.mean(np.abs(d_theta))
    curvature_variation = np.std(d_theta)

    # Entropy of radius distribution
    hist, _ = np.histogram(r, bins=20, density=True)
    radius_entropy = entropy(hist + 1e-9)

    # Efficient auto-intersection detection with KDTree
    points = np.column_stack((x, y))
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=1e-2)
    auto_intersections = len(pairs)

    features = {
        'freq': freq,
        'centro_x': cx,
        'centro_y': cy,
        'raio_medio': np.mean(r),
        'raio_std': np.std(r),
        'raio_max': np.max(r),
        'raio_min': np.min(r),
        'simetria_x': np.mean(np.abs(x + x[::-1])),
        'simetria_y': np.mean(np.abs(y - y[::-1])),
        'densidade_nucleo': np.mean(r < (0.2 * np.max(r))),
        'comprimento_curva': curve_length,
        'variacao_curvatura': curvature_variation,
        'mudanca_media_direcao': mean_angle_change,
        'entropia_raio': radius_entropy,
        'auto_intersecoes': auto_intersections,
    }

    return features

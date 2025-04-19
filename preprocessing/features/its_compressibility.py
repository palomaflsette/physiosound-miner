import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy
import zlib


def compressibility_pca(vector: np.ndarray, variance_threshold: float = 0.95) -> int:
    """
    Returns the number of PCA components required to explain a given percentage of variance.

    Parameters:
        vector (np.ndarray): ITS vector, shape (n_features,)
        variance_threshold (float): Variance explanation threshold (default 0.95)

    Returns:
        int: Number of components needed to reach the variance threshold
    """
    pca = PCA()
    pca.fit(vector.reshape(1, -1))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return np.searchsorted(cumulative_variance, variance_threshold) + 1


def compressibility_zlib(vector: np.ndarray) -> float:
    """
    Computes the compression ratio using zlib (lossless compression).

    Parameters:
        vector (np.ndarray): ITS vector

    Returns:
        float: Compression ratio (compressed size / original size)
    """
    raw = vector.astype(np.float32).tobytes()
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)


def shannon_entropy(vector: np.ndarray, bins: int = 20) -> float:
    """
    Estimates the Shannon entropy of the vector distribution.

    Parameters:
        vector (np.ndarray): ITS vector
        bins (int): Number of histogram bins (default 20)

    Returns:
        float: Entropy value
    """
    hist, _ = np.histogram(vector, bins=bins, density=True)
    return entropy(hist + 1e-9)


def compute_its_compressibility(df: pd.DataFrame, variance_thresholds=[0.9, 0.95, 0.99], plot=True):
    """
    Aplica PCA nos vetores ITS e avalia quantos componentes explicam diferentes níveis de variância.

    Parameters:
        df (pd.DataFrame): DataFrame contendo os vetores ITS.
        variance_thresholds (list): Lista com níveis de variância a verificar.
        plot (bool): Se True, plota a curva de variância explicada acumulada.

    Returns:
        dict: Número de componentes necessários para cada threshold.
    """
    # Seleciona apenas colunas numéricas de ITS (descarta file_id, freq, etc.)
    its_data = df.select_dtypes(include=[np.number]).copy()

    # Aplica PCA
    pca = PCA()
    pca.fit(its_data)
    explained = np.cumsum(pca.explained_variance_ratio_)

    result = {}
    for v in variance_thresholds:
        n_components = np.argmax(explained >= v) + 1
        result[f"{int(v*100)}%"] = n_components

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(explained)+1),
                 explained, marker='o', color='royalblue')
        plt.axhline(0.9, color='gray', linestyle='--', label='90%')
        plt.axhline(0.95, color='gray', linestyle='--', label='95%')
        plt.axhline(0.99, color='gray', linestyle='--', label='99%')
        plt.xlabel("Número de Componentes")
        plt.ylabel("Variância Explicada Acumulada")
        plt.title("Compressibilidade dos Vetores ITS via PCA")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result

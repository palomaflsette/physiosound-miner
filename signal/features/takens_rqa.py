"""
Módulo para extração de métricas de Recurrence Quantification Analysis (RQA)
a partir de janelas de sinais unidimensionais, utilizando o embedding de Takens.
"""

import numpy as np
from typing import Dict
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric


def extract_rqa_features(
    signal: np.ndarray,
    fs: int,
    file_id: str = "unknown",
    window_id: int = -1,
    embedding_dimension: int = 5,
    time_delay: int = 2,
    threshold: float = 0.1,
    normalize: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Extrai métricas de Recurrence Quantification Analysis (RQA) a partir de uma janela de sinal.

    Parâmetros:
        signal (np.ndarray): Sinal unidimensional no domínio do tempo (janela).
        fs (int): Taxa de amostragem do sinal em Hz.
        file_id (str): Identificador do arquivo de origem (opcional).
        window_id (int): Índice da janela dentro do arquivo (opcional).
        embedding_dimension (int): Dimensão do embedding (Takens).
        time_delay (int): Delay temporal entre componentes do vetor embutido.
        threshold (float): Raio (epsilon) para definir recorrência entre pontos.
        normalize (bool): Se True, normaliza o sinal antes da análise (zero-mean, unit-std).
        verbose (bool): Se True, imprime as métricas extraídas no terminal.

    Retorna:
        Dict[str, float]: Dicionário com as métricas RQA e metadados da janela.
    """
    if normalize:
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

    ts = TimeSeries(
        signal.tolist(), embedding_dimension=embedding_dimension, time_delay=time_delay)
    settings = Settings(
        time_series=ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(threshold),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1
    )

    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()

    features = {
        "recurrence_rate": result.recurrence_rate,
        "determinism": result.determinism,
        "average_diagonal_line": result.average_diagonal_line,
        "longest_diagonal_line": result.longest_diagonal_line,
        "entropy_diagonal_lines": result.entropy_diagonal_lines,
        "laminarity": result.laminarity,
        "trapping_time": result.trapping_time,
        "longest_vertical_line": result.longest_vertical_line,
        "entropy_vertical_lines": result.entropy_vertical_lines,
        "file_id": file_id,
        "window_id": window_id,
        "fs": fs,
        "embedding_dim": embedding_dimension,
        "delay": time_delay,
        "rqa_threshold": threshold
    }

    if verbose:
        print(f"[RQA] file: {file_id}, window: {window_id}")
        for k, v in features.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return features


def extract_rqa_features_smart(
    signal: np.ndarray,
    fs: int,
    file_id: str = "unknown",
    window_id: int = -1,
    embedding_dimension: int = 6,
    time_delay: int = 3,
    thresholds: list = [0.2, 0.3, 0.4, 0.5],
    normalize: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.computation import RQAComputation
    from pyrqa.analysis_type import Classic
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.metric import EuclideanMetric

    if normalize:
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

    best_result = None
    best_valid_count = -1
    best_threshold = None

    for th in thresholds:
        try:
            ts = TimeSeries(
                signal.tolist(), embedding_dimension=embedding_dimension, time_delay=time_delay)
            settings = Settings(ts, analysis_type=Classic, neighbourhood=FixedRadius(
                th), similarity_measure=EuclideanMetric, theiler_corrector=1)
            result = RQAComputation.create(settings, verbose=False).run()

            # Checar quantas métricas são não-nulas
            metrics = {
                "recurrence_rate": result.recurrence_rate,
                "determinism": result.determinism,
                "average_diagonal_line": result.average_diagonal_line,
                "longest_diagonal_line": result.longest_diagonal_line,
                "entropy_diagonal_lines": result.entropy_diagonal_lines,
                "laminarity": result.laminarity,
                "trapping_time": result.trapping_time,
                "longest_vertical_line": result.longest_vertical_line,
                "entropy_vertical_lines": result.entropy_vertical_lines,
            }

            valid_count = sum(1 for v in metrics.values()
                              if v not in [0.0, np.nan, None])

            if result.recurrence_rate > 0.01 and valid_count > best_valid_count:
                best_valid_count = valid_count
                best_result = metrics.copy()
                best_threshold = th

        except Exception as e:
            if verbose:
                print(f"[RQA] Threshold {th} falhou: {e}")
            continue

    if best_result is None:
        # Se nenhum threshold produziu métricas úteis, retorna tudo -1
        best_result = {k: -1 for k in [
            "recurrence_rate", "determinism", "average_diagonal_line", "longest_diagonal_line",
            "entropy_diagonal_lines", "laminarity", "trapping_time", "longest_vertical_line",
            "entropy_vertical_lines"
        ]}
        best_threshold = -1

    # Adicionar metadados
    best_result.update({
        "file_id": file_id,
        "window_id": window_id,
        "fs": fs,
        "embedding_dim": embedding_dimension,
        "delay": time_delay,
        "rqa_threshold": best_threshold
    })

    if verbose:
        print(
            f"[RQA Smart] file: {file_id}, window: {window_id}, threshold: {best_threshold}")
        for k, v in best_result.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return best_result

import sys
import os
sys.path.append(os.path.abspath("../.."))
import argparse
from tqdm import tqdm
import glob
from scipy.signal import decimate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import openpyxl
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool
from pathlib import Path
from signal.features.its_compressibility import compute_its_compressibility
from utils.audio_io import load_audio
from utils.plot_utils import plot_takens_embedding, plot_winding_xy
from signal.signal_processing.preprocessing import preprocess_signal, prepare_signal
from signal.features.winding import get_winding_curve
from signal.signal_processing.fourier import apply_fft, get_dominant_frequencies
from signal.features.its import extract_its
from signal.features.features_extractor import extract_features_from_segmented_signal
from signal.features.mfcc import extract_mfcc_features
from signal.features.wavelet import extract_wavelet_features
from signal.segment_signal import segment_signal
from signal.features.takens_rqa import extract_rqa_features
import subprocess


def extrair_features_topologicas(wav_path):
    fs, signal = load_audio(wav_path)
    signal, fs = prepare_signal(
        signal, fs, max_audio_duration=15, decimation_factor=5, use_kalman=True)

    df_topologico = extract_features_from_segmented_signal(
        signal=signal,
        fs=fs,
        file_id=wav_path,
        extract_its_fn=extract_its,
        fft_fn=apply_fft,
        get_dom_freqs_fn=get_dominant_frequencies,
        extract_rqa_fn=extract_rqa_features,
        window_duration_sec=3.0,
        overlap=0.5,
        threshold=0.5,
        winding_duration=2.5
    )

    rqa_cols = [col for col in df_topologico.columns if col.startswith(
        "rqa_") and df_topologico[col].isna().any()]
    df_topologico[rqa_cols] = df_topologico[rqa_cols].fillna(-1)
    df_topologico["has_trapping"] = (
        df_topologico["rqa_trapping_time"] != -1).astype(int)

    return df_topologico


def extrair_todas_as_features(wav_path, mix_df, id_column_name):
    base_id = os.path.basename(wav_path).replace('.wav', '')
    df_feat = extrair_features_topologicas(wav_path)

    meta = mix_df[mix_df[id_column_name] == base_id].iloc[0]
    for col in meta.index:
        df_feat[col] = meta[col]

    return df_feat


def processar_arquivo(args):
    path, df_rotulos_path, id_column_name, tmp_dir = args
    try:
        df_rotulos = pd.read_csv(df_rotulos_path)
        df = extrair_todas_as_features(path, df_rotulos, id_column_name)

        nome = Path(path).stem
        df.to_parquet(f"{tmp_dir}/{nome}.parquet", index=False)
    except Exception as e:
        return f"Erro no {path}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Extrai features topológicas com paralelismo e salva em parquet.")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Caminho da pasta onde estão os arquivos .wav")
    parser.add_argument("--audio_ext", type=str, default=".wav",
                        help="Extensão dos arquivos de áudio (padrão: .wav)")
    parser.add_argument("--rotulos_csv", type=str, required=True,
                        help="Caminho do arquivo CSV com os rótulos")
    parser.add_argument("--id_col", type=str, required=True,
                        help="Nome da coluna de ID para match")
    parser.add_argument("--tmp_dir", type=str, default="tmp_parquet",
                        help="Pasta temporária para arquivos parquet")
    parser.add_argument("--output", type=str, required=True,
                        help="Caminho do arquivo CSV final consolidado")
    parser.add_argument("--processes", type=int, default=4,
                        help="Número de processos paralelos")
    args = parser.parse_args()

    path_audio_files = f"{args.audio_dir}/**/*{args.audio_ext}"
    arquivos = sorted(glob.glob(path_audio_files, recursive=True))
    Path(args.tmp_dir).mkdir(exist_ok=True)

    args_list = [(path, args.rotulos_csv, args.id_col, args.tmp_dir)
                 for path in arquivos]

    with Pool(processes=args.processes) as p:
        with tqdm(total=len(args_list)) as pbar:
            for resultado in p.imap_unordered(processar_arquivo, args_list):
                if resultado:  # pode ser None se tudo deu certo
                    print(resultado)
                pbar.update()

    parquet_files = sorted(glob.glob(f"{args.tmp_dir}/*.parquet"))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_csv(args.output, index=False)
    print(f"✔️ Dataset salvo em: {args.output}")


if __name__ == "__main__":
    main()

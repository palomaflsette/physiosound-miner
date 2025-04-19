import sys
import os
sys.path.append(os.path.abspath(".."))
from preprocessing.segment_signal import segment_signal
from preprocessing.features.wavelet import extract_wavelet_features
from preprocessing.features.mfcc import extract_mfcc_features
from preprocessing.features.features_extractor import extract_features_from_segmented_signal
from preprocessing.features.its import extract_its
from preprocessing.signal_processing.fourier import apply_fft, get_dominant_frequencies
from preprocessing.features.winding import get_winding_curve
from preprocessing.signal_processing.preprocessing import preprocess_signal
from utils.plot_utils import plot_takens_embedding, plot_winding_xy
from utils.audio_io import load_audio
import openpyxl
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import decimate

""" 
set PYTHONPATH=.
streamlit run interface/app.py
"""

nome_do_arquivo = "data/raw/ICBHI_Respiratory_Database/101_1b1_Al_sc_Meditron.wav"
file_id = os.path.basename(nome_do_arquivo).replace(".wav", "")

fs, signal = load_audio(
    nome_do_arquivo)

np.save("interface/signal_example.npy", signal)


def compute_winding(signal, fs, freq, duration):
    t = np.arange(len(signal)) / fs
    t = t[:int(fs * duration)]
    signal = signal[:int(fs * duration)]
    signal = signal - np.mean(signal)

    winding = signal * np.exp(-2j * np.pi * freq * t)
    return winding.real, winding.imag


# === Interface === #
st.set_page_config(layout="wide")
st.title("🔍 Visual Explorer: ITS, Windings, Embeddings")

# Carregamento do áudio já processado (simulado)
fs = 4410
# ou substitua com st.file_uploader
signal = np.load("signal_example.npy")

# Parâmetros interativos
window_sec = st.sidebar.slider("Duração da janela (s)", 0.5, 2.0, 1.0, 0.1)
overlap = st.sidebar.slider("Overlap", 0.0, 0.9, 0.5, 0.1)
duration = st.sidebar.slider(
    "Duração usada na winding (s)", 0.5, 2.0, 1.0, 0.1)
tau = st.sidebar.slider("Tau (embedding)", 1, 30, 10)
dim = st.sidebar.selectbox("Dimensão do embedding", [2, 3])

segments = segment_signal(signal, fs, window_sec, overlap)
selected_win = st.sidebar.slider("Janela", 0, len(segments)-1, 0)
selected_segment = segments[selected_win]

# FFT para obter frequências dominantes
freqs, mags, _ = apply_fft(selected_segment, fs)
dominantes = get_dominant_frequencies(freqs, mags, threshold=0.2)
dominantes = [f for f in dominantes if f >= 5.0]

selected_freqs = st.sidebar.multiselect(
    "Frequências para winding", dominantes, default=dominantes[:3])

# Layout: Embedding + Windings + Vetores
col1, col2 = st.columns([1.2, 1.2])

with col1:
    st.subheader("🌌 Takens Embedding (3D ou 2D)")
    fig = plt.figure(figsize=(6, 6))
    fig = plot_takens_embedding(selected_segment, tau=tau, dim=3,
                                title=f"Takens Embedding da Janela {selected_win}")
    st.pyplot(fig)
    if len(selected_segment) < (dim - 1) * tau:
         st.warning("Segmento muito curto para gerar embedding com os parâmetros atuais.")



with col2:
    st.subheader("🌀 Windings por Frequência Dominante")
    for f in selected_freqs:
        x, y = compute_winding(selected_segment, fs, f, duration)
        fig = plt.figure(figsize=(2, 2))
        fig = plot_winding_xy(
            x, y, freq=f, title=f"Winding - f = {f:.1f} Hz", return_fig=True)
        st.pyplot(fig)


# Vetores de características (exemplo)
st.subheader("📊 Vetores extraídos da Janela")
its_vectors = extract_its(selected_segment, fs, dominantes, duration)
mfcc_vector = extract_mfcc_features(selected_segment, fs)
wavelet_vector = extract_wavelet_features(selected_segment)

df = pd.DataFrame(its_vectors)
for i, val in enumerate(mfcc_vector):
    df[f"mfcc_{i}"] = val
for k, v in wavelet_vector.items():
    df[f"{k}"] = v

st.dataframe(df)

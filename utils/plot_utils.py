from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from typing import List
from utils.audio_io import load_audio
from core.signal.signal_processing.preprocessing import (normalize_signal, 
                                                         bandpass_filter, 
                                                         binomial_filter, 
                                                         kalman_filter)

def plot_signal_preprocessing_steps(signal: np.ndarray, fs: int, title_prefix='', file_name =None):
    zoom_duration = 2.5  # segundos
    max_samples = int(fs * zoom_duration)
    signal = signal[:max_samples]
    time = np.arange(len(signal)) / fs

    signal_norm = normalize_signal(signal)
    signal_band = bandpass_filter(signal_norm, fs)
    signal_binom = binomial_filter(signal_band)
    signal_kalman = kalman_filter(signal_band)

    fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    axs[0].plot(time, signal)
    axs[0].set_title(f'{title_prefix} Sinal original')
    axs[1].plot(time, signal_norm)
    axs[1].set_title(f'{title_prefix} Normalizado [-1, 1]')
    axs[2].plot(time, signal_band)
    axs[2].set_title(f'{title_prefix} Após filtro passa-banda (20–800 Hz)')
    # axs[3].plot(time, signal_binom)
    # axs[3].set_title(f'{title_prefix} Após suavização binomial')
    axs[3].plot(time, signal_kalman)
    axs[3].set_title(f'{title_prefix} Após filtro de Kalman')
    
    axs[4].plot(time, signal_norm, label='Normalizado', alpha=0.7)
    axs[4].plot(time, signal_band, label='Passa-banda', alpha=0.7)
    axs[4].plot(time, signal_binom, label='Binomial', alpha=0.7)
    axs[4].plot(time, signal_kalman, label='Kalman', alpha=0.7)
    axs[4].set_title(f'{title_prefix} Sinal normalizado e filtrado (sobreposição)')
    axs[4].legend()
    axs[4].set_ylabel("Amplitude")

    axs[5].plot(time, signal, label='Original', alpha=0.7)
    axs[5].plot(time, signal_kalman, label='Kalman (final)', alpha=0.7)
    axs[5].set_title(f'{title_prefix} Original vs Kalman')
    axs[5].legend()
    axs[5].set_xlabel("Tempo (s)")
    axs[5].set_ylabel("Amplitude")
    plt.tight_layout()
    if file_name is None:
        output_path = "assets/preprocessing_pipeline_sobreposicao_zoom.png"
    else: output_path = "assets/" + file_name
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_time_domain(signal: np.ndarray, fs: int, title: str = "Sinal no Domínio do Tempo") -> None:
     duration = len(signal) / fs
     time = np.linspace(0, duration, len(signal))
     plt.figure(figsize=(12, 4))
     plt.plot(time, signal, color='blue')
     plt.title(title)
     plt.xlabel("Tempo (s)")
     plt.ylabel("Amplitude")
     plt.grid(True)
     plt.tight_layout()
     plt.show()


def plot_spectrum(freqs: np.ndarray, magnitudes: np.ndarray, title: str = "Espectro de Frequências") -> None:
    """
    Plota o espectro de frequência.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitudes, color='orange')
    plt.title(title)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_time_components(signal: np.ndarray, fs: int, freqs: List[float], duration: float = 1.0) -> None:
    """
    Plota as componentes senoidais correspondentes a frequências fornecidas,
    reconstruídas a partir da FFT do sinal original.

    Args:
        signal: vetor com o sinal no tempo (mono)
        fs: taxa de amostragem do sinal
        freqs: lista de frequências a extrair (em Hz)
        duration: tempo (em segundos) a ser exibido nos gráficos
    """
    from scipy.fft import fft, ifft, fftfreq

    N = len(signal)
    t = np.arange(N) / fs
    spectrum = fft(signal)
    fft_freqs = fftfreq(N, 1/fs)

    # Constrói os sinais por faixa de frequência
    components = []

    for f in freqs:
        band = (f - 5, f + 5)  # faixa de ±5Hz ao redor da frequência alvo
        filtered = np.zeros_like(spectrum, dtype=complex)
        mask = (np.abs(fft_freqs) >= band[0]) & (np.abs(fft_freqs) <= band[1])
        filtered[mask] = spectrum[mask]
        reconstructed = np.real(ifft(filtered))
        components.append(reconstructed)

    # Plotar
    num_plots = len(freqs)
    max_samples = int(fs * duration)
    time_axis = t[:max_samples]

    plt.figure(figsize=(12, 2.5 * num_plots))
    for i, comp in enumerate(components):
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(time_axis, comp[:max_samples])
        plt.title(f"Componente: ~{freqs[i]} Hz")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_frequency_components(signal: np.ndarray, fs: int, freqs: List[float], bandwidth: float = 10.0) -> None:
    """
    Plota os espectros (no domínio da frequência) das componentes senoidais
    centradas nas frequências fornecidas, reconstruídas a partir da FFT do sinal.

    Args:
        signal: vetor com o sinal no tempo (mono)
        fs: taxa de amostragem do sinal
        freqs: lista de frequências centrais (em Hz) a analisar
        bandwidth: largura da faixa (em Hz) em torno da frequência central
    """
    from scipy.fft import fft, ifft, fftfreq

    N = len(signal)
    spectrum = fft(signal)
    fft_freqs = fftfreq(N, 1/fs)

    plt.figure(figsize=(12, 2.5 * len(freqs)))

    for i, f in enumerate(freqs):
        band = (f - bandwidth / 2, f + bandwidth / 2)
        filtered = np.zeros_like(spectrum, dtype=complex)
        mask = (np.abs(fft_freqs) >= band[0]) & (np.abs(fft_freqs) <= band[1])
        filtered[mask] = spectrum[mask]
        reconstructed = np.real(ifft(filtered))

        # FFT da componente reconstruída
        component_spectrum = np.abs(fft(reconstructed))[:N // 2] * 2 / N
        component_freqs = fft_freqs[:N // 2]

        plt.subplot(len(freqs), 1, i + 1)
        plt.plot(component_freqs, component_spectrum)
        plt.title(f"Espectro da Componente ~{f:.2f} Hz")
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    

def plot_winding_xy(x, y, freq=None, title=None, show_center=True, return_fig=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    cx, cy = np.mean(x), np.mean(y)

    ax.plot(x, y, color='mediumturquoise', linewidth=1)
    if show_center:
        ax.scatter(cx, cy, color='red', label='Centroid')
    ax.axis('equal')
    ax.grid(True)

    if title:
        ax.set_title(title)
    elif freq:
        ax.set_title(f"Winding – {freq:.2f} Hz")
    else:
        ax.set_title("Winding Curve")

    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imaginary Axis")
    ax.legend()

    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()  # só funciona no Jupyter


def generate_winding_data(signal: np.ndarray, fs: int, freq: float):
    """
    Gera os dados X e Y para a curva winding de um sinal de áudio.
    
    Parameters:
        signal: sinal de áudio (1D array)
        fs: taxa de amostragem
        freq: frequência para fazer o winding (Hz)
    
    Returns:
        x, y: coordenadas da curva winding
    """
    N = len(signal)
    t = np.arange(N) / fs  # vetor tempo

    winding_freq = 2 * np.pi * freq
    complex_signal = signal * np.exp(-1j * winding_freq * t)

    x = np.real(complex_signal)
    y = np.imag(complex_signal)

    return x, y

def plot_winding_xy(x, y, freq=None, title=None, show_center=True, return_fig=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    cx, cy = np.mean(x), np.mean(y)

    # Curva por baixo
    ax.plot(x, y, color='mediumturquoise', linewidth=1, alpha=0.7, zorder=1)
    
    # Centróide por cima - MAIS VISÍVEL
    if show_center:
        ax.scatter(cx, cy, color='red', s=150, label='Centroid', 
                  zorder=10, edgecolors='black', linewidth=2)  # ← MUDANÇAS AQUI
    
    ax.axis('equal')
    ax.grid(True, alpha=0.3, zorder=0)  # Grid por trás

    if title:
        ax.set_title(title)
    elif freq:
        ax.set_title(f"Winding – {freq:.2f} Hz")
    else:
        ax.set_title("Winding Curve")

    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imaginary Axis")
    ax.legend()

    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()

def plot_winding_xy_enhanced(x, y, freq=None, title=None, show_center=True, return_fig=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    cx, cy = np.mean(x), np.mean(y)

    # Curva winding
    ax.plot(x, y, color='mediumturquoise', linewidth=1.5, alpha=0.6, zorder=1)
    
    if show_center:
        # Centróide com halo para destacar ainda mais
        ax.scatter(cx, cy, color='white', s=200, zorder=9, alpha=0.8)  # Halo branco
        ax.scatter(cx, cy, color='red', s=120, zorder=10, 
                  edgecolors='darkred', linewidth=3, label='Centroid')  # Centróide vermelho
        
        # Adicionar coordenadas do centróide
        ax.annotate(f'({cx:.3f}, {cy:.3f})', 
                   xy=(cx, cy), xytext=(cx+0.1, cy+0.1),
                   fontsize=10, ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                   zorder=11)
    
    ax.axis('equal')
    ax.grid(True, alpha=0.3, zorder=0)

    if title:
        ax.set_title(title, fontsize=14)
    elif freq:
        ax.set_title(f"Winding – {freq:.2f} Hz", fontsize=14)
    else:
        ax.set_title("Winding Curve", fontsize=14)

    ax.set_xlabel("Real Axis", fontsize=12)
    ax.set_ylabel("Imaginary Axis", fontsize=12)
    ax.legend(fontsize=10)

    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()


def plot_winding_xy_professional(x, y, freq=None, title=None, show_center=True, return_fig=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    cx, cy = np.mean(x), np.mean(y)

    # Curva winding com gradiente de cor (opcional)
    ax.plot(x, y, color='mediumturquoise', linewidth=1.5, alpha=0.7, zorder=1)
    
    if show_center:
        # Cruz de referência no centróide
        ax.axhline(y=cy, color='gray', linestyle='--', alpha=0.5, zorder=2)
        ax.axvline(x=cx, color='gray', linestyle='--', alpha=0.5, zorder=2)
        
        # Centróide destacado
        ax.scatter(cx, cy, color='red', s=200, zorder=10, 
                  edgecolors='darkred', linewidth=3, 
                  marker='o', label='Centroid')
        
        # Distância do centróide à origem
        distance = np.sqrt(cx**2 + cy**2)
        ax.plot([0, cx], [0, cy], 'r--', linewidth=2, alpha=0.8, zorder=9,
               label=f'Distance: {distance:.3f}')
        
        # Origem marcada
        ax.scatter(0, 0, color='black', s=100, marker='x', linewidth=3, zorder=10)
    
    ax.axis('equal')
    ax.grid(True, alpha=0.3, zorder=0)

    if title:
        ax.set_title(title, fontsize=16, pad=20)
    elif freq:
        ax.set_title(f"Winding Analysis – {freq:.1f} Hz", fontsize=16, pad=20)
    else:
        ax.set_title("Winding Curve Analysis", fontsize=16, pad=20)

    ax.set_xlabel("Real Axis", fontsize=14)
    ax.set_ylabel("Imaginary Axis", fontsize=14)
    ax.legend(fontsize=12)

    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()


def plot_audio_winding(filepath: str, freq: float, duration: float = 2.0, title: str = None):
    """
    Plota a curva winding de um arquivo de áudio.
    
    Parameters:
        filepath: caminho para o arquivo de áudio
        freq: frequência para winding (Hz)
        duration: duração do sinal a usar (segundos)
        title: título customizado
    """
    fs, signal = load_audio(filepath)

    if signal.ndim > 1:
        signal = signal[:, 0]

    max_samples = int(fs * duration)
    signal = signal[:max_samples]

    signal = signal / np.max(np.abs(signal))

    x, y = generate_winding_data(signal, fs, freq)

    if title is None:
        title = f"Winding Curve - {freq:.1f} Hz"

    plot_winding_xy(x, y, freq=freq, title=title, show_center=True)



def plot_multiple_windings(filepath: str, frequencies: list, duration: float = 2.0):
    """
    Plota múltiplas curvas winding para diferentes frequências.
    """
    fs, signal = load_audio(filepath)

    if signal.ndim > 1:
        signal = signal[:, 0]

    max_samples = int(fs * duration)
    signal = signal[:max_samples]
    signal = signal / np.max(np.abs(signal))

    n_freqs = len(frequencies)
    fig, axes = plt.subplots(1, n_freqs, figsize=(6*n_freqs, 6))

    if n_freqs == 1:
        axes = [axes]

    for i, freq in enumerate(frequencies):
        x, y = generate_winding_data(signal, fs, freq)
        cx, cy = np.mean(x), np.mean(y)

        axes[i].plot(x, y, color='mediumturquoise', linewidth=1)
        axes[i].scatter(cx, cy, color='red', s=100, label='Centroid')
        axes[i].set_title(f"Winding - {freq:.1f} Hz")
        axes[i].set_xlabel("Real Axis")
        axes[i].set_ylabel("Imaginary Axis")
        axes[i].axis('equal')
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_signal_in_time(filepath: str, duration: float = 5.0):
    """
    Plota o gráfico no tempo da senoide da música original.
    
    Parameters:
        filepath (str): caminho do arquivo .wav
        duration (float): duração (em segundos) a ser exibida no gráfico
    """
    fs, signal = load_audio(filepath)

    if signal.ndim > 1:
        signal = signal[:, 0]

    max_samples = int(fs * duration)
    time = np.linspace(0, duration, max_samples)
    signal = signal[:max_samples]

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, color='deepskyblue')
    plt.title(f"Sinal no tempo (primeiros {duration} segundos)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def takens_embedding(signal: np.ndarray, tau: int, dim: int = 3) -> np.ndarray:
    """
    Gera um embedding de Takens a partir de um sinal 1D.

    Parameters:
        signal (np.ndarray): Sinal 1D de entrada.
        tau (int): Tempo de atraso (lag).
        dim (int): Dimensão do espaço embutido (tipicamente 2 ou 3).

    Returns:
        np.ndarray: Matriz de shape (N, dim), onde N = len(signal) - (dim - 1)*tau.
    """
    n_points = len(signal) - (dim - 1) * tau
    return np.array([signal[i:i + tau * dim:tau] for i in range(n_points)])


def plot_takens_embedding(signal: np.ndarray, tau: int = 10, dim: int = 3, title: str = "Takens Embedding"):
    """
    Plota o embedding de Takens em 2D ou 3D.

    Parameters:
        signal (np.ndarray): Sinal de entrada.
        tau (int): Atraso temporal.
        dim (int): Dimensão do embedding.
        title (str): Título do gráfico.
    """
    embedded = takens_embedding(signal, tau, dim)

    fig = plt.figure(figsize=(6, 6))
    if embedded.shape[0] == 0:
        return fig  # figura vazia se não houver dados suficientes

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], lw=0.8)
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t+{tau})")
        ax.set_zlabel(f"x(t+{2*tau})")
    elif dim == 2:
        ax = fig.add_subplot(111)
        ax.plot(embedded[:, 0], embedded[:, 1], lw=0.8)
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t+{tau})")
    else:
        raise ValueError("Dimensão suportada: 2 ou 3")

    plt.title(title)
    plt.tight_layout()
    return fig


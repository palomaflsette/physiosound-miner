from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from typing import List
from utils.audio_io import load_audio



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




def plot_signal_in_time(filepath: str, duration: float = 5.0):
    """
    Plota o gráfico no tempo da senoide da música original.
    
    Parameters:
        filepath (str): caminho do arquivo .wav
        duration (float): duração (em segundos) a ser exibida no gráfico
    """
    fs, signal = load_audio(filepath)

    # Garante mono
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Recorta os primeiros segundos
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


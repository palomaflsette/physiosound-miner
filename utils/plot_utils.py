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
    

def plot_winding_xy(x: np.ndarray, y: np.ndarray, freq: float = None, title: str = None, show_center: bool = True) -> None:
    """
    Plot a 2D winding curve from x and y components.

    Parameters:
        x (np.ndarray): Real part of winding.
        y (np.ndarray): Imaginary part of winding.
        freq (float): Optional frequency label to display in the title.
        title (str): Optional custom title.
        show_center (bool): Whether to show the centroid point.
    """
    cx, cy = np.mean(x), np.mean(y)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color='mediumturquoise', linewidth=1)
    if show_center:
        plt.scatter(cx, cy, color='red', label='Centroid')
    plt.axis('equal')
    plt.grid(True)
    if title:
        plt.title(title)
    elif freq:
        plt.title(f"Winding – {freq:.2f} Hz")
    else:
        plt.title("Winding Curve")
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.legend()
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


def extract_winding_features(signal: np.ndarray, fs: int, freq: float, duration: float = 1.0) -> dict:
    """
    Gera um vetor de características numéricas a partir da winding de uma frequência.

    Returns:
        dict com features: centroide, raio médio, desvio, simetria, etc.
    """
    t = np.arange(len(signal)) / fs
    n_samples = int(fs * duration)
    t = t[:n_samples]
    signal = signal[:n_samples]

    # Gera winding
    winding = signal * np.exp(-2j * np.pi * freq * t)
    x = winding.real
    y = winding.imag

    # Centroide
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
        'simetria_x': np.mean(np.abs(x + x[::-1])),  # eixo x
        'simetria_y': np.mean(np.abs(y - y[::-1])),  # eixo y
        'densidade_nucleo': np.mean(r < (0.2 * np.max(r)))  # % de pontos no núcleo
    }

    return features
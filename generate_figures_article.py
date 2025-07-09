import os
from utils.plot_utils import *
import numpy as np
from scipy.io import wavfile

wav_path = 'data/raw_data/HS_normal_sopro/F_S3_LLSB.wav'

def senoides_comparativas(wav_path):

     fs, signal = wavfile.read(wav_path)

     if signal.ndim > 1:
          signal = signal[:, 0]

     signal = signal.astype(np.float32)

     output_file = plot_signal_preprocessing_steps(
     signal, fs, title_prefix='Sinal real:', file_name="graficos")
     print(f"Figura salva em: {output_file}")

def gera_windings_frequencia_especifca(wav_path):

     plot_audio_winding(wav_path, freq=440.0, duration=2.0)

     #frequencies = [100, 200, 440, 1000]  # Hz
     #plot_multiple_windings(wav_path, frequencies, duration=1.0)

     fs, signal = load_audio(wav_path)
     signal = signal[:int(fs * 2)]  # 2 segundos
     x, y = generate_winding_data(signal, fs, 440.0)
     plot_winding_xy(x, y, freq=440.0, title="Meu Winding Custom")


def analyze_audio_winding_detailed(filepath: str, freq_range: tuple = (20, 500),
                                   n_freqs: int = 100, duration: float = 2.0,
                                   min_prominence: float = 0.1):
    """
    Análise detalhada com detecção de MÚLTIPLAS frequências dominantes.
    """
    fs, signal = load_audio(filepath)

    if signal.ndim > 1:
        signal = signal[:, 0]

    max_samples = int(fs * duration)
    signal = signal[:max_samples]
    signal = signal / np.max(np.abs(signal))

    frequencies = np.linspace(freq_range[0], freq_range[1], n_freqs)
    centroid_distances = []

    print(
        f"Analisando {n_freqs} frequências de {freq_range[0]} a {freq_range[1]} Hz...")

    for freq in frequencies:
        x, y = generate_winding_data(signal, fs, freq)
        cx, cy = np.mean(x), np.mean(y)
        distance = np.sqrt(cx**2 + cy**2)
        centroid_distances.append(distance)

    centroid_distances = np.array(centroid_distances)

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(
        centroid_distances,
        height=np.max(centroid_distances) *
        min_prominence,  
        distance=5,  
        prominence=np.max(centroid_distances) * 0.05  
    )

    dominant_freqs = frequencies[peaks]
    dominant_strengths = centroid_distances[peaks]

    sorted_indices = np.argsort(dominant_strengths)[::-1]
    dominant_freqs = dominant_freqs[sorted_indices]
    dominant_strengths = dominant_strengths[sorted_indices]

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(frequencies, centroid_distances, 'b-', linewidth=1.5, alpha=0.7)
    plt.scatter(dominant_freqs, dominant_strengths,
                color='red', s=100, zorder=5)

    for i, (freq, strength) in enumerate(zip(dominant_freqs[:5], dominant_strengths[:5])):
        plt.annotate(f'{freq:.1f} Hz',
                     xy=(freq, strength),
                     xytext=(freq, strength + 0.02),
                     ha='center', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Força da Frequência (Distância do Centróide)")
    plt.title(
        f"Análise Winding - Todas as Frequências Detectadas ({duration}s)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    if len(dominant_freqs) > 0:
        top_freqs = dominant_freqs[:10]
        top_strengths = dominant_strengths[:10]

        bars = plt.bar(range(len(top_freqs)), top_strengths,
                       color='orange', alpha=0.7)
        plt.xticks(range(len(top_freqs)), [
                   f'{f:.1f}' for f in top_freqs], rotation=45)
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Força")
        plt.title("Top 10 Frequências Mais Fortes")
        plt.grid(True, alpha=0.3)

        for i, (bar, strength) in enumerate(zip(bars, top_strengths)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{strength:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"\n FREQUÊNCIAS DETECTADAS (Total: {len(dominant_freqs)}):")
    print("="*50)
    for i, (freq, strength) in enumerate(zip(dominant_freqs, dominant_strengths)):
        print(f"{i+1:2d}. {freq:6.1f} Hz - Força: {strength:.4f}")
        if i >= 9:  # Mostrar só as top 10
            print(f"    ... e mais {len(dominant_freqs)-10} frequências")
            break

    return dominant_freqs, dominant_strengths



def analyze_audio_winding_temporal(filepath: str, window_duration: float = 0.5,
                                   overlap: float = 0.25, freq_range: tuple = (20, 300)):
    """
    Análise temporal: detecta frequências dominantes em múltiplas janelas.
    """
    fs, signal = load_audio(filepath)

    if signal.ndim > 1:
        signal = signal[:, 0]

    signal = signal / np.max(np.abs(signal))

    window_samples = int(fs * window_duration)
    step_samples = int(fs * overlap)

    windows = []
    window_times = []

    for start in range(0, len(signal) - window_samples, step_samples):
        end = start + window_samples
        windows.append(signal[start:end])
        window_times.append(start / fs)

    print(f"Analisando {len(windows)} janelas de {window_duration}s cada...")

    all_results = []
    for i, window in enumerate(windows):
        frequencies = np.linspace(freq_range[0], freq_range[1], 50)
        centroid_distances = []

        for freq in frequencies:
            x, y = generate_winding_data(window, fs, freq)
            cx, cy = np.mean(x), np.mean(y)
            distance = np.sqrt(cx**2 + cy**2)
            centroid_distances.append(distance)

        all_results.append(centroid_distances)

    all_results = np.array(all_results)

    plt.figure(figsize=(15, 8))
    plt.imshow(all_results.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Força da Frequência')
    plt.xlabel('Janela Temporal')
    plt.ylabel('Frequência (Hz)')
    plt.title(
        f'Evolução Temporal das Frequências (janelas de {window_duration}s)')

    plt.yticks(range(0, len(frequencies), 10),
               [f'{frequencies[i]:.0f}' for i in range(0, len(frequencies), 10)])
    plt.xticks(range(0, len(window_times), max(1, len(window_times)//10)),
               [f'{window_times[i]:.1f}s' for i in range(0, len(window_times), max(1, len(window_times)//10))])

    plt.tight_layout()
    plt.show()

    return all_results, frequencies, window_times


def analyze_winding_universal(signal=None, fs=None, filepath=None,
                              freq_range=(50, 500), n_freqs=20, duration=2.0,
                              plot_curves=True, plot_spectrum=True):
    """
    Função universal para análise winding - aceita sinal OU filepath.
    
    Parameters:
        signal: array numpy do sinal (opcional)
        fs: taxa de amostragem (obrigatório se signal fornecido)
        filepath: caminho do arquivo de áudio (opcional)
        freq_range: faixa de frequências para análise
        n_freqs: número de frequências a testar
        duration: duração do sinal em segundos
        plot_curves: se deve plotar as curvas winding
        plot_spectrum: se deve plotar o espectro de frequências
    
    Returns:
        dict com frequências dominantes, distâncias e dados completos
    """

    if signal is not None and fs is not None:
        audio_signal = signal.copy()
        sample_rate = fs
    elif filepath is not None:
        sample_rate, audio_signal = load_audio(filepath)
    else:
        raise ValueError("Forneça 'signal + fs' OU 'filepath'")

    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]

    max_samples = int(sample_rate * duration)
    audio_signal = audio_signal[:max_samples]

    audio_signal = audio_signal / np.max(np.abs(audio_signal))


    frequencies = np.linspace(freq_range[0], freq_range[1], n_freqs)
    centroid_distances = []
    winding_data = [] 

    print(
        f"Analisando {n_freqs} frequências de {freq_range[0]} a {freq_range[1]} Hz...")

    for freq in frequencies:
        x, y = generate_winding_data(audio_signal, sample_rate, freq)
        cx, cy = np.mean(x), np.mean(y)
        distance = np.sqrt(cx**2 + cy**2)

        centroid_distances.append(distance)
        winding_data.append({
            'frequency': freq,
            'x': x, 'y': y,
            'centroid': (cx, cy),
            'distance': distance
        })

    centroid_distances = np.array(centroid_distances)

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(
        centroid_distances,
        height=np.max(centroid_distances) * 0.3,
        distance=max(1, n_freqs // 10)  
    )

    dominant_freqs = frequencies[peaks]
    dominant_distances = centroid_distances[peaks]

    if plot_spectrum:
        plt.figure(figsize=(12, 4))
        plt.plot(frequencies, centroid_distances, 'b-', linewidth=2, alpha=0.7)
        plt.scatter(dominant_freqs, dominant_distances,
                    color='red', s=100, zorder=5)

        for freq, dist in zip(dominant_freqs, dominant_distances):
            plt.annotate(f'{freq:.1f} Hz',
                         xy=(freq, dist), xytext=(freq, dist + 0.01),
                         ha='center', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Força da Frequência (Distância do Centróide)")
        plt.title("Análise Winding - Detecção de Frequências Dominantes")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    if plot_curves and len(dominant_freqs) > 0:
        n_plots = min(4, len(dominant_freqs))  
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))

        if n_plots == 1:
            axes = [axes]

        for i in range(n_plots):
            freq = dominant_freqs[i]
            freq_data = next(d for d in winding_data if abs(
                d['frequency'] - freq) < 0.1)
            x, y = freq_data['x'], freq_data['y']
            cx, cy = freq_data['centroid']

            axes[i].plot(x, y, color='mediumturquoise',
                         linewidth=1.5, alpha=0.7, zorder=1)
            axes[i].scatter(cx, cy, color='red', s=150, zorder=10,
                            edgecolors='darkred', linewidth=2, label='Centroid')
            axes[i].scatter(0, 0, color='black', s=50,
                            marker='x', zorder=10)  # Origem

            axes[i].set_title(f"Winding - {freq:.1f} Hz", fontsize=14)
            axes[i].set_xlabel("Real Axis")
            axes[i].set_ylabel("Imaginary Axis")
            axes[i].axis('equal')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    results = {
        'frequencies': frequencies,
        'centroid_distances': centroid_distances,
        'dominant_frequencies': dominant_freqs,
        'dominant_distances': dominant_distances,
        'winding_data': winding_data,
        'signal_info': {
            'duration': duration,
            'sample_rate': sample_rate,
            'n_samples': len(audio_signal)
        }
    }

    print(f"\nFREQUÊNCIAS DOMINANTES DETECTADAS: {len(dominant_freqs)}")
    for i, (freq, dist) in enumerate(zip(dominant_freqs, dominant_distances)):
        print(f"{i+1}. {freq:6.1f} Hz - Força: {dist:.4f}")

    return results




def analyze_from_file(filepath, **kwargs):
    """Analisar diretamente de arquivo."""
    return analyze_winding_universal(filepath=filepath, **kwargs)


def analyze_from_signal(signal, fs, **kwargs):
    """Analisar de sinal já carregado."""
    return analyze_winding_universal(signal=signal, fs=fs, **kwargs)


def quick_winding_analysis(signal, fs, target_freq):
    """Análise rápida de uma frequência específica."""
    x, y = generate_winding_data(signal, fs, target_freq)
    cx, cy = np.mean(x), np.mean(y)
    distance = np.sqrt(cx**2 + cy**2)

    plot_winding_xy_enhanced(x, y, freq=target_freq, show_center=True)

    return {'centroid': (cx, cy), 'distance': distance}


analyze_winding_universal(filepath=wav_path, duration=2.5)

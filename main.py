import DFT, FFT, QFT
import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor

def calculate_DFT(signal: list):
    print("Starting DFT computation...")
    dft = DFT.DFT(signal)
    time_start = time.time()
    result = dft.compute_dft()
    time_end = time.time()
    print(f"DFT computation time: {time_end - time_start} seconds")

    return range(len(result)), result

def calculate_FFT(signal: list):
    print("Starting FFT computation...")
    fft = FFT.FFT(signal)
    time_start = time.time()
    result = fft.compute_fft()
    time_end = time.time()
    print(f"FFT computation time: {time_end - time_start} seconds")

    return range(len(result)), result 

def calculate_QFT(signal: list):
    print("Starting QFT computation...")
    qft = QFT.QFT(signal)

    # print("QFT circuit:")
    # print(qft.draw())

    # print("\nInput (time-domain) amplitudes |x⟩:")
    # print(qft.time_state.data)

    service = QiskitRuntimeService()
    backend = service.backend("ibm_torino")

    time_start = time.time()
    counts = qft.measure_counts(backend)
    time_end = time.time()
    print(f"QFT computation time: {time_end - time_start} seconds")

    states = sorted(counts.keys())
    values = [counts[s] for s in states]
    total_shots = sum(values)

    probs = [v / total_shots for v in values]

    freqs = [int(s, 2) for s in states]
    mags = [p**0.5 for p in probs]

    return freqs, mags

def plot_spectrum(freq, mag, title):
    plt.figure(figsize=(8, 4))
    plt.bar(freq, mag, color='royalblue', edgecolor='black')

    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.ylim(0, max(mag) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

def main():
    # Simple test
    total_N = 8
    signal = [1] * total_N

    # Longer test with audio file
    # sampling, data = wavfile.read("C major chord.wav")
    # print(f"Sampling Rate: {sampling} Hz")
    # signal = data[:2**8]

    with ThreadPoolExecutor(max_workers=1) as executor:
        # kick off remote QFT work
        future_qft = executor.submit(calculate_QFT, signal)

        # local CPU work
        freqs_fft, mag_fft = calculate_FFT(signal)
        freqs_dft, mag_dft = calculate_DFT(signal)

        # wait for QFT only when needed
        freqs_qft, mag_qft = future_qft.result()

    plot_spectrum(freqs_dft, mag_dft, "DFT Spectrum")
    plot_spectrum(freqs_fft, mag_fft, "FFT Spectrum")
    plot_spectrum(freqs_qft, mag_qft, "QFT Spectrum")

    plt.show()

if __name__ == "__main__":
    main()


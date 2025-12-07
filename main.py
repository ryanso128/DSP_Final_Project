import DFT, FFT, QFT
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    total_N = 16
    signal = [1] * total_N
    dft = DFT.DFT(signal)
    time_start = time.time()
    result = dft.compute_dft()
    time_end = time.time()
    print(f"DFT computation time: {time_end - time_start} seconds")

    markerline, stemline, baseline = plt.stem(range(len(result)), result)
    plt.title("DFT Result")
    baseline.set_color("black")
    plt.show()

    fft = FFT.FFT(signal)
    time_start = time.time()
    result = fft.compute_fft()
    time_end = time.time()
    print(f"FFT computation time: {time_end - time_start} seconds")

    markerline, stemline, baseline = plt.stem(range(len(result)), result)
    plt.title("FFT Result")
    baseline.set_color("black")
    plt.show()


if __name__ == "__main__":
    main()


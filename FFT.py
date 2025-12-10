import numpy as np

class FFT:
    def __init__(self, signal: list):
        # Store signal as complex128 to avoid integer overflow
        self.signal = np.asarray(signal, dtype=np.complex128)
        self.N = len(self.signal)

    def _fft_recursive(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        if N == 1:
            return x

        # Even and odd parts
        X_even = self._fft_recursive(x[0::2])
        X_odd  = self._fft_recursive(x[1::2])

        # Twiddle factors for this stage (depend on current N, not total length)
        k = np.arange(N // 2)
        twiddle = np.exp(-2j * np.pi * k / N) * X_odd

        # Combine (this returns outputs in standard order – no bit-reversal needed)
        return np.concatenate([X_even + twiddle,
                               X_even - twiddle])

    def compute_fft(self) -> list:
        X = self._fft_recursive(self.signal)
        mags = np.abs(X)

        # Optional: zero-out tiny numerical noise
        mags[np.isclose(mags, 0)] = 0.0

        # Return as plain Python list, like your original function
        return mags.tolist()
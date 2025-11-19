import numpy as np

class DFT:
    def __init__ (self, signal: list):
        self.signal = signal
        self.N = len(signal)
    
    def compute_dft(self) -> list:
        dft_result = []
        for k in range(self.N):
            sum_val = 0
            for n in range(self.N):
                angle = 2j * np.pi * k * n / self.N
                sum_val += self.signal[n] * np.exp(-angle)
            magnitude = np.abs(sum_val)
            if np.isclose(magnitude, 0): 
                magnitude = 0
            dft_result.append(float(magnitude))
        return dft_result
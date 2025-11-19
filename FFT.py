import numpy as np

class FFT:
    def __init__ (self, signal: list):
        self.signal = signal
        self.N = len(signal)
    
    def compute_fft(self) -> list:
        # Decimation in frequency FFT implementation
        def fft_recursive(x):
            first_half = x[0:len(x) // 2]
            second_half = x[len(x) // 2:]
            return_list = []
            twiddle_factors = [np.exp(-2j * np.pi * k / self.N) for k in range(len(first_half))]

            # Butterfly computations
            for i in range(len(first_half)):
                return_list.append(first_half[i] + second_half[i])
            for i in range(len(first_half)):
                return_list.append((first_half[i] - second_half[i]) * twiddle_factors[i])
            
            if len(return_list) <= 2:
                return return_list
            else:
                half_size = len(return_list) // 2
                return fft_recursive(return_list[0:half_size]) + fft_recursive(return_list[half_size:])

        # Get the magnitudes of the FFT results
        return_list = fft_recursive(self.signal)
        for i in range(len(return_list)):
            magnitude = np.abs(return_list[i])
            if np.isclose(magnitude, 0): 
                magnitude = 0
            return_list[i] = float(magnitude)
        
        # Bit-reverse the order of the results
        flipped = set() 
        for i in range(len(return_list) // 2):
            bitflip_index = int('{:0{width}b}'.format(i, width=int(np.log2(self.N)))[::-1], 2)
            if i not in flipped and bitflip_index not in flipped:
                return_list[i], return_list[bitflip_index] = return_list[bitflip_index], return_list[i]
                flipped.add(i)
                flipped.add(bitflip_index)
        
        return return_list

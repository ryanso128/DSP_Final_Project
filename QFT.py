import numpy as np
from math import pi
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt


class QFT:
    def __init__(self, signal: np.ndarray):
        signal = np.asarray(signal, dtype=complex)
        N = signal.size

        # Check length is power of 2
        if N & (N - 1) != 0:
            raise ValueError("Signal length must be a power of 2 (2^n).")

        self.N = N
        self.n_qubits = int(np.log2(N))

        # Normalize to create a valid quantum state
        norm = np.linalg.norm(signal)
        if norm == 0:
            raise ValueError("Signal has zero norm; cannot normalize.")
        self.time_amplitudes = signal / norm

        # Build circuit: initialize state and apply manual QFT
        self.qc = QuantumCircuit(self.n_qubits)
        self.qc.initialize(self.time_amplitudes, range(self.n_qubits))
        self._manual_qft()

        # Store statevectors
        self._time_state = Statevector(self.time_amplitudes)
        self._fourier_state = Statevector.from_instruction(self.qc)

    def _manual_qft(self, qubits=None):
        if qubits is None:
            qubits = list(range(self.qc.num_qubits))
        else:
            qubits = list(qubits)

        n = len(qubits)

        for j in range(n):
            k = qubits[j]
            self.qc.h(k)
            # Controlled phase rotations
            for m in range(j + 1, n):
                l = qubits[m]
                angle = pi / (2 ** (m - j))
                self.qc.cp(angle, l, k)

        # Bit-reversal via swaps
        for i in range(n // 2):
            self.qc.swap(qubits[i], qubits[n - i - 1])

    # -------------------------
    # Public helpers
    # -------------------------
    @property
    def time_state(self) -> Statevector:
        """Statevector of the input (time-domain) signal |x⟩."""
        return self._time_state

    @property
    def fourier_state(self) -> Statevector:
        """Statevector after QFT, i.e., |X⟩ = QFT|x⟩."""
        return self._fourier_state

    @property
    def fourier_amplitudes(self) -> np.ndarray:
        """Return Fourier-domain amplitudes as a numpy array."""
        return self._fourier_state.data

    def draw(self, *args, **kwargs):
        """Draw the QFT circuit."""
        return self.qc.draw(*args, **kwargs)

    def measure_counts(self, backend, shots: int = 4096):

        qc_meas = self.qc.copy()
        qc_meas.measure_all()

        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        isa_qc = pm.run(qc_meas)

        sampler = Sampler(backend)
        # SamplerV2 takes a list of "pubs"; simplest is just [qc_meas]
        job = sampler.run([isa_qc], shots=shots)
        pub_result = job.result()[0]

        # Combined counts over all classical registers
        return pub_result.join_data().get_counts()
    

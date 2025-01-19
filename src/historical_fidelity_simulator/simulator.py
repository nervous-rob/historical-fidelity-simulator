"""Core simulator module implementing the Generalized Historical Fidelity framework."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.linalg import expm
import qutip as qt
from .utils.gpu_accelerator import GPUAccelerator, HAS_CUPY


class SimulationMode(Enum):
    """Simulation mode enumeration."""
    CLASSICAL = "classical"
    QUANTUM = "quantum"


class GeneralizedHistoricalSimulator:
    """Main simulator class implementing both classical and quantum approaches."""

    def __init__(
        self,
        n_sites: int,
        coupling_strength: float,
        field_strength: float,
        temperature: float,
        hbar_h: float,
        mode: SimulationMode = SimulationMode.CLASSICAL,
        f_scaling: Optional[Callable[[float, float], float]] = None,
        use_gpu: bool = True
    ) -> None:
        """Initialize the simulator.

        Args:
            n_sites: Number of sites in the system
            coupling_strength: Interaction strength between sites (J)
            field_strength: External field strength (h)
            temperature: System temperature (T)
            hbar_h: Information Planck constant
            mode: Simulation mode (classical or quantum)
            f_scaling: Custom scaling function f(E, T)
            use_gpu: Whether to use GPU acceleration if available

        Raises:
            ValueError: If any parameters are invalid
        """
        # Validate parameters
        if n_sites <= 0:
            raise ValueError("Number of sites must be positive")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if hbar_h <= 0:
            raise ValueError("Information Planck constant must be positive")
        if not isinstance(mode, SimulationMode):
            raise ValueError("Invalid simulation mode")

        self.n_sites = n_sites
        self.J = coupling_strength
        self.h = field_strength
        self.T = temperature
        self.hbar_h = hbar_h
        self.beta = 1.0 / self.T
        self.mode = mode

        # Initialize GPU accelerator
        self.gpu = GPUAccelerator(n_sites, use_gpu) if mode == SimulationMode.CLASSICAL else None
        self.use_gpu = use_gpu and (self.gpu is not None or (mode == SimulationMode.QUANTUM and HAS_CUPY))

        # Default scaling function if none provided
        # f(E, T) = 1 + E/T + α(T/Tₖ)^β
        self.f_scaling = f_scaling or self._default_scaling_function

        # Initialize state based on mode
        if self.mode == SimulationMode.CLASSICAL:
            self.state = np.random.choice([-1, 1], size=n_sites).astype(np.int32)
            if self.use_gpu:
                self.energies = self.gpu.compute_classical_energies(self.state, self.J, self.h)
        else:
            # For quantum mode, create a product state |0...0⟩
            self.state = qt.basis([2]*n_sites, [0]*n_sites)
            self.hamiltonian = self._construct_quantum_hamiltonian()
            if self.use_gpu and HAS_CUPY:
                # Convert quantum operators to CuPy arrays for GPU acceleration
                import cupy as cp
                # Convert the entire Hamiltonian to a dense numpy array first
                ham_dense = self.hamiltonian.full()
                # Then convert to CuPy array
                self.hamiltonian_gpu = cp.asarray(ham_dense)
                # Also convert initial state
                self.state_gpu = cp.asarray(self.state.full())
            else:
                self.hamiltonian_gpu = None
                self.state_gpu = None

        self.time = 0.0
        self.history: List[Dict] = []

    def _default_scaling_function(self, energy: float, temperature: float) -> float:
        """Default implementation of the scaling function.
        
        f(E, T) = 1 + E/T + α(T/Tₖ)^β

        Args:
            energy: System energy
            temperature: Current temperature

        Returns:
            Scaling factor
        """
        T_c = 3.0  # Critical temperature
        alpha = 0.5  # Scaling prefactor
        beta = -0.8  # Critical exponent
        
        return 1.0 + energy / temperature + alpha * (temperature / T_c) ** beta

    def _construct_quantum_hamiltonian(self) -> qt.Qobj:
        """Construct the quantum Hamiltonian using QuTiP.
        
        Returns:
            QuTiP Quantum object representing the Hamiltonian
        """
        # Create operators for each site
        sx_list = [qt.sigmax() for _ in range(self.n_sites)]
        sz_list = [qt.sigmaz() for _ in range(self.n_sites)]
        
        # Initialize Hamiltonian
        H = 0
        
        # Nearest-neighbor coupling: -J Σᵢ σˣᵢσˣᵢ₊₁
        for i in range(self.n_sites):
            j = (i + 1) % self.n_sites
            # Create the tensor product for sites i and j
            op_list = [qt.qeye(2)] * self.n_sites
            op_list[i] = sx_list[i]
            op_list[j] = sx_list[j]
            H += -self.J * qt.tensor(op_list)
        
        # Transverse field: -h Σᵢ σᶻᵢ
        for i in range(self.n_sites):
            op_list = [qt.qeye(2)] * self.n_sites
            op_list[i] = sz_list[i]
            H += -self.h * qt.tensor(op_list)
        
        return H

    def _compute_local_energy(self, site: int) -> float:
        """Compute local energy contribution from a single site.
        
        Args:
            site: Site index
            
        Returns:
            Local energy contribution
        """
        if self.mode != SimulationMode.CLASSICAL:
            raise ValueError("Local energy only defined for classical mode")
            
        if self.use_gpu:
            energies = self.gpu.compute_classical_energies(self.state, self.J, self.h)
            return energies[site]
        else:
            left = (site - 1) % self.n_sites
            right = (site + 1) % self.n_sites
            spin = self.state[site]
            neighbor_sum = self.state[left] + self.state[right]
            return -self.J * spin * neighbor_sum - self.h * spin

    def _compute_quantum_fidelity(self) -> float:
        """Compute fidelity for quantum state.
        
        Returns:
            Quantum fidelity measure
        """
        if self.use_gpu and HAS_CUPY:
            import cupy as cp
            exp_H = cp.real(cp.vdot(self.state_gpu, cp.matmul(self.hamiltonian_gpu, self.state_gpu)))
            return -float(exp_H) / self.n_sites
        else:
            exp_H = qt.expect(self.hamiltonian, self.state)
            return -exp_H / self.n_sites

    def _compute_fidelity(self) -> float:
        """Compute current historical fidelity metric.
        
        Returns:
            Current fidelity value
        """
        if self.mode == SimulationMode.CLASSICAL:
            total_energy = sum(
                self._compute_local_energy(i) for i in range(self.n_sites)
            )
            return -total_energy / self.n_sites
        else:
            return self._compute_quantum_fidelity()

    def _evolve_quantum_state(self, dt: float) -> None:
        """Evolve quantum state with Hamiltonian + Lindblad decoherence.
        
        Args:
            dt: Time step
        """
        if self.use_gpu and HAS_CUPY:
            import cupy as cp
            from .utils.gpu_accelerator import matrix_exp_gpu
            # Simple Euler step for now (can be improved with better integrators)
            # dρ/dt = -i[H,ρ] + T Σᵢ (σᶻᵢρσᶻᵢ - ρ)
            # For pure states, we can use the Schrödinger equation
            # dψ/dt = -iHψ - T/2 ψ
            evolution_matrix = -1j * dt * self.hamiltonian_gpu - dt * self.T/2 * cp.eye(self.hamiltonian_gpu.shape[0])
            propagator = matrix_exp_gpu(evolution_matrix)
            self.state_gpu = cp.matmul(propagator, self.state_gpu)
            # Normalize
            self.state_gpu /= cp.sqrt(cp.vdot(self.state_gpu, self.state_gpu))
        else:
            # Create decoherence operators: √T σᶻ for each site
            c_ops = []
            for i in range(self.n_sites):
                op_list = [qt.qeye(2)] * self.n_sites
                op_list[i] = np.sqrt(self.T) * qt.sigmaz()
                c_ops.append(qt.tensor(op_list))

            result = qt.mesolve(
                self.hamiltonian,
                self.state,
                [0, dt],
                c_ops=c_ops
            )
            self.state = result.states[-1]

    def compute_generalized_bound(self) -> float:
        """Compute ℏ_h * f(⟨E⟩, T).
        
        Returns:
            Current value of the generalized bound
        """
        avg_energy = abs(self._compute_fidelity())
        return self.hbar_h * self.f_scaling(avg_energy, self.T)

    def run_simulation(
        self,
        n_steps: int,
        dt: float,
        measure_interval: int = 100
    ) -> List[Dict]:
        """Run the simulation for a specified number of steps.
        
        Args:
            n_steps: Number of time steps
            dt: Time step size
            measure_interval: Number of steps between measurements
            
        Returns:
            List of measurement dictionaries containing time, fidelity, and bound
        """
        self.history.clear()
        
        for step in range(n_steps):
            if self.mode == SimulationMode.CLASSICAL:
                if self.use_gpu:
                    # GPU-accelerated Metropolis update
                    self.state, self.energies, _ = self.gpu.metropolis_update(
                        self.state, self.energies, self.J, self.h, self.T
                    )
                else:
                    # Single-site Metropolis update
                    site = np.random.randint(0, self.n_sites)
                    old_energy = self._compute_local_energy(site)
                    self.state[site] *= -1
                    new_energy = self._compute_local_energy(site)
                    
                    delta_E = new_energy - old_energy
                    if delta_E > 0 and np.random.random() >= np.exp(-self.beta * delta_E):
                        self.state[site] *= -1  # Reject move
            else:
                self._evolve_quantum_state(dt)
            
            self.time += dt
            
            if step % measure_interval == 0:
                self.history.append({
                    'time': self.time,
                    'fidelity': self._compute_fidelity(),
                    'bound': self.compute_generalized_bound()
                })
        
        return self.history 
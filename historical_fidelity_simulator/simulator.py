"""Core simulator module implementing the Generalized Historical Fidelity framework."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.linalg import expm
import qutip as qt


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
        f_scaling: Optional[Callable[[float, float], float]] = None
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

        # Default scaling function if none provided
        # f(E, T) = 1 + E/T + α(T/Tₖ)^β
        self.f_scaling = f_scaling or self._default_scaling_function

        # Initialize state based on mode
        if self.mode == SimulationMode.CLASSICAL:
            self.state = np.random.choice([-1, 1], size=n_sites)
        else:
            # For quantum mode, create a product state |0...0⟩
            self.state = qt.basis([2]*n_sites, [0]*n_sites)
            self.hamiltonian = self._construct_quantum_hamiltonian()

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
        """Compute energy contribution from a single site (classical mode).
        
        Args:
            site: Site index

        Returns:
            Local energy contribution
        
        Raises:
            ValueError: If called in quantum mode
        """
        if self.mode == SimulationMode.QUANTUM:
            raise ValueError("Local energy undefined in quantum mode")

        neighbors = [(site + 1) % self.n_sites, (site - 1) % self.n_sites]
        local_energy = -self.h * self.state[site]
        local_energy += -self.J * sum(
            self.state[site] * self.state[n] for n in neighbors
        )
        return local_energy

    def _compute_quantum_fidelity(self) -> float:
        """Compute fidelity for quantum state.
        
        Returns:
            Quantum fidelity measure
        """
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
            n_steps: Number of simulation steps
            dt: Time step size
            measure_interval: Interval between measurements

        Returns:
            List of dictionaries containing simulation history
        """
        self.history.clear()

        for step in range(n_steps):
            if self.mode == SimulationMode.CLASSICAL:
                # Metropolis update
                site = np.random.randint(0, self.n_sites)
                old_e = self._compute_local_energy(site)
                self.state[site] *= -1
                new_e = self._compute_local_energy(site)

                delta_e = new_e - old_e
                if delta_e > 0 and np.random.random() >= np.exp(-self.beta * delta_e):
                    # Reject flip
                    self.state[site] *= -1
            else:
                self._evolve_quantum_state(dt)

            self.time += dt

            # Periodic measurement
            if step % measure_interval == 0:
                fidelity = self._compute_fidelity()
                bound = self.compute_generalized_bound()

                self.history.append({
                    'time': self.time,
                    'fidelity': fidelity,
                    'bound': bound,
                    'step': step
                })

        return self.history 
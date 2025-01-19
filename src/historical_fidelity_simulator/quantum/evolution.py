"""Quantum evolution module for the historical fidelity simulator.

This module handles the time evolution of quantum states using QuTiP's
master equation solver, including both unitary evolution and decoherence.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import qutip as qt
from .operators import construct_lindblad_operators, validate_temperature


def validate_evolution_params(dt: float, n_samples: Optional[int] = None) -> None:
    """Validate evolution parameters.
    
    Args:
        dt: Time step for evolution
        n_samples: Optional number of samples for uncertainty calculation
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters have wrong type
    """
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be numeric")
    if dt <= 0:
        raise ValueError("dt must be positive")
    
    if n_samples is not None:
        if not isinstance(n_samples, int):
            raise TypeError("n_samples must be an integer")
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")


def validate_quantum_state(state: qt.Qobj, n_sites: int) -> None:
    """Validate a quantum state.
    
    Args:
        state: Quantum state to validate
        n_sites: Expected number of sites
        
    Raises:
        ValueError: If state is invalid
        TypeError: If state has wrong type
    """
    if not isinstance(state, qt.Qobj):
        raise TypeError("state must be a QuTiP Qobj")
    
    expected_dims = [[2] * n_sites, [1] * n_sites]  # For pure states
    expected_dims_mixed = [[2] * n_sites, [2] * n_sites]  # For mixed states
    
    if state.dims != expected_dims and state.dims != expected_dims_mixed:
        raise ValueError(f"state dimensions {state.dims} do not match n_sites={n_sites}")


class QuantumEvolver:
    """Class to handle quantum state evolution."""
    
    def __init__(
        self,
        hamiltonian: qt.Qobj,
        n_sites: int,
        temperature: float,
        decoherence_strength: Optional[float] = None
    ):
        """Initialize the quantum evolver.
        
        Args:
            hamiltonian: The system Hamiltonian
            n_sites: Number of sites in the system
            temperature: Temperature of the environment
            decoherence_strength: Optional scaling factor for decoherence.
                If None, uses sqrt(temperature)
                
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        if not isinstance(hamiltonian, qt.Qobj):
            raise TypeError("hamiltonian must be a QuTiP Qobj")
        if not hamiltonian.isherm:
            raise ValueError("hamiltonian must be Hermitian")
            
        expected_dims = [[2] * n_sites, [2] * n_sites]
        if hamiltonian.dims != expected_dims:
            raise ValueError(f"hamiltonian dimensions {hamiltonian.dims} do not match n_sites={n_sites}")
        
        validate_temperature(temperature)
        
        if decoherence_strength is not None:
            if not isinstance(decoherence_strength, (int, float)):
                raise TypeError("decoherence_strength must be numeric")
            if decoherence_strength < 0:
                raise ValueError("decoherence_strength must be non-negative")
        
        self.hamiltonian = hamiltonian
        self.n_sites = n_sites
        self.temperature = temperature
        self.c_ops = construct_lindblad_operators(
            n_sites,
            temperature,
            decoherence_strength
        )
        
    def evolve_state(
        self,
        initial_state: qt.Qobj,
        dt: float,
        store_states: bool = False
    ) -> Tuple[qt.Qobj, Optional[Dict[str, Any]]]:
        """Evolve the quantum state forward in time.
        
        Args:
            initial_state: Initial quantum state
            dt: Time step for evolution
            store_states: Whether to store intermediate states
        
        Returns:
            Tuple of (final state, optional dict with evolution data)
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        validate_quantum_state(initial_state, self.n_sites)
        validate_evolution_params(dt)
        
        # Set up the master equation solver with options as dict
        solver_options = {'store_states': store_states}
        result = qt.mesolve(
            H=self.hamiltonian,
            rho0=initial_state,
            tlist=[0, dt],
            c_ops=self.c_ops,
            options=solver_options
        )
        
        # Get the final state
        final_state = result.states[-1]
        
        # Prepare evolution data if requested
        evolution_data = None
        if store_states:
            evolution_data = {
                'times': result.times,
                'states': result.states,
                'expect': []  # Can add expectation values if needed
            }
            
        return final_state, evolution_data
    
    def compute_fidelity(self, state: qt.Qobj) -> float:
        """Compute the historical fidelity metric for a quantum state.
        
        For quantum systems, we define fidelity as the negative expectation
        value of the Hamiltonian (normalized by system size), analogous to
        the classical case.
        
        Args:
            state: Quantum state to compute fidelity for
        
        Returns:
            The fidelity metric
            
        Raises:
            ValueError: If state is invalid
            TypeError: If state has wrong type
        """
        validate_quantum_state(state, self.n_sites)
        energy = qt.expect(self.hamiltonian, state)
        return -energy / self.n_sites
    
    def compute_uncertainty_product(
        self,
        initial_state: qt.Qobj,
        dt: float,
        n_samples: int = 10
    ) -> float:
        """Compute the uncertainty product ΔH_f Δt.
        
        Args:
            initial_state: Initial quantum state
            dt: Time interval
            n_samples: Number of intermediate points to sample
        
        Returns:
            The uncertainty product
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        validate_quantum_state(initial_state, self.n_sites)
        validate_evolution_params(dt, n_samples)
        
        # Evolve with intermediate points
        times = np.linspace(0, dt, n_samples)
        # Use dict for solver options
        solver_options = {'store_states': True}
        result = qt.mesolve(
            H=self.hamiltonian,
            rho0=initial_state,
            tlist=times,
            c_ops=self.c_ops,
            options=solver_options
        )
        
        # Compute fidelities at each point
        fidelities = [self.compute_fidelity(state) for state in result.states]
        
        # Compute standard deviation of fidelities
        delta_hf = np.std(fidelities)
        
        return delta_hf * dt 
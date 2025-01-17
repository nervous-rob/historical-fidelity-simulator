"""Quantum operators for the historical fidelity simulator.

This module provides the quantum operators needed for the simulation,
including the transverse-field Ising Hamiltonian and Lindblad operators
for decoherence.
"""

from typing import List, Optional, Tuple
import numpy as np
import qutip as qt


def validate_system_params(n_sites: int, coupling_strength: float, field_strength: float) -> None:
    """Validate system parameters for quantum operators.
    
    Args:
        n_sites: Number of sites in the system
        coupling_strength: J, the coupling strength between neighboring spins
        field_strength: h, the transverse field strength
        
    Raises:
        ValueError: If any parameters are invalid
    """
    if not isinstance(n_sites, int):
        raise TypeError("n_sites must be an integer")
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if not isinstance(coupling_strength, (int, float)):
        raise TypeError("coupling_strength must be numeric")
    if not isinstance(field_strength, (int, float)):
        raise TypeError("field_strength must be numeric")


def validate_temperature(temperature: float) -> None:
    """Validate temperature parameter.
    
    Args:
        temperature: Temperature of the system
        
    Raises:
        ValueError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be numeric")
    if temperature <= 0:
        raise ValueError("temperature must be positive")


def construct_ising_hamiltonian(
    n_sites: int,
    coupling_strength: float,
    field_strength: float,
    periodic: bool = True
) -> qt.Qobj:
    """Construct the transverse-field Ising Hamiltonian.
    
    H = -J Σ σ_i^x σ_{i+1}^x - h Σ σ_i^z
    
    Args:
        n_sites: Number of sites in the system
        coupling_strength: J, the coupling strength between neighboring spins
        field_strength: h, the transverse field strength
        periodic: Whether to use periodic boundary conditions
    
    Returns:
        The Hamiltonian as a QuTiP Qobj
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters have wrong type
    """
    validate_system_params(n_sites, coupling_strength, field_strength)
    
    H = 0
    
    # Nearest-neighbor Ising coupling: -J Σ σ_i^x σ_{i+1}^x
    for i in range(n_sites - 1):
        j = i + 1
        # Create operator lists for tensor product
        op_list = [qt.qeye(2)] * n_sites
        op_list[i] = qt.sigmax()
        op_list[j] = qt.sigmax()
        H += -coupling_strength * qt.tensor(op_list)
    
    # Add periodic boundary term if requested
    if periodic:
        op_list = [qt.qeye(2)] * n_sites
        op_list[0] = qt.sigmax()
        op_list[-1] = qt.sigmax()
        H += -coupling_strength * qt.tensor(op_list)
    
    # Transverse field terms: -h Σ σ_i^z
    for i in range(n_sites):
        op_list = [qt.qeye(2)] * n_sites
        op_list[i] = qt.sigmaz()
        H += -field_strength * qt.tensor(op_list)
    
    return H


def construct_lindblad_operators(
    n_sites: int,
    temperature: float,
    decoherence_strength: Optional[float] = None
) -> List[qt.Qobj]:
    """Construct Lindblad operators for decoherence.
    
    Creates operators that model the system's interaction with a thermal bath,
    causing decoherence proportional to temperature.
    
    Args:
        n_sites: Number of sites in the system
        temperature: Temperature of the environment
        decoherence_strength: Optional scaling factor for decoherence rate.
            If None, uses sqrt(temperature)
    
    Returns:
        List of Lindblad operators
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters have wrong type
    """
    validate_system_params(n_sites, 1.0, 1.0)  # Validate n_sites
    validate_temperature(temperature)
    
    if decoherence_strength is not None:
        if not isinstance(decoherence_strength, (int, float)):
            raise TypeError("decoherence_strength must be numeric")
        if decoherence_strength < 0:
            raise ValueError("decoherence_strength must be non-negative")
    
    if decoherence_strength is None:
        decoherence_strength = np.sqrt(temperature)
    
    c_ops = []
    
    # Add dephasing operators (σ_z) for each site
    for i in range(n_sites):
        op_list = [qt.qeye(2)] * n_sites
        op_list[i] = qt.sigmaz()
        c_ops.append(decoherence_strength * qt.tensor(op_list))
        
    # Add energy relaxation operators (σ_-) for each site
    # Rate depends on temperature through detailed balance
    relaxation_rate = decoherence_strength * np.sqrt(1 / (1 + np.exp(-2/temperature)))
    for i in range(n_sites):
        op_list = [qt.qeye(2)] * n_sites
        op_list[i] = qt.sigmam()
        c_ops.append(relaxation_rate * qt.tensor(op_list))
        
    # Add energy excitation operators (σ_+) for each site
    # Rate satisfies detailed balance with relaxation
    excitation_rate = decoherence_strength * np.sqrt(1 / (1 + np.exp(2/temperature)))
    for i in range(n_sites):
        op_list = [qt.qeye(2)] * n_sites
        op_list[i] = qt.sigmap()
        c_ops.append(excitation_rate * qt.tensor(op_list))
    
    return c_ops


def compute_observables(
    state: qt.Qobj,
    n_sites: int
) -> Tuple[float, float]:
    """Compute relevant observables from the quantum state.
    
    Args:
        state: The quantum state as a QuTiP Qobj
        n_sites: Number of sites in the system
    
    Returns:
        Tuple of (magnetization, entropy)
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameters have wrong type
    """
    validate_system_params(n_sites, 1.0, 1.0)  # Validate n_sites
    
    if not isinstance(state, qt.Qobj):
        raise TypeError("state must be a QuTiP Qobj")
    
    expected_dims = [[2] * n_sites, [1] * n_sites]  # For pure states
    expected_dims_mixed = [[2] * n_sites, [2] * n_sites]  # For mixed states
    
    if state.dims != expected_dims and state.dims != expected_dims_mixed:
        raise ValueError(f"state dimensions {state.dims} do not match n_sites={n_sites}")
    
    # Compute magnetization (average σ_z)
    magnetization = 0
    for i in range(n_sites):
        op_list = [qt.qeye(2)] * n_sites
        op_list[i] = qt.sigmaz()
        sz_i = qt.tensor(op_list)
        magnetization += qt.expect(sz_i, state)
    magnetization /= n_sites
    
    # Compute von Neumann entropy
    # First trace out all but one site to get reduced density matrix
    rho_reduced = state.ptrace(0)
    entropy = -qt.entropy_vn(rho_reduced)
    
    return magnetization, entropy 
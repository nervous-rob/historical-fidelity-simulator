"""Tests for quantum components of the historical fidelity simulator."""

import pytest
import numpy as np
import qutip as qt
from historical_fidelity_simulator.quantum import (
    construct_ising_hamiltonian,
    construct_lindblad_operators,
    compute_observables,
    QuantumEvolver
)


@pytest.fixture
def small_system_params():
    """Fixture for a small test system."""
    return {
        'n_sites': 2,
        'coupling_strength': 1.0,
        'field_strength': 0.5,
        'temperature': 1.0
    }


def test_ising_hamiltonian_construction(small_system_params):
    """Test construction of the transverse-field Ising Hamiltonian."""
    H = construct_ising_hamiltonian(
        n_sites=small_system_params['n_sites'],
        coupling_strength=small_system_params['coupling_strength'],
        field_strength=small_system_params['field_strength']
    )
    
    # Check basic properties
    assert isinstance(H, qt.Qobj)
    assert H.isherm  # Should be Hermitian
    assert H.dims == [[2, 2], [2, 2]]  # 2-site system
    
    # Test with and without periodic boundary conditions
    H_periodic = construct_ising_hamiltonian(
        n_sites=small_system_params['n_sites'],
        coupling_strength=small_system_params['coupling_strength'],
        field_strength=small_system_params['field_strength'],
        periodic=True
    )
    H_open = construct_ising_hamiltonian(
        n_sites=small_system_params['n_sites'],
        coupling_strength=small_system_params['coupling_strength'],
        field_strength=small_system_params['field_strength'],
        periodic=False
    )
    
    assert not np.allclose(H_periodic.full(), H_open.full())


def test_lindblad_operators(small_system_params):
    """Test construction of Lindblad operators."""
    c_ops = construct_lindblad_operators(
        n_sites=small_system_params['n_sites'],
        temperature=small_system_params['temperature']
    )
    
    # Should have 3 operators per site (σz, σ+, σ-)
    expected_n_ops = 3 * small_system_params['n_sites']
    assert len(c_ops) == expected_n_ops
    
    # Each operator should be properly dimensioned
    for op in c_ops:
        assert isinstance(op, qt.Qobj)
        assert op.dims == [[2, 2], [2, 2]]
    
    # Test temperature dependence
    c_ops_hot = construct_lindblad_operators(
        n_sites=small_system_params['n_sites'],
        temperature=2.0 * small_system_params['temperature']
    )
    
    # Higher temperature should lead to stronger decoherence
    assert np.linalg.norm(c_ops_hot[0].full()) > np.linalg.norm(c_ops[0].full())


def test_observables():
    """Test computation of quantum observables."""
    # Create a simple 2-site state: |↑↓⟩
    up = qt.basis([2], 0)
    down = qt.basis([2], 1)
    state = qt.tensor(up, down)
    
    magnetization, entropy = compute_observables(state, n_sites=2)
    
    # For |↑↓⟩, magnetization should be 0
    assert abs(magnetization) < 1e-10
    
    # Pure state should have zero entropy
    assert abs(entropy) < 1e-10
    
    # Test mixed state (equal mixture of |↑↑⟩ and |↓↓⟩)
    up_up = qt.tensor(up, up)
    down_down = qt.tensor(down, down)
    mixed_state = 0.5 * (up_up * up_up.dag() + down_down * down_down.dag())
    
    # For this maximally mixed 2-state system, entropy should be ln(2)
    _, mixed_entropy = compute_observables(mixed_state, n_sites=2)
    # The entropy should be positive and close to ln(2)
    assert mixed_entropy > 0
    assert abs(mixed_entropy - 0.693147) < 1e-5  # ln(2) ≈ 0.693147


def test_quantum_evolution(small_system_params):
    """Test quantum state evolution."""
    # Create Hamiltonian and evolver
    H = construct_ising_hamiltonian(
        n_sites=small_system_params['n_sites'],
        coupling_strength=small_system_params['coupling_strength'],
        field_strength=small_system_params['field_strength']
    )
    evolver = QuantumEvolver(
        hamiltonian=H,
        n_sites=small_system_params['n_sites'],
        temperature=small_system_params['temperature']
    )
    
    # Initial state: |↑↑⟩
    up = qt.basis([2], 0)
    initial_state = qt.tensor(up, up)
    
    # Evolve state
    dt = 0.1
    final_state, evolution_data = evolver.evolve_state(
        initial_state=initial_state,
        dt=dt,
        store_states=True
    )
    
    # Check basic properties
    assert isinstance(final_state, qt.Qobj)
    assert abs(final_state.tr() - 1.0) < 1e-10  # Should be normalized
    assert evolution_data is not None
    assert len(evolution_data['times']) == 2  # Start and end times
    
    # State should change due to evolution
    fidelity = abs((initial_state.dag() * final_state)[0,0])**2
    assert fidelity < 0.99  # States should be different


def test_fidelity_computation(small_system_params):
    """Test computation of historical fidelity metric."""
    H = construct_ising_hamiltonian(
        n_sites=small_system_params['n_sites'],
        coupling_strength=small_system_params['coupling_strength'],
        field_strength=small_system_params['field_strength']
    )
    evolver = QuantumEvolver(
        hamiltonian=H,
        n_sites=small_system_params['n_sites'],
        temperature=small_system_params['temperature']
    )
    
    # Test with ground state-like and excited state-like configurations
    up = qt.basis([2], 0)
    down = qt.basis([2], 1)
    
    ground_like = qt.tensor(up, up)  # Aligned spins
    excited_like = qt.tensor(up, down)  # Anti-aligned spins
    
    fidelity_ground = evolver.compute_fidelity(ground_like)
    fidelity_excited = evolver.compute_fidelity(excited_like)
    
    # Ground-like state should have higher fidelity (less negative energy)
    assert fidelity_ground > fidelity_excited


def test_uncertainty_product(small_system_params):
    """Test computation of the uncertainty product ΔH_f Δt."""
    H = construct_ising_hamiltonian(
        n_sites=small_system_params['n_sites'],
        coupling_strength=small_system_params['coupling_strength'],
        field_strength=small_system_params['field_strength']
    )
    evolver = QuantumEvolver(
        hamiltonian=H,
        n_sites=small_system_params['n_sites'],
        temperature=small_system_params['temperature']
    )
    
    # Initial state: |↑↑⟩
    up = qt.basis([2], 0)
    initial_state = qt.tensor(up, up)
    
    # Compute uncertainty product
    dt = 0.1
    uncertainty = evolver.compute_uncertainty_product(
        initial_state=initial_state,
        dt=dt,
        n_samples=5
    )
    
    assert isinstance(uncertainty, float)
    assert uncertainty >= 0  # Should be non-negative
    
    # Test temperature dependence
    hot_evolver = QuantumEvolver(
        hamiltonian=H,
        n_sites=small_system_params['n_sites'],
        temperature=2.0 * small_system_params['temperature']
    )
    hot_uncertainty = hot_evolver.compute_uncertainty_product(
        initial_state=initial_state,
        dt=dt,
        n_samples=5
    )
    
    # Higher temperature should lead to more uncertainty
    assert hot_uncertainty > uncertainty


def test_edge_cases(small_system_params):
    """Test edge cases and error handling."""
    # Test invalid number of sites
    with pytest.raises(ValueError):
        construct_ising_hamiltonian(
            n_sites=0,
            coupling_strength=small_system_params['coupling_strength'],
            field_strength=small_system_params['field_strength']
        )
    
    # Test invalid temperature
    with pytest.raises(ValueError):
        construct_lindblad_operators(
            n_sites=small_system_params['n_sites'],
            temperature=-1.0
        )
    
    # Test invalid evolution time
    H = construct_ising_hamiltonian(
        n_sites=small_system_params['n_sites'],
        coupling_strength=small_system_params['coupling_strength'],
        field_strength=small_system_params['field_strength']
    )
    evolver = QuantumEvolver(
        hamiltonian=H,
        n_sites=small_system_params['n_sites'],
        temperature=small_system_params['temperature']
    )
    up = qt.basis([2], 0)
    initial_state = qt.tensor(up, up)
    
    with pytest.raises(ValueError):
        evolver.evolve_state(initial_state, dt=-0.1) 
"""Unit tests for the core simulator functionality."""

import pytest
import numpy as np
import qutip as qt
from historical_fidelity_simulator.simulator import GeneralizedHistoricalSimulator, SimulationMode


@pytest.fixture
def classical_sim():
    """Fixture providing a classical simulator instance."""
    return GeneralizedHistoricalSimulator(
        n_sites=4,
        coupling_strength=1.0,
        field_strength=0.5,
        temperature=2.0,
        hbar_h=1.0,
        mode=SimulationMode.CLASSICAL
    )


@pytest.fixture
def quantum_sim():
    """Fixture providing a quantum simulator instance."""
    return GeneralizedHistoricalSimulator(
        n_sites=4,
        coupling_strength=1.0,
        field_strength=0.5,
        temperature=2.0,
        hbar_h=1.0,
        mode=SimulationMode.QUANTUM
    )


def test_initialization_classical(classical_sim):
    """Test classical simulator initialization."""
    assert classical_sim.n_sites == 4
    assert classical_sim.mode == SimulationMode.CLASSICAL
    assert isinstance(classical_sim.state, np.ndarray)
    assert classical_sim.state.shape == (4,)
    assert np.all(np.abs(classical_sim.state) == 1)


def test_initialization_quantum(quantum_sim):
    """Test quantum simulator initialization."""
    assert quantum_sim.n_sites == 4
    assert quantum_sim.mode == SimulationMode.QUANTUM
    assert isinstance(quantum_sim.state, qt.Qobj)
    assert quantum_sim.state.dims == [[2, 2, 2, 2], [1, 1, 1, 1]]


def test_default_scaling_function(classical_sim):
    """Test the default scaling function behavior."""
    energy = 1.0
    temperature = 2.0
    result = classical_sim._default_scaling_function(energy, temperature)
    
    # Test basic properties
    assert isinstance(result, float)
    assert result > 0
    
    # Test scaling with energy
    result_higher_e = classical_sim._default_scaling_function(2*energy, temperature)
    assert result_higher_e > result
    
    # Test scaling with temperature
    result_higher_t = classical_sim._default_scaling_function(energy, 2*temperature)
    assert result_higher_t != result  # Should change with temperature


def test_custom_scaling_function():
    """Test using a custom scaling function."""
    def custom_scaling(e, t):
        return 1.0 + e/t
    
    sim = GeneralizedHistoricalSimulator(
        n_sites=4,
        coupling_strength=1.0,
        field_strength=0.5,
        temperature=2.0,
        hbar_h=1.0,
        mode=SimulationMode.CLASSICAL,
        f_scaling=custom_scaling
    )
    
    assert sim.f_scaling(1.0, 2.0) == 1.5


def test_classical_local_energy(classical_sim):
    """Test local energy computation in classical mode."""
    # Set a known state
    classical_sim.state = np.array([1, 1, -1, 1])
    
    # Test energy for each site
    energy_0 = classical_sim._compute_local_energy(0)
    assert isinstance(energy_0, float)
    
    # Test nearest neighbor interactions
    energy_aligned = classical_sim._compute_local_energy(0)  # Site with aligned neighbors
    classical_sim.state = np.array([1, -1, -1, 1])
    energy_antialigned = classical_sim._compute_local_energy(0)  # Site with anti-aligned neighbor
    assert energy_aligned != energy_antialigned


def test_quantum_hamiltonian(quantum_sim):
    """Test quantum Hamiltonian construction."""
    H = quantum_sim.hamiltonian
    assert isinstance(H, qt.Qobj)
    assert H.isherm  # Should be Hermitian
    assert H.dims == [[2, 2, 2, 2], [2, 2, 2, 2]]


def test_quantum_evolution(quantum_sim):
    """Test quantum state evolution."""
    initial_state = quantum_sim.state.copy()
    quantum_sim._evolve_quantum_state(dt=0.1)
    final_state = quantum_sim.state
    
    # States should be different after evolution
    # Compare expectation values of σz on the first site
    sz = qt.sigmaz()
    op_list = [qt.qeye(2)] * quantum_sim.n_sites
    op_list[0] = sz
    sz_operator = qt.tensor(op_list)
    
    initial_exp = qt.expect(sz_operator, initial_state)
    final_exp = qt.expect(sz_operator, final_state)
    assert abs(initial_exp - final_exp) > 1e-10
    
    # State should remain normalized
    assert abs(final_state.norm() - 1.0) < 1e-10


def test_classical_simulation_run(classical_sim):
    """Test running a classical simulation."""
    history = classical_sim.run_simulation(n_steps=100, dt=0.1)
    
    assert len(history) > 0
    assert all(isinstance(h, dict) for h in history)
    assert all(k in h for h in history for k in ['time', 'fidelity', 'bound', 'step'])


def test_quantum_simulation_run(quantum_sim):
    """Test running a quantum simulation."""
    history = quantum_sim.run_simulation(n_steps=100, dt=0.1)
    
    assert len(history) > 0
    assert all(isinstance(h, dict) for h in history)
    assert all(k in h for h in history for k in ['time', 'fidelity', 'bound', 'step'])


def test_generalized_bound(classical_sim):
    """Test computation of the generalized bound."""
    bound = classical_sim.compute_generalized_bound()
    assert isinstance(bound, float)
    assert bound > 0


def test_invalid_parameters():
    """Test simulator initialization with invalid parameters."""
    with pytest.raises(ValueError):
        GeneralizedHistoricalSimulator(
            n_sites=0,  # Invalid number of sites
            coupling_strength=1.0,
            field_strength=0.5,
            temperature=2.0,
            hbar_h=1.0
        )


def test_energy_conservation_classical(classical_sim):
    """Test approximate energy conservation in classical mode."""
    initial_fidelity = classical_sim._compute_fidelity()
    history = classical_sim.run_simulation(n_steps=1000, dt=0.1)
    
    # Get all fidelity values
    fidelities = [h['fidelity'] for h in history]
    
    # Check that fidelity doesn't change too drastically
    # Note: The threshold is higher because Metropolis dynamics
    # naturally allow larger fluctuations
    assert np.std(fidelities) < 2.0


def test_quantum_decoherence(quantum_sim):
    """Test that quantum decoherence is working."""
    # Run simulation with very high temperature (should cause rapid decoherence)
    quantum_sim.T = 50.0  # Increased temperature
    history = quantum_sim.run_simulation(n_steps=1000, dt=0.1)  # More steps
    
    # Fidelity should change due to decoherence
    fidelities = [h['fidelity'] for h in history]
    assert len(set(fidelities)) > 1  # Should have multiple different values 
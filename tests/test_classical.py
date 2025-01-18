"""Tests for classical simulation components."""

import numpy as np
import pytest
from historical_fidelity_simulator.classical import IsingModel, MetropolisDynamics


class TestIsingModel:
    """Test suite for IsingModel class."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple 4-site model for testing."""
        return IsingModel(
            n_sites=4,
            coupling_strength=1.0,
            field_strength=0.0,
            temperature=1.0,
            periodic=True
        )
    
    def test_initialization(self):
        """Test model initialization."""
        model = IsingModel(
            n_sites=10,
            coupling_strength=1.5,
            field_strength=0.5,
            temperature=2.0,
            periodic=True
        )
        
        assert model.n_sites == 10
        assert model.J == 1.5
        assert model.h == 0.5
        assert model.T == 2.0
        assert model.beta == 0.5
        assert model.periodic is True
        assert len(model.state) == 10
        assert all(s in [-1, 1] for s in model.state)
    
    def test_neighbors(self, simple_model):
        """Test neighbor calculations."""
        # Test middle site
        assert simple_model.get_neighbors(1) == [0, 2]
        
        # Test periodic boundaries
        assert simple_model.get_neighbors(0) == [3, 1]
        assert simple_model.get_neighbors(3) == [2, 0]
        
        # Test non-periodic case
        model = IsingModel(4, 1.0, 0.0, 1.0, periodic=False)
        assert model.get_neighbors(0) == [1]
        assert model.get_neighbors(3) == [2]
    
    def test_local_energy(self, simple_model):
        """Test local energy calculations."""
        # Set up a known configuration
        simple_model.state = np.array([1, 1, -1, -1])
        
        # Test with J=1, h=0
        # Site 0: E = -J(s₀s₁ + s₀s₃) = -(1×1 + 1×-1) = 0
        assert simple_model.local_energy(0) == 0
        
        # Add field
        simple_model.h = 0.5
        # Now E = -J(s₀s₁ + s₀s₃) - hs₀ = 0 - 0.5×1 = -0.5
        assert simple_model.local_energy(0) == -0.5
    
    def test_total_energy(self, simple_model):
        """Test total energy calculation."""
        # Set up a known configuration
        simple_model.state = np.array([1, 1, -1, -1])
        simple_model.h = 0.0
        
        # Calculate expected energy
        # E = -J Σ sᵢsⱼ = -1×(1×1 + 1×-1 + -1×-1 + -1×1) = 0
        assert simple_model.total_energy() == 0
    
    def test_magnetization(self, simple_model):
        """Test magnetization calculation."""
        simple_model.state = np.array([1, 1, -1, -1])
        assert simple_model.magnetization() == 0.0
        
        simple_model.state = np.array([1, 1, 1, -1])
        assert simple_model.magnetization() == 0.5
    
    def test_flip_spin(self, simple_model):
        """Test spin flip mechanics."""
        # Set up a known configuration
        simple_model.state = np.array([1, 1, -1, -1])
        simple_model.T = 0.0  # Zero temperature
        
        # Flipping aligned spins should be rejected (energy would increase)
        # Site 0 has one aligned neighbor (site 1) and one anti-aligned (site 3)
        de, accepted = simple_model.flip_spin(0)
        assert not accepted
        assert simple_model.state[0] == 1
        
        # Create a configuration where flipping will lower energy
        simple_model.state = np.array([1, -1, -1, -1])
        # Now site 0 has two anti-aligned neighbors, flipping it will lower energy
        de, accepted = simple_model.flip_spin(0)
        assert accepted  # Should be accepted as it lowers energy
        assert simple_model.state[0] == -1  # Spin should be flipped
        assert de < 0  # Energy change should be negative
    
    def test_get_state(self, simple_model):
        """Test state getter."""
        initial_state = simple_model.get_state()
        
        # Modify returned state
        initial_state[0] *= -1
        
        # Original state should be unchanged
        assert np.array_equal(initial_state, simple_model.state) is False


class TestMetropolisDynamics:
    """Test suite for MetropolisDynamics class."""
    
    @pytest.fixture
    def dynamics(self):
        """Create a dynamics instance for testing."""
        model = IsingModel(
            n_sites=4,
            coupling_strength=1.0,
            field_strength=0.0,
            temperature=1.0
        )
        return MetropolisDynamics(model, random_seed=42)
    
    def test_initialization(self, dynamics):
        """Test dynamics initialization."""
        assert dynamics.time == 0.0
        assert len(dynamics.history) == 0
        assert isinstance(dynamics.model, IsingModel)
    
    def test_step(self, dynamics):
        """Test single step evolution."""
        result = dynamics.step(dt=0.1)
        
        assert dynamics.time == 0.1
        assert len(dynamics.history) == 1
        assert set(result.keys()) == {
            'time', 'energy', 'magnetization',
            'energy_change', 'acceptance_rate'
        }
    
    def test_run(self, dynamics):
        """Test multiple step evolution."""
        results = dynamics.run(
            total_time=1.0,
            dt=0.1,
            measure_interval=2
        )
        
        assert len(results) == 5  # 10 steps, measuring every 2
        assert np.isclose(dynamics.time, 1.0)
        assert all('time' in r for r in results)
        assert all('energy' in r for r in results)
    
    def test_compute_fidelity(self, dynamics):
        """Test fidelity computation."""
        # Set up known configuration
        dynamics.model.state = np.array([1, 1, -1, -1])
        dynamics.model.h = 0.0
        
        # With zero energy, fidelity should be zero
        assert dynamics.compute_fidelity() == 0.0
    
    def test_compute_generalized_bound(self, dynamics):
        """Test generalized bound computation."""
        hbar_h = 1.0
        dynamics.model.state = np.array([1, 1, -1, -1])
        
        # Test with default parameters
        bound = dynamics.compute_generalized_bound(hbar_h)
        assert bound > 0
        
        # Test with custom parameters
        bound = dynamics.compute_generalized_bound(
            hbar_h=1.0,
            alpha=2.0,
            beta=0.5,
            t_c=2.0
        )
        assert bound > 0 
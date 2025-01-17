"""Tests for visualization utilities."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from historical_fidelity_simulator.utils.visualization import (
    plot_simulation_history,
    plot_phase_diagram,
    plot_state_evolution,
    plot_quantum_observables
)


@pytest.fixture
def sample_history():
    """Generate sample simulation history."""
    return [
        {'time': t, 'fidelity': np.sin(t), 'bound': 1.0 + 0.1*t, 'step': i}
        for i, t in enumerate(np.linspace(0, 10, 100))
    ]


@pytest.fixture
def sample_phase_data():
    """Generate sample phase transition data."""
    temps = np.linspace(1, 5, 50)
    fidelities = -np.tanh(2/temps)
    products = np.ones_like(temps) + 0.1*np.random.randn(50)
    bounds = np.ones_like(temps)
    return temps, fidelities, products, bounds


def test_plot_simulation_history(sample_history):
    """Test simulation history plotting."""
    fig, ax = plot_simulation_history(sample_history, "Test Simulation")
    
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) == 2  # Fidelity and bound lines
    assert ax.get_title() == "Test Simulation"
    
    plt.close(fig)


def test_plot_phase_diagram(sample_phase_data):
    """Test phase diagram plotting."""
    temps, fidelities, products, bounds = sample_phase_data
    fig, axes = plot_phase_diagram(
        temps, fidelities, products, bounds,
        param_name="Temperature"
    )
    
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 3  # Three subplots
    assert all(ax.get_xlabel() == "Temperature" for ax in axes)
    
    plt.close(fig)


def test_plot_state_evolution():
    """Test state evolution plotting."""
    times = np.linspace(0, 10, 100)
    states = [np.random.choice([-1, 1], size=10) for _ in range(100)]
    
    fig, ax = plot_state_evolution(states, times)
    
    assert isinstance(fig, plt.Figure)
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Site"
    
    plt.close(fig)


def test_plot_quantum_observables(sample_history):
    """Test quantum observables plotting."""
    observables = {
        "σx": np.sin(np.linspace(0, 10, 100)),
        "σz": np.cos(np.linspace(0, 10, 100))
    }
    
    fig, ax = plot_quantum_observables(sample_history, observables)
    
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) == 2  # Two observables
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Expectation Value"
    
    plt.close(fig)


def test_figure_sizes():
    """Test custom figure size handling."""
    history = [{'time': 0, 'fidelity': 0, 'bound': 0, 'step': 0}]
    figsize = (8, 5)
    
    fig, _ = plot_simulation_history(history, figsize=figsize)
    assert fig.get_size_inches().tolist() == list(figsize)
    
    plt.close(fig) 
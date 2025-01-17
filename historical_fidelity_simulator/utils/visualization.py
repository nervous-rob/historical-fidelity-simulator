"""Visualization utilities for simulation results."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_simulation_history(
    history: List[Dict],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple[Figure, Axes]:
    """Plot simulation history including fidelity and bounds.

    Args:
        history: List of simulation history dictionaries
        title: Optional plot title
        figsize: Figure size (width, height)

    Returns:
        Tuple of (Figure, Axes) for further customization
    """
    times = [h['time'] for h in history]
    fidelities = [h['fidelity'] for h in history]
    bounds = [h['bound'] for h in history]

    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot fidelity and bound
    ax.plot(times, fidelities, 'b-', label='Fidelity')
    ax.plot(times, bounds, 'r--', label='ℏ_h Bound')
    
    # Add uncertainty region
    ax.fill_between(times, bounds, alpha=0.2, color='red')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy / N')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax


def plot_phase_diagram(
    parameter_values: np.ndarray,
    fidelities: np.ndarray,
    uncertainty_products: np.ndarray,
    bounds: np.ndarray,
    param_name: str = 'Temperature',
    figsize: Tuple[float, float] = (12, 4)
) -> Tuple[Figure, Axes]:
    """Plot phase diagram showing fidelity, uncertainty product, and bounds.

    Args:
        parameter_values: Array of parameter values (e.g., temperature)
        fidelities: Array of fidelity values
        uncertainty_products: Array of ΔH_f Δt products
        bounds: Array of ℏ_h bounds
        param_name: Name of the parameter being varied
        figsize: Figure size (width, height)

    Returns:
        Tuple of (Figure, Axes) for further customization
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot fidelity
    ax1.plot(parameter_values, fidelities, 'b-')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Fidelity')
    ax1.grid(True)
    
    # Plot uncertainty product
    ax2.plot(parameter_values, uncertainty_products, 'g-')
    ax2.plot(parameter_values, bounds, 'r--', label='Bound')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('ΔH_f Δt')
    ax2.legend()
    ax2.grid(True)
    
    # Plot ratio of uncertainty product to bound
    ratios = uncertainty_products / bounds
    ax3.plot(parameter_values, ratios, 'k-')
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Ratio to Bound')
    ax3.grid(True)
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_state_evolution(
    states: List[np.ndarray],
    times: List[float],
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple[Figure, Axes]:
    """Plot evolution of classical spin states.

    Args:
        states: List of spin state arrays
        times: List of corresponding times
        figsize: Figure size (width, height)

    Returns:
        Tuple of (Figure, Axes) for further customization
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create 2D array of states
    state_matrix = np.array(states)
    
    # Plot heatmap
    im = ax.imshow(
        state_matrix.T,
        aspect='auto',
        extent=[times[0], times[-1], -0.5, state_matrix.shape[1]-0.5],
        cmap='RdBu'
    )
    
    plt.colorbar(im, ax=ax, label='Spin')
    ax.set_xlabel('Time')
    ax.set_ylabel('Site')
    ax.set_title('Spin State Evolution')
    
    return fig, ax


def plot_quantum_observables(
    history: List[Dict],
    observables: Dict[str, List[float]],
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple[Figure, Axes]:
    """Plot quantum observables over time.

    Args:
        history: Simulation history
        observables: Dictionary of observable names and their values
        figsize: Figure size (width, height)

    Returns:
        Tuple of (Figure, Axes) for further customization
    """
    times = [h['time'] for h in history]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, values in observables.items():
        ax.plot(times, values, label=name)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation Value')
    ax.legend()
    ax.grid(True)
    
    return fig, ax 
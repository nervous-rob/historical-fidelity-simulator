"""Getting Started with Historical Fidelity Simulator

This script demonstrates the basic usage of the Historical Fidelity Simulator package.
We'll cover:
1. Setting up a simple Ising system
2. Running classical simulations with GPU acceleration
3. Running quantum simulations with CuPy support
4. Comparing and visualizing results

For more advanced usage, see other examples in the examples/ directory.
"""

from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from historical_fidelity_simulator import (
    GeneralizedHistoricalSimulator,
    SimulationMode
)

def setup_output_directory() -> Path:
    """Set up output directory for saving results.
    
    Returns:
        Path: Path to output directory
    """
    # Try different relative paths to find the output directory
    output_dir = Path('output')
    if not output_dir.exists():
        # If running from examples directory, go up one level
        output_dir = Path('..') / 'output'
    if not output_dir.exists():
        # If running from a subdirectory of examples, go up two levels
        output_dir = Path('../..') / 'output'
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory for getting started results
    getting_started_dir = output_dir / 'getting_started'
    getting_started_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to: {getting_started_dir.absolute()}")
    return getting_started_dir

def run_classical_simulation(
    n_sites: int,
    params: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Run classical simulation with specified parameters.
    
    Args:
        n_sites: Number of lattice sites
        params: Dictionary containing simulation parameters
        
    Returns:
        List of simulation results at each timestep
    """
    classical_sim = GeneralizedHistoricalSimulator(
        n_sites=n_sites,
        coupling_strength=params['J'],
        field_strength=params['h'],
        temperature=params['T'],
        hbar_h=params['hbar_h'],
        mode=SimulationMode.CLASSICAL,
        use_gpu=False
    )
    
    print(f"Classical simulation using GPU: {classical_sim.use_gpu}")
    
    return classical_sim.run_simulation(
        n_steps=params['n_steps'],
        dt=params['dt'],
        measure_interval=params['measure_interval']
    )

def run_quantum_simulation(
    n_sites: int,
    params: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Run quantum simulation with specified parameters.
    
    Args:
        n_sites: Number of lattice sites
        params: Dictionary containing simulation parameters
        
    Returns:
        List of simulation results at each timestep
    """
    quantum_sim = GeneralizedHistoricalSimulator(
        n_sites=n_sites,
        coupling_strength=params['J'],
        field_strength=params['h'],
        temperature=params['T'],
        hbar_h=params['hbar_h'],
        mode=SimulationMode.QUANTUM,
        use_gpu=True
    )
    
    print(f"Quantum simulation using GPU: {quantum_sim.use_gpu}")
    
    return quantum_sim.run_simulation(
        n_steps=params['n_steps_quantum'],
        dt=params['dt'],
        measure_interval=params['measure_interval_quantum']
    )

def plot_simulation_results(
    classical_results: List[Dict[str, Any]],
    quantum_results: List[Dict[str, Any]],
    params: Dict[str, float],
    output_dir: Path
) -> None:
    """Plot and save simulation results.
    
    Args:
        classical_results: Results from classical simulation
        quantum_results: Results from quantum simulation
        params: Simulation parameters
        output_dir: Directory to save plots
    """
    # Extract time series
    classical_times = [step['time'] for step in classical_results]
    classical_fidelities = [step['fidelity'] for step in classical_results]
    quantum_times = [step['time'] for step in quantum_results]
    quantum_fidelities = [step['fidelity'] for step in quantum_results]
    
    # Compute moving average for classical results
    window = 20
    moving_avg = np.convolve(classical_fidelities, np.ones(window)/window, mode='valid')
    moving_avg_times = classical_times[window-1:]
    
    # Plot classical vs quantum comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    classical_bounds = [step['bound'] for step in classical_results]
    plt.plot(classical_times, classical_fidelities, label='Fidelity', alpha=0.3, linewidth=0.5)
    plt.plot(moving_avg_times, moving_avg, 'r-', linewidth=2.0, label='Avg Fidelity')
    plt.plot(classical_times, classical_bounds, '--', label='ℏ_h bound', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Classical: Fidelity vs Bound (N={params["n_sites"]}, T={params["T"]})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    quantum_bounds = [step['bound'] for step in quantum_results]
    plt.plot(quantum_times, quantum_fidelities, label='Fidelity', linewidth=2.0)
    plt.plot(quantum_times, quantum_bounds, '--', label='ℏ_h bound', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Quantum: Fidelity vs Bound (N={params["n_sites"]}, T={params["T"]})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fidelity_comparison.png')
    plt.show()

def main() -> None:
    """Run the getting started example."""
    # System parameters
    params = {
        'n_sites': 8,  # Small enough for quantum simulation (2^8 = 256 states)
        'J': 1.0,  # Coupling strength
        'h': 0.5,  # Field strength for interesting dynamics
        'T': 1.5,  # Temperature for good classical exploration
        'hbar_h': 0.1,  # Information Planck constant
        'dt': 0.05,  # Time step
        'n_steps': 5000,  # Classical simulation steps
        'n_steps_quantum': 300,  # Quantum simulation steps
        'measure_interval': 20,  # Classical measurement interval
        'measure_interval_quantum': 2  # Quantum measurement interval
    }
    
    # Set up output directory
    output_dir = setup_output_directory()
    
    # Run simulations
    classical_results = run_classical_simulation(params['n_sites'], params)
    quantum_results = run_quantum_simulation(params['n_sites'], params)
    
    # Plot and save results
    plot_simulation_results(classical_results, quantum_results, params, output_dir)
    
    # Print performance comparison
    print("\nPerformance Summary:")
    print(f"Classical simulation steps per second: "
          f"{params['n_steps'] / (classical_results[-1]['time'] - classical_results[0]['time']):.2f}")
    print(f"Quantum simulation steps per second: "
          f"{params['n_steps_quantum'] / (quantum_results[-1]['time'] - quantum_results[0]['time']):.2f}")

if __name__ == '__main__':
    main() 
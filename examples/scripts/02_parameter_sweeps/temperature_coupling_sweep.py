"""
Temperature and Coupling Sweep in Historical Fidelity Simulator

This script demonstrates how to perform parameter sweeps to explore:
1. Temperature dependence of fidelity and bounds
2. Coupling strength effects on system behavior
3. GPU-accelerated phase diagram generation
4. Classical vs quantum regime comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Any

from historical_fidelity_simulator.simulator import GeneralizedHistoricalSimulator, SimulationMode

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
    
    # Create a subdirectory for temperature coupling sweep results
    sweep_dir = output_dir / 'temperature_coupling_sweep'
    sweep_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to: {sweep_dir.absolute()}")
    return sweep_dir

def run_temperature_sweep_for_mode(
    T_range: np.ndarray,
    params: Dict[str, Any],
    mode: SimulationMode,
    n_steps: int,
    measure_interval: int,
    results_cache: Dict[Tuple[float, float], float] = None
) -> Tuple[List[float], List[float]]:
    """Run temperature sweep for given mode.
    
    Args:
        T_range: Array of temperatures to sweep
        params: Dictionary of simulation parameters
        mode: Simulation mode (classical or quantum)
        n_steps: Number of simulation steps
        measure_interval: Interval between measurements
        results_cache: Optional cache for storing results
        
    Returns:
        Tuple of (fidelities, bounds) lists
    """
    fidelities = []
    bounds = []
    
    for T in tqdm(T_range, desc=f"{mode.value} mode"):
        sim = GeneralizedHistoricalSimulator(
            n_sites=params['n_sites_classical' if mode == SimulationMode.CLASSICAL else 'n_sites_quantum'],
            coupling_strength=params['J'],
            field_strength=params['h'],
            temperature=T,
            hbar_h=params['hbar_h'],
            mode=mode,
            use_gpu=(mode == SimulationMode.QUANTUM)
        )
        
        results = sim.run_simulation(
            n_steps=n_steps,
            dt=params['dt'],
            measure_interval=measure_interval
        )
        
        fidelity = results[-1]['fidelity']
        fidelities.append(fidelity)
        bounds.append(results[-1]['bound'])
        
        # Cache the result if cache is available
        if results_cache is not None:
            results_cache[(T, params['J'])] = fidelity
    
    return fidelities, bounds

def run_coupling_sweep_for_mode(
    J_range: np.ndarray,
    params: Dict[str, Any],
    mode: SimulationMode,
    n_steps: int,
    measure_interval: int
) -> Tuple[List[float], List[float]]:
    """Run coupling strength sweep for given mode.
    
    Args:
        J_range: Array of coupling strengths to sweep
        params: Dictionary of simulation parameters
        mode: Simulation mode (classical or quantum)
        n_steps: Number of simulation steps
        measure_interval: Interval between measurements
        
    Returns:
        Tuple of (fidelities, bounds) lists
    """
    fidelities = []
    bounds = []
    
    for J in tqdm(J_range, desc=f"{mode.value} mode"):
        sim = GeneralizedHistoricalSimulator(
            n_sites=params['n_sites_classical' if mode == SimulationMode.CLASSICAL else 'n_sites_quantum'],
            coupling_strength=J,
            field_strength=params['h'],
            temperature=params['T'],
            hbar_h=params['hbar_h'],
            mode=mode,
            use_gpu=(mode == SimulationMode.QUANTUM)
        )
        
        results = sim.run_simulation(
            n_steps=n_steps,
            dt=params['dt'],
            measure_interval=measure_interval
        )
        
        fidelities.append(results[-1]['fidelity'])
        bounds.append(results[-1]['bound'])
    
    return fidelities, bounds

def compute_phase_diagram(
    params: Dict[str, Any],
    T_range: np.ndarray,
    J_range: np.ndarray,
    mode: SimulationMode
) -> np.ndarray:
    """Compute phase diagram for given mode.
    
    Args:
        params: Dictionary of simulation parameters
        T_range: Array of temperatures
        J_range: Array of coupling strengths
        mode: Simulation mode (CLASSICAL or QUANTUM)
    
    Returns:
        2D array of fidelity values
    """
    n_sites = params['n_sites_classical' if mode == SimulationMode.CLASSICAL else 'n_sites_quantum']
    n_steps = params['n_steps_classical' if mode == SimulationMode.CLASSICAL else 'n_steps_quantum']
    measure_interval = 50 if mode == SimulationMode.CLASSICAL else 5
    
    grid = np.zeros((len(J_range), len(T_range)))
    
    for i, J in enumerate(tqdm(J_range, desc=f"{mode.name.title()} mode")):
        for j, T in enumerate(T_range):
            sim = GeneralizedHistoricalSimulator(
                n_sites=n_sites,
                coupling_strength=J,
                field_strength=params['h'],
                temperature=T,
                hbar_h=params['hbar_h'],
                mode=mode,
                use_gpu=(mode == SimulationMode.QUANTUM)
            )
            
            results = sim.run_simulation(
                n_steps=n_steps,
                dt=params['dt'],
                measure_interval=measure_interval
            )
            
            grid[i, j] = results[-1]['fidelity']
    
    return grid

def plot_temperature_sweep_results(
    T_range: np.ndarray,
    classical_data: Tuple[List[float], List[float]],
    quantum_data: Tuple[List[float], List[float]],
    params: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot and save temperature sweep results.
    
    Args:
        T_range: Array of temperatures
        classical_data: Tuple of classical (fidelities, bounds)
        quantum_data: Tuple of quantum (fidelities, bounds)
        params: Simulation parameters
        output_dir: Directory to save plots
    """
    classical_fidelities, classical_bounds = classical_data
    quantum_fidelities, quantum_bounds = quantum_data
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(T_range, classical_fidelities, 'o-', label=f'Classical (N={params["n_sites_classical"]})')
    plt.plot(T_range, quantum_fidelities, 's-', label=f'Quantum (N={params["n_sites_quantum"]})')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Temperature')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(T_range, classical_bounds, 'o-', label=f'Classical (N={params["n_sites_classical"]})')
    plt.plot(T_range, quantum_bounds, 's-', label=f'Quantum (N={params["n_sites_quantum"]})')
    plt.xlabel('Temperature (T)')
    plt.ylabel('ℏ_h Bound')
    plt.title('Generalized Bound vs Temperature')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_sweep_results.png')
    plt.show()

def plot_coupling_sweep_results(
    J_range: np.ndarray,
    classical_data: Tuple[List[float], List[float]],
    quantum_data: Tuple[List[float], List[float]],
    params: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot and save coupling sweep results.
    
    Args:
        J_range: Array of coupling strengths
        classical_data: Tuple of classical (fidelities, bounds)
        quantum_data: Tuple of quantum (fidelities, bounds)
        params: Simulation parameters
        output_dir: Directory to save plots
    """
    classical_fidelities, classical_bounds = classical_data
    quantum_fidelities, quantum_bounds = quantum_data
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(J_range, classical_fidelities, 'o-', label=f'Classical (N={params["n_sites_classical"]})')
    plt.plot(J_range, quantum_fidelities, 's-', label=f'Quantum (N={params["n_sites_quantum"]})')
    plt.xlabel('Coupling Strength (J)')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Coupling')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(J_range, classical_bounds, 'o-', label=f'Classical (N={params["n_sites_classical"]})')
    plt.plot(J_range, quantum_bounds, 's-', label=f'Quantum (N={params["n_sites_quantum"]})')
    plt.xlabel('Coupling Strength (J)')
    plt.ylabel('ℏ_h Bound')
    plt.title('Generalized Bound vs Coupling')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'coupling_sweep_results.png')
    plt.show()

def plot_phase_diagrams(
    T_grid: np.ndarray,
    J_grid: np.ndarray,
    classical_grid: np.ndarray,
    quantum_grid: np.ndarray,
    params: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot and save phase diagrams.
    
    Args:
        T_grid: 2D array of temperatures
        J_grid: 2D array of coupling strengths
        classical_grid: Classical fidelity values
        quantum_grid: Quantum fidelity values
        params: Simulation parameters
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.pcolormesh(T_grid, J_grid, classical_grid, shading='auto')
    plt.colorbar(label='Fidelity')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Coupling Strength (J)')
    plt.title(f'Classical Phase Diagram (N={params["n_sites_classical"]})')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(T_grid, J_grid, quantum_grid, shading='auto')
    plt.colorbar(label='Fidelity')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Coupling Strength (J)')
    plt.title(f'Quantum Phase Diagram (N={params["n_sites_quantum"]})')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_diagram.png')
    plt.show()

def main() -> None:
    """Run the temperature and coupling sweep analysis."""
    # System parameters
    params = {
        'n_sites_classical': 32,  # Increased for better finite-size effects
        'n_sites_quantum': 8,     # Increased from 6 (2^8 = 256 states, still manageable)
        'J': 1.0,                 # Default coupling strength
        'h': 0.1,                 # External field strength
        'T': 2.0,                 # Default temperature
        'hbar_h': 0.1,            # Information Planck constant
        'n_steps_classical': 2000, # Increased for better equilibration
        'n_steps_quantum': 200,    # Increased for better evolution
        'dt': 0.05                # Decreased for better time resolution
    }
    
    # Setup sweep parameters
    T_range = np.linspace(0.1, 5.0, 15)  # Temperature sweep
    J_range = np.linspace(0.1, 3.0, 12)  # Coupling sweep
    T_range_2d = np.linspace(0.1, 5.0, 12)  # Temperature range for phase diagram
    J_range_2d = np.linspace(0.1, 3.0, 10)  # Coupling range for phase diagram
    T_grid, J_grid = np.meshgrid(T_range_2d, J_range_2d)
    
    # Set up output directory
    output_dir = setup_output_directory()
    
    # Initialize results cache for quantum mode (to avoid recomputing)
    classical_results_cache = {}
    quantum_results_cache = {}
    
    # Run temperature sweeps
    print("\nRunning temperature sweeps...")
    classical_temp_data = run_temperature_sweep_for_mode(
        T_range, params, SimulationMode.CLASSICAL,
        params['n_steps_classical'], 50, classical_results_cache
    )
    quantum_temp_data = run_temperature_sweep_for_mode(
        T_range, params, SimulationMode.QUANTUM,
        params['n_steps_quantum'], 5, quantum_results_cache
    )
    
    # Plot temperature sweep results
    plot_temperature_sweep_results(T_range, classical_temp_data, quantum_temp_data, params, output_dir)
    
    # Run coupling sweeps
    print("\nRunning coupling sweeps...")
    classical_coupling_data = run_coupling_sweep_for_mode(
        J_range, params, SimulationMode.CLASSICAL,
        params['n_steps_classical'], 50
    )
    quantum_coupling_data = run_coupling_sweep_for_mode(
        J_range, params, SimulationMode.QUANTUM,
        params['n_steps_quantum'], 5
    )
    
    # Plot coupling sweep results
    plot_coupling_sweep_results(J_range, classical_coupling_data, quantum_coupling_data, params, output_dir)
    
    # Generate phase diagrams
    print("\nGenerating phase diagrams...")
    classical_grid = compute_phase_diagram(params, T_range_2d, J_range_2d, SimulationMode.CLASSICAL)
    quantum_grid = compute_phase_diagram(params, T_range_2d, J_range_2d, SimulationMode.QUANTUM)
    
    # Plot phase diagrams
    plot_phase_diagrams(T_grid, J_grid, classical_grid, quantum_grid, params, output_dir)
    
    print("\nResults saved to:")
    print(f"- {(output_dir / 'temperature_sweep_results.png').absolute()}")
    print(f"- {(output_dir / 'coupling_sweep_results.png').absolute()}")
    print(f"- {(output_dir / 'phase_diagram.png').absolute()}")

if __name__ == '__main__':
    main() 
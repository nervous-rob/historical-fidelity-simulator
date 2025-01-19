"""Field Strength Sweep in Historical Fidelity Simulator

This script demonstrates how to perform field strength sweeps to explore:
1. System response to external fields
2. Field-induced phase transitions
3. Comparison of classical and quantum behaviors
4. GPU-accelerated parameter sweeps for faster exploration

For more details on the theory, see docs/research-proposal.md.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import numpy.typing as npt

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
    
    # Create a subdirectory for field strength sweep results
    sweep_dir = output_dir / 'field_strength_sweep'
    sweep_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to: {sweep_dir.absolute()}")
    return sweep_dir

def run_sweep_for_mode(
    h_range: npt.NDArray[np.float64],
    params: Dict[str, Any],
    mode: SimulationMode,
    n_steps: int,
    measure_interval: int
) -> Tuple[List[float], List[float]]:
    """Run field strength sweep for given mode.
    
    Args:
        h_range: Array of field strengths to sweep
        params: Dictionary of simulation parameters
        mode: Simulation mode (classical or quantum)
        n_steps: Number of simulation steps
        measure_interval: Interval between measurements
        
    Returns:
        Tuple of (fidelities, bounds) lists
    """
    fidelities = []
    bounds = []
    
    for h in tqdm(h_range, desc=f"{mode.value} mode"):
        sim = GeneralizedHistoricalSimulator(
            n_sites=params['n_sites_classical' if mode == SimulationMode.CLASSICAL else 'n_sites_quantum'],
            coupling_strength=params['J'],
            field_strength=h,
            temperature=params['T'],
            hbar_h=params['hbar_h'],
            mode=mode,
            use_gpu=(mode == SimulationMode.QUANTUM)  # GPU only for quantum mode
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
    T_range: npt.NDArray[np.float64],
    h_range: npt.NDArray[np.float64],
    mode: SimulationMode
) -> npt.NDArray[np.float64]:
    """Compute phase diagram for given mode.
    
    Args:
        params: Dictionary of simulation parameters
        T_range: Array of temperatures
        h_range: Array of field strengths
        mode: Simulation mode (CLASSICAL or QUANTUM)
    
    Returns:
        2D array of fidelity values
    """
    n_sites = params['n_sites_classical' if mode == SimulationMode.CLASSICAL else 'n_sites_quantum']
    n_steps = params['n_steps_classical' if mode == SimulationMode.CLASSICAL else 'n_steps_quantum']
    measure_interval = 50 if mode == SimulationMode.CLASSICAL else 5
    
    grid = np.zeros((len(h_range), len(T_range)))
    
    for i, h in enumerate(tqdm(h_range, desc=f"{mode.name.title()} mode")):
        for j, T in enumerate(T_range):
            sim = GeneralizedHistoricalSimulator(
                n_sites=n_sites,
                coupling_strength=params['J'],
                field_strength=h,
                temperature=T,
                hbar_h=params['hbar_h'],
                mode=mode,
                use_gpu=(mode == SimulationMode.QUANTUM)  # GPU only for quantum mode
            )
            
            results = sim.run_simulation(
                n_steps=n_steps,
                dt=params['dt'],
                measure_interval=measure_interval
            )
            
            grid[i, j] = results[-1]['fidelity']
    
    return grid

def plot_sweep_results(
    h_range: npt.NDArray[np.float64],
    classical_data: Tuple[List[float], List[float]],
    quantum_data: Tuple[List[float], List[float]],
    params: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot and save field sweep results.
    
    Args:
        h_range: Array of field strengths
        classical_data: Tuple of classical (fidelities, bounds)
        quantum_data: Tuple of quantum (fidelities, bounds)
        params: Simulation parameters
        output_dir: Directory to save plots
    """
    classical_fidelities, classical_bounds = classical_data
    quantum_fidelities, quantum_bounds = quantum_data
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(h_range, classical_fidelities, 'o-', label=f'Classical (N={params["n_sites_classical"]})')
    plt.plot(h_range, quantum_fidelities, 's-', label=f'Quantum (N={params["n_sites_quantum"]})')
    plt.xlabel('Field Strength (h)')
    plt.ylabel('Fidelity')
    plt.title('Fidelity vs Field Strength')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(h_range, classical_bounds, 'o-', label=f'Classical (N={params["n_sites_classical"]})')
    plt.plot(h_range, quantum_bounds, 's-', label=f'Quantum (N={params["n_sites_quantum"]})')
    plt.xlabel('Field Strength (h)')
    plt.ylabel('ℏ_h Bound')
    plt.title('Generalized Bound vs Field Strength')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'field_sweep_results.png')
    plt.show()

def plot_phase_diagrams(
    T_grid: npt.NDArray[np.float64],
    h_grid: npt.NDArray[np.float64],
    classical_grid: npt.NDArray[np.float64],
    quantum_grid: npt.NDArray[np.float64],
    params: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot and save phase diagrams.
    
    Args:
        T_grid: 2D array of temperatures
        h_grid: 2D array of field strengths
        classical_grid: Classical fidelity values
        quantum_grid: Quantum fidelity values
        params: Simulation parameters
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.pcolormesh(T_grid, h_grid, classical_grid, shading='auto')
    plt.colorbar(label='Fidelity')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Field Strength (h)')
    plt.title(f'Classical Phase Diagram (N={params["n_sites_classical"]})')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(T_grid, h_grid, quantum_grid, shading='auto')
    plt.colorbar(label='Fidelity')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Field Strength (h)')
    plt.title(f'Quantum Phase Diagram (N={params["n_sites_quantum"]})')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'field_phase_diagram.png')
    plt.show()

def main() -> None:
    """Run the field strength sweep analysis."""
    # System parameters
    params = {
        'n_sites_classical': 32,  # Increased for better finite-size effects
        'n_sites_quantum': 8,     # 2^8 = 256 states, still manageable
        'J': 1.0,                 # Coupling strength
        'T': 2.0,                 # Temperature
        'hbar_h': 0.1,            # Information Planck constant
        'n_steps_classical': 2000, # Increased for better equilibration
        'n_steps_quantum': 200,    # Increased for better evolution
        'dt': 0.05                # Decreased for better time resolution
    }
    
    # Setup sweep parameters
    h_range = np.linspace(0.0, 3.0, 20)  # Field strength sweep
    T_range = np.linspace(0.1, 5.0, 15)  # Temperature range for phase diagram
    h_range_2d = np.linspace(0.0, 3.0, 15)  # Field range for phase diagram
    T_grid, h_grid = np.meshgrid(T_range, h_range_2d)
    
    # Set up output directory
    output_dir = setup_output_directory()
    
    # Run field strength sweeps
    print("\nRunning field strength sweeps...")
    classical_data = run_sweep_for_mode(h_range, params, SimulationMode.CLASSICAL, params['n_steps_classical'], 50)
    quantum_data = run_sweep_for_mode(h_range, params, SimulationMode.QUANTUM, params['n_steps_quantum'], 5)
    
    # Plot sweep results
    plot_sweep_results(h_range, classical_data, quantum_data, params, output_dir)
    
    # Generate phase diagrams
    print("\nGenerating phase diagrams...")
    classical_grid = compute_phase_diagram(params, T_range, h_range_2d, SimulationMode.CLASSICAL)
    quantum_grid = compute_phase_diagram(params, T_range, h_range_2d, SimulationMode.QUANTUM)
    
    # Plot phase diagrams
    plot_phase_diagrams(T_grid, h_grid, classical_grid, quantum_grid, params, output_dir)
    
    print("\nResults saved to:")
    print(f"- {(output_dir / 'field_sweep_results.png').absolute()}")
    print(f"- {(output_dir / 'field_phase_diagram.png').absolute()}")

if __name__ == '__main__':
    main() 
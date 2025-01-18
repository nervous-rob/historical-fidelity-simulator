"""
Information Planck Constant Sweep in Historical Fidelity Simulator

This script explores the role of the information Planck constant (ℏ_h):
1. Impact on fidelity and uncertainty bounds
2. Scaling behavior in classical and quantum regimes
3. GPU-accelerated parameter sweeps for efficient exploration
4. Comparison of classical and quantum responses
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from historical_fidelity_simulator.simulator import GeneralizedHistoricalSimulator, SimulationMode

# Get the benchmarks directory relative to the current file
benchmarks_dir = Path('benchmarks')
if not benchmarks_dir.exists():
    # If running from examples directory, go up one level
    benchmarks_dir = Path('..') / 'benchmarks'
if not benchmarks_dir.exists():
    # If running from a subdirectory of examples, go up two levels
    benchmarks_dir = Path('../..') / 'benchmarks'

print(f"Saving results to: {benchmarks_dir.absolute()}")

# System parameters
n_sites_classical = 32  # Larger system for classical
n_sites_quantum = 8    # Smaller system for quantum (2^8 states)
J = 1.0  # Coupling strength
h = 0.1  # Field strength
T = 2.0  # Temperature
n_steps_classical = 2000  # More steps for better statistics
n_steps_quantum = 200    # Fewer steps for quantum
dt = 0.05  # Time step

# Setup ℏ_h sweep parameters
hbar_h_range = np.logspace(-3, 1, 20)  # Log scale from 0.001 to 10

# Arrays to store results
classical_fidelities = []
quantum_fidelities = []
classical_bounds = []
quantum_bounds = []

# Perform ℏ_h sweep
print("Performing information Planck constant sweep...")
for hbar_h in tqdm(hbar_h_range):
    # Classical simulation (GPU-accelerated by default)
    classical_sim = GeneralizedHistoricalSimulator(
        n_sites=n_sites_classical,
        coupling_strength=J,
        field_strength=h,
        temperature=T,
        hbar_h=hbar_h,
        mode=SimulationMode.CLASSICAL
    )
    
    classical_results = classical_sim.run_simulation(
        n_steps=n_steps_classical,
        dt=dt,
        measure_interval=50
    )
    
    # Quantum simulation (GPU-accelerated by default)
    quantum_sim = GeneralizedHistoricalSimulator(
        n_sites=n_sites_quantum,
        coupling_strength=J,
        field_strength=h,
        temperature=T,
        hbar_h=hbar_h,
        mode=SimulationMode.QUANTUM
    )
    
    quantum_results = quantum_sim.run_simulation(
        n_steps=n_steps_quantum,
        dt=dt,
        measure_interval=5
    )
    
    # Store final values
    classical_fidelities.append(classical_results[-1]['fidelity'])
    quantum_fidelities.append(quantum_results[-1]['fidelity'])
    classical_bounds.append(classical_results[-1]['bound'])
    quantum_bounds.append(quantum_results[-1]['bound'])

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogx(hbar_h_range, classical_fidelities, 'o-', label=f'Classical (N={n_sites_classical})')
plt.semilogx(hbar_h_range, quantum_fidelities, 's-', label=f'Quantum (N={n_sites_quantum})')
plt.xlabel('Information Planck Constant (ℏ_h)')
plt.ylabel('Fidelity')
plt.title('Fidelity vs ℏ_h')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.loglog(hbar_h_range, classical_bounds, 'o-', label=f'Classical (N={n_sites_classical})')
plt.loglog(hbar_h_range, quantum_bounds, 's-', label=f'Quantum (N={n_sites_quantum})')
plt.xlabel('Information Planck Constant (ℏ_h)')
plt.ylabel('ℏ_h Bound')
plt.title('Generalized Bound vs ℏ_h')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(benchmarks_dir / 'information_planck_sweep.png')
plt.show()

# Generate phase diagram
print("\nGenerating phase diagram...")
T_range = np.linspace(0.1, 5.0, 15)  # Temperature range
hbar_h_range_2d = np.logspace(-3, 1, 15)  # Log scale for phase diagram
T_grid, hbar_h_grid = np.meshgrid(T_range, hbar_h_range_2d)
classical_grid = np.zeros_like(T_grid)
quantum_grid = np.zeros_like(T_grid)

for i, hbar_h in enumerate(tqdm(hbar_h_range_2d)):
    for j, T in enumerate(T_range):
        # Classical simulation (GPU-accelerated by default)
        classical_sim = GeneralizedHistoricalSimulator(
            n_sites=n_sites_classical,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.CLASSICAL
        )
        
        classical_results = classical_sim.run_simulation(
            n_steps=n_steps_classical,
            dt=dt,
            measure_interval=50
        )
        
        # Quantum simulation (GPU-accelerated by default)
        quantum_sim = GeneralizedHistoricalSimulator(
            n_sites=n_sites_quantum,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.QUANTUM
        )
        
        quantum_results = quantum_sim.run_simulation(
            n_steps=n_steps_quantum,
            dt=dt,
            measure_interval=5
        )
        
        classical_grid[i, j] = classical_results[-1]['fidelity']
        quantum_grid[i, j] = quantum_results[-1]['fidelity']

# Plot phase diagrams
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.pcolormesh(T_grid, hbar_h_grid, classical_grid, shading='auto')
plt.colorbar(label='Fidelity')
plt.xlabel('Temperature (T)')
plt.ylabel('Information Planck Constant (ℏ_h)')
plt.title(f'Classical Phase Diagram (N={n_sites_classical})')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.pcolormesh(T_grid, hbar_h_grid, quantum_grid, shading='auto')
plt.colorbar(label='Fidelity')
plt.xlabel('Temperature (T)')
plt.ylabel('Information Planck Constant (ℏ_h)')
plt.title(f'Quantum Phase Diagram (N={n_sites_quantum})')
plt.yscale('log')

plt.tight_layout()
plt.savefig(benchmarks_dir / 'information_planck_phase_diagram.png')
plt.show()

print("\nResults saved to:")
print(f"- {(benchmarks_dir / 'information_planck_sweep.png').absolute()}")
print(f"- {(benchmarks_dir / 'information_planck_phase_diagram.png').absolute()}") 
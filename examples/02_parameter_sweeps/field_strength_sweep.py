"""
Field Strength Sweep in Historical Fidelity Simulator

This script demonstrates how to perform field strength sweeps to explore:
1. System response to external fields
2. Field-induced phase transitions
3. Comparison of classical and quantum behaviors
4. GPU-accelerated parameter sweeps for faster exploration
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

# Common parameters
n_sites_classical = 32  # Increased for better finite-size effects
n_sites_quantum = 8    # Increased from 6 (2^8 = 256 states, still manageable)
J = 1.0  # Coupling strength
T = 2.0  # Temperature
hbar_h = 0.1  # Information Planck constant
n_steps_classical = 2000  # Increased for better equilibration
n_steps_quantum = 200   # Increased for better evolution
dt = 0.05  # Decreased for better time resolution

# Setup field strength sweep parameters
h_range = np.linspace(0.0, 3.0, 20)  # Increased points for better resolution

# Arrays to store results
classical_fidelities = []
quantum_fidelities = []
classical_bounds = []
quantum_bounds = []

# Perform field strength sweep
print("Performing field strength sweep...")
for h in tqdm(h_range):
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
        measure_interval=50  # More frequent measurements
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
        measure_interval=5  # More frequent measurements
    )
    
    # Store final values
    classical_fidelities.append(classical_results[-1]['fidelity'])
    quantum_fidelities.append(quantum_results[-1]['fidelity'])
    classical_bounds.append(classical_results[-1]['bound'])
    quantum_bounds.append(quantum_results[-1]['bound'])

# Plot field strength sweep results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(h_range, classical_fidelities, 'o-', label=f'Classical (N={n_sites_classical})')
plt.plot(h_range, quantum_fidelities, 's-', label=f'Quantum (N={n_sites_quantum})')
plt.xlabel('Field Strength (h)')
plt.ylabel('Fidelity')
plt.title('Fidelity vs Field Strength')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(h_range, classical_bounds, 'o-', label=f'Classical (N={n_sites_classical})')
plt.plot(h_range, quantum_bounds, 's-', label=f'Quantum (N={n_sites_quantum})')
plt.xlabel('Field Strength (h)')
plt.ylabel('ℏ_h Bound')
plt.title('Generalized Bound vs Field Strength')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(benchmarks_dir / 'field_sweep_results.png')
plt.show()

# Generate phase diagram
print("\nGenerating phase diagram...")
T_range = np.linspace(0.1, 5.0, 15)  # Temperature range
h_range = np.linspace(0.0, 3.0, 15)  # Field strength range
T_grid, h_grid = np.meshgrid(T_range, h_range)
classical_grid = np.zeros_like(T_grid)
quantum_grid = np.zeros_like(T_grid)

for i, h in enumerate(tqdm(h_range)):
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
plt.pcolormesh(T_grid, h_grid, classical_grid, shading='auto')
plt.colorbar(label='Fidelity')
plt.xlabel('Temperature (T)')
plt.ylabel('Field Strength (h)')
plt.title(f'Classical Phase Diagram (N={n_sites_classical})')

plt.subplot(1, 2, 2)
plt.pcolormesh(T_grid, h_grid, quantum_grid, shading='auto')
plt.colorbar(label='Fidelity')
plt.xlabel('Temperature (T)')
plt.ylabel('Field Strength (h)')
plt.title(f'Quantum Phase Diagram (N={n_sites_quantum})')

plt.tight_layout()
plt.savefig(benchmarks_dir / 'field_phase_diagram.png')
plt.show()

print("\nResults saved to:")
print(f"- {(benchmarks_dir / 'field_sweep_results.png').absolute()}")
print(f"- {(benchmarks_dir / 'field_phase_diagram.png').absolute()}") 
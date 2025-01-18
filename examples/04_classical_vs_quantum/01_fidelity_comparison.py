"""
Classical vs Quantum Comparison in Historical Fidelity Simulator

This script provides detailed comparisons between classical and quantum behaviors:
1. Time evolution characteristics
2. Response to temperature and coupling
3. Uncertainty relations and bounds
4. GPU-accelerated analysis for faster exploration
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
n_sites = 8  # Small enough for quantum simulation
h = 0.1  # External field strength
hbar_h = 0.1  # Information Planck constant
dt = 0.05  # Time step

# Compare evolution at different temperatures
temperatures = [0.5, 1.0, 2.0]
J = 1.0  # Fixed coupling strength

# Time evolution comparison
n_steps_classical = 2000
n_steps_quantum = 200

plt.figure(figsize=(15, 5))
for i, T in enumerate(temperatures):
    plt.subplot(1, 3, i+1)
    
    # Classical simulation (GPU-accelerated by default)
    classical_sim = GeneralizedHistoricalSimulator(
        n_sites=n_sites,
        coupling_strength=J,
        field_strength=h,
        temperature=T,
        hbar_h=hbar_h,
        mode=SimulationMode.CLASSICAL
    )
    
    classical_results = classical_sim.run_simulation(
        n_steps=n_steps_classical,
        dt=dt,
        measure_interval=10
    )
    
    # Quantum simulation (GPU-accelerated by default)
    quantum_sim = GeneralizedHistoricalSimulator(
        n_sites=n_sites,
        coupling_strength=J,
        field_strength=h,
        temperature=T,
        hbar_h=hbar_h,
        mode=SimulationMode.QUANTUM
    )
    
    quantum_results = quantum_sim.run_simulation(
        n_steps=n_steps_quantum,
        dt=dt,
        measure_interval=2
    )
    
    # Plot results
    times_classical = [step['time'] for step in classical_results]
    fidelities_classical = [step['fidelity'] for step in classical_results]
    bounds_classical = [step['bound'] for step in classical_results]
    
    times_quantum = [step['time'] for step in quantum_results]
    fidelities_quantum = [step['fidelity'] for step in quantum_results]
    bounds_quantum = [step['bound'] for step in quantum_results]
    
    # Add moving average for classical
    window = 20
    moving_avg = np.convolve(fidelities_classical, np.ones(window)/window, mode='valid')
    moving_avg_times = times_classical[window-1:]
    
    plt.plot(times_classical, fidelities_classical, 'b-', alpha=0.2, linewidth=0.5, label='Classical (Raw)')
    plt.plot(moving_avg_times, moving_avg, 'b-', linewidth=2.0, label='Classical (Avg)')
    plt.plot(times_quantum, fidelities_quantum, 'r-', linewidth=2.0, label='Quantum')
    plt.plot(times_classical, bounds_classical, 'b--', alpha=0.5, label='Classical Bound')
    plt.plot(times_quantum, bounds_quantum, 'r--', alpha=0.5, label='Quantum Bound')
    
    plt.xlabel('Time')
    plt.ylabel('Fidelity / Bound')
    plt.title(f'Evolution at T={T}')
    if i == 0:
        plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(benchmarks_dir / 'evolution_comparison.png')
plt.show()

# Compare uncertainty products
print("\nAnalyzing uncertainty relations...")
coupling_range = np.linspace(0.1, 3.0, 10)
temperature_range = np.linspace(0.5, 5.0, 10)

uncertainty_products = {
    'classical': np.zeros((len(temperature_range), len(coupling_range))),
    'quantum': np.zeros((len(temperature_range), len(coupling_range)))
}

for i, T in enumerate(tqdm(temperature_range)):
    for j, J in enumerate(coupling_range):
        # Classical simulation (GPU-accelerated by default)
        classical_sim = GeneralizedHistoricalSimulator(
            n_sites=n_sites,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.CLASSICAL
        )
        
        classical_results = classical_sim.run_simulation(
            n_steps=1000,
            dt=dt,
            measure_interval=50
        )
        
        # Quantum simulation (GPU-accelerated by default)
        quantum_sim = GeneralizedHistoricalSimulator(
            n_sites=n_sites,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.QUANTUM
        )
        
        quantum_results = quantum_sim.run_simulation(
            n_steps=100,
            dt=dt,
            measure_interval=5
        )
        
        # Store final uncertainty products
        uncertainty_products['classical'][i, j] = classical_results[-1]['bound']
        uncertainty_products['quantum'][i, j] = quantum_results[-1]['bound']

# Plot uncertainty product phase diagrams
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.pcolormesh(coupling_range, temperature_range, uncertainty_products['classical'],
               shading='auto')
plt.colorbar(label='ℏ_h Bound')
plt.xlabel('Coupling Strength (J)')
plt.ylabel('Temperature (T)')
plt.title('Classical Uncertainty Products')

plt.subplot(1, 2, 2)
plt.pcolormesh(coupling_range, temperature_range, uncertainty_products['quantum'],
               shading='auto')
plt.colorbar(label='ℏ_h Bound')
plt.xlabel('Coupling Strength (J)')
plt.ylabel('Temperature (T)')
plt.title('Quantum Uncertainty Products')

plt.tight_layout()
plt.savefig(benchmarks_dir / 'uncertainty_products.png')
plt.show()

# Analyze scaling of differences
differences = np.abs(uncertainty_products['quantum'] - uncertainty_products['classical'])
plt.figure(figsize=(10, 6))
plt.pcolormesh(coupling_range, temperature_range, differences, shading='auto')
plt.colorbar(label='|Quantum - Classical|')
plt.xlabel('Coupling Strength (J)')
plt.ylabel('Temperature (T)')
plt.title('Quantum-Classical Difference in Uncertainty Products')
plt.tight_layout()
plt.savefig(benchmarks_dir / 'quantum_classical_differences.png')
plt.show()

# Print key observations
print("\nKey Observations:")
print(f"1. Maximum quantum-classical difference: {np.max(differences):.3f}")
print(f"2. Average quantum-classical difference: {np.mean(differences):.3f}")
print(f"3. Temperature with largest differences: T ≈ {temperature_range[np.argmax(np.mean(differences, axis=1))]:.2f}")
print(f"4. Coupling with largest differences: J ≈ {coupling_range[np.argmax(np.mean(differences, axis=0))]:.2f}")

print("\nResults saved to:")
print(f"- {(benchmarks_dir / 'evolution_comparison.png').absolute()}")
print(f"- {(benchmarks_dir / 'uncertainty_products.png').absolute()}")
print(f"- {(benchmarks_dir / 'quantum_classical_differences.png').absolute()}") 
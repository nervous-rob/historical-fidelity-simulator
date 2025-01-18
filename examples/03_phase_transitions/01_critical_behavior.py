"""
Critical Behavior Analysis in Historical Fidelity Simulator

This script analyzes phase transitions and critical behavior, focusing on:
1. Order parameter scaling near critical points
2. Finite-size scaling analysis with GPU acceleration
3. Critical exponents estimation
4. Correlation function behavior
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
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
h = 0.1  # External field strength
hbar_h = 0.1  # Information Planck constant
n_steps = 5000  # Steps for equilibration
dt = 0.05  # Time step

# Temperature range focused around critical point
T_c = 2.27  # Theoretical critical temperature for 2D Ising
T_range = np.linspace(1.5, 3.0, 30)  # Dense sampling around T_c
reduced_t = (T_range - T_c) / T_c  # Reduced temperature

# System sizes for finite-size scaling
sizes = [8, 16, 24, 32]  # Larger sizes for classical
quantum_sizes = [4, 6, 8]  # Limited sizes for quantum
J = 1.0  # Fixed coupling strength

# Arrays to store results
classical_fidelities = {N: [] for N in sizes}
quantum_fidelities = {N: [] for N in quantum_sizes}
classical_bounds = {N: [] for N in sizes}
quantum_bounds = {N: [] for N in quantum_sizes}

# Perform temperature sweep for different system sizes
print("Analyzing finite-size scaling...")
for N in sizes:
    print(f"\nSystem size N = {N}")
    for T in tqdm(T_range):
        # Classical simulation (GPU-accelerated by default)
        classical_sim = GeneralizedHistoricalSimulator(
            n_sites=N,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.CLASSICAL
        )
        
        classical_results = classical_sim.run_simulation(
            n_steps=n_steps,
            dt=dt,
            measure_interval=50
        )
        
        classical_fidelities[N].append(classical_results[-1]['fidelity'])
        classical_bounds[N].append(classical_results[-1]['bound'])

# Quantum simulations for smaller sizes
for N in quantum_sizes:
    print(f"\nQuantum system size N = {N}")
    for T in tqdm(T_range):
        # Quantum simulation (GPU-accelerated by default)
        quantum_sim = GeneralizedHistoricalSimulator(
            n_sites=N,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.QUANTUM
        )
        
        quantum_results = quantum_sim.run_simulation(
            n_steps=1000,  # Fewer steps for quantum
            dt=dt,
            measure_interval=5
        )
        
        quantum_fidelities[N].append(quantum_results[-1]['fidelity'])
        quantum_bounds[N].append(quantum_results[-1]['bound'])

# Plot finite-size scaling for classical systems
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for N in sizes:
    plt.plot(reduced_t, np.abs(classical_fidelities[N]), 'o-', label=f'N={N}')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Reduced Temperature (t)')
plt.ylabel('|Fidelity|')
plt.title('Classical Finite-Size Scaling')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for N in sizes:
    plt.plot(reduced_t, classical_bounds[N], 'o-', label=f'N={N}')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Reduced Temperature (t)')
plt.ylabel('ℏ_h Bound')
plt.title('Classical Bound Scaling')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(benchmarks_dir / 'classical_scaling.png')
plt.show()

# Fit critical exponents
def power_law(x, beta, a):
    return a * np.abs(x)**beta

# Analyze critical exponents for largest system
N_max = max(sizes)
t_fit = reduced_t[reduced_t > 0]  # Use T > T_c
m_fit = np.abs(classical_fidelities[N_max])[reduced_t > 0]

# Fit power law to get beta exponent
popt, _ = curve_fit(power_law, t_fit, m_fit)
beta_fit, a_fit = popt

plt.figure(figsize=(10, 6))
plt.plot(t_fit, m_fit, 'o', label='Data')
plt.plot(t_fit, power_law(t_fit, beta_fit, a_fit), 'r-',
         label=f'Fit: β ≈ {beta_fit:.3f}')
plt.xlabel('Reduced Temperature (t)')
plt.ylabel('|Fidelity|')
plt.title(f'Critical Exponent Analysis (N={N_max})')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.savefig(benchmarks_dir / 'critical_exponent.png')
plt.show()

# Compare quantum and classical behavior near T_c
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for N in quantum_sizes:
    plt.plot(reduced_t, np.abs(quantum_fidelities[N]), 'o-', label=f'Quantum N={N}')
plt.plot(reduced_t, np.abs(classical_fidelities[8]), 's-', label='Classical N=8')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Reduced Temperature (t)')
plt.ylabel('|Fidelity|')
plt.title('Quantum vs Classical Critical Behavior')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for N in quantum_sizes:
    plt.plot(reduced_t, quantum_bounds[N], 'o-', label=f'Quantum N={N}')
plt.plot(reduced_t, classical_bounds[8], 's-', label='Classical N=8')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Reduced Temperature (t)')
plt.ylabel('ℏ_h Bound')
plt.title('Quantum vs Classical Bound Scaling')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(benchmarks_dir / 'quantum_classical_comparison.png')
plt.show()

print(f"\nEstimated critical exponent β = {beta_fit:.3f}")
print("(Compare to 2D Ising β = 0.125)")

print("\nResults saved to:")
print(f"- {(benchmarks_dir / 'classical_scaling.png').absolute()}")
print(f"- {(benchmarks_dir / 'critical_exponent.png').absolute()}")
print(f"- {(benchmarks_dir / 'quantum_classical_comparison.png').absolute()}") 
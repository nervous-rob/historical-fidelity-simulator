"""
Getting Started with Historical Fidelity Simulator

This script demonstrates the basic usage of the Historical Fidelity Simulator package.
We'll cover:
1. Setting up a simple Ising system
2. Running classical simulations with GPU acceleration
3. Running quantum simulations with CuPy support
4. Comparing and visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt

from historical_fidelity_simulator.simulator import GeneralizedHistoricalSimulator, SimulationMode

# System parameters
n_sites = 8  # Small enough for quantum simulation (2^8 = 256 states)
J = 1.0  # Coupling strength
h = 0.5  # Increased field strength for more interesting dynamics
T = 1.5  # Increased temperature for better classical exploration
hbar_h = 0.1  # Information Planck constant

# Create classical simulator instance (GPU acceleration enabled by default)
classical_sim = GeneralizedHistoricalSimulator(
    n_sites=n_sites,
    coupling_strength=J,
    field_strength=h,
    temperature=T,
    hbar_h=hbar_h,
    mode=SimulationMode.CLASSICAL
)

print(f"Classical simulation using GPU: {classical_sim.use_gpu}")

# Run classical simulation
n_steps = 5000  # Increased for better sampling
dt = 0.05  # Decreased for better time resolution
classical_results = classical_sim.run_simulation(
    n_steps=n_steps,
    dt=dt,
    measure_interval=20  # Adjusted for smoother averaging
)

# Plot classical evolution
plt.figure(figsize=(10, 6))
times = [step['time'] for step in classical_results]
fidelities = [step['fidelity'] for step in classical_results]

# Add moving average to see trends better
window = 20
moving_avg = np.convolve(fidelities, np.ones(window)/window, mode='valid')
moving_avg_times = times[window-1:]

plt.plot(times, fidelities, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
plt.plot(moving_avg_times, moving_avg, 'r-', linewidth=2.0, label='Moving Average')
plt.xlabel('Time')
plt.ylabel('Fidelity')
plt.title(f'GPU-Accelerated Classical Simulation: Fidelity Evolution (N={n_sites}, T={T})')
plt.legend()
plt.grid(True)
plt.show()

# Create quantum simulator instance (GPU support enabled by default)
quantum_sim = GeneralizedHistoricalSimulator(
    n_sites=n_sites,
    coupling_strength=J,
    field_strength=h,
    temperature=T,  # Using same temperature for comparison
    hbar_h=hbar_h,
    mode=SimulationMode.QUANTUM
)

print(f"Quantum simulation using GPU: {quantum_sim.use_gpu}")

# Run quantum simulation (fewer steps for quantum)
n_steps_quantum = 300  # Increased to see full decoherence
quantum_results = quantum_sim.run_simulation(
    n_steps=n_steps_quantum,
    dt=dt,
    measure_interval=2  # More frequent measurements for smoother curve
)

# Plot quantum evolution
plt.figure(figsize=(10, 6))
times = [step['time'] for step in quantum_results]
fidelities = [step['fidelity'] for step in quantum_results]
plt.plot(times, fidelities, linewidth=2.0)
plt.xlabel('Time')
plt.ylabel('Fidelity')
plt.title(f'GPU-Accelerated Quantum Simulation: Fidelity Evolution (N={n_sites}, T={T})')
plt.grid(True)
plt.show()

# Compare generalized bounds
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
classical_bounds = [step['bound'] for step in classical_results]
plt.plot(times, fidelities, label='Fidelity', alpha=0.3, linewidth=0.5)
plt.plot(moving_avg_times, moving_avg, 'r-', linewidth=2.0, label='Avg Fidelity')
plt.plot(times, classical_bounds, '--', label='ℏ_h bound', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'Classical: Fidelity vs Bound (N={n_sites}, T={T})')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
quantum_bounds = [step['bound'] for step in quantum_results]
plt.plot(times, fidelities, label='Fidelity', linewidth=2.0)
plt.plot(times, quantum_bounds, '--', label='ℏ_h bound', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'Quantum: Fidelity vs Bound (N={n_sites}, T={T})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print performance comparison
print("\nPerformance Summary:")
print(f"Classical simulation steps per second: {n_steps / (times[-1] - times[0]):.2f}")
print(f"Quantum simulation steps per second: {n_steps_quantum / (times[-1] - times[0]):.2f}") 
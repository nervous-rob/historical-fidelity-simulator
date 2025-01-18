"""
GPU vs CPU Performance Comparison

This script demonstrates the performance benefits of GPU acceleration by:
1. Running identical simulations on both CPU and GPU
2. Comparing execution times for different system sizes
3. Measuring speedup factors for both classical and quantum modes
4. Analyzing memory usage and scaling behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import psutil
import os
import json
from datetime import datetime

from historical_fidelity_simulator.simulator import GeneralizedHistoricalSimulator, SimulationMode

# Test parameters
classical_sizes = [8, 16, 32, 64, 128, 256]  # Linear scaling with N
quantum_sizes = [2, 4, 6, 8, 10]  # Exponential scaling with N (2^N states)
cpu_limit = 8  # Maximum size for CPU tests
n_steps_classical = 1000  # Fixed number of steps for classical
n_steps_quantum = 100    # Fewer steps for quantum due to complexity
dt = 0.05
J = 1.0
h = 0.1
T = 2.0
hbar_h = 0.1

# Arrays to store timing results
classical_cpu_times = []
classical_gpu_times = []
quantum_cpu_times = []
quantum_gpu_times = []
memory_usage_cpu = []
memory_usage_gpu = []
quantum_memory_cpu = []
quantum_memory_gpu = []

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("Running performance comparison...")

# Classical Mode Tests
print("\nClassical Mode Tests:")

# GPU tests for all classical sizes
print("\nRunning GPU tests for classical sizes:")
for n_sites in tqdm(classical_sizes, desc="GPU Classical"):
    print(f"\nClassical Mode GPU Test (N={n_sites}):")
    sim_gpu = GeneralizedHistoricalSimulator(
        n_sites=n_sites,
        coupling_strength=J,
        field_strength=h,
        temperature=T,
        hbar_h=hbar_h,
        mode=SimulationMode.CLASSICAL,
        use_gpu=True
    )
    
    start_mem = get_memory_usage()
    start_time = time.time()
    sim_gpu.run_simulation(n_steps=n_steps_classical, dt=dt)
    gpu_time = time.time() - start_time
    classical_gpu_times.append(gpu_time)
    memory_usage_gpu.append(get_memory_usage() - start_mem)
    print(f"N={n_sites:3d}: GPU={gpu_time:.3f}s")

# CPU tests for small classical sizes only
print("\nRunning CPU tests for classical sizes <= 8:")
for n_sites in tqdm([s for s in classical_sizes if s <= cpu_limit], desc="CPU Classical"):
    print(f"\nClassical Mode CPU Test (N={n_sites}):")
    sim_cpu = GeneralizedHistoricalSimulator(
        n_sites=n_sites,
        coupling_strength=J,
        field_strength=h,
        temperature=T,
        hbar_h=hbar_h,
        mode=SimulationMode.CLASSICAL,
        use_gpu=False
    )
    
    start_mem = get_memory_usage()
    start_time = time.time()
    sim_cpu.run_simulation(n_steps=n_steps_classical, dt=dt)
    cpu_time = time.time() - start_time
    classical_cpu_times.append(cpu_time)
    memory_usage_cpu.append(get_memory_usage() - start_mem)
    gpu_time = classical_gpu_times[classical_sizes.index(n_sites)]
    speedup = cpu_time / gpu_time
    print(f"N={n_sites:3d}: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, Speedup={speedup:.1f}x")

# Quantum Mode Tests
print("\nQuantum Mode Tests:")

# GPU tests for all quantum sizes
print("\nRunning GPU tests for quantum sizes:")
for n_sites in tqdm(quantum_sizes, desc="GPU Quantum"):
    print(f"\nQuantum Mode GPU Test (N={n_sites}):")
    try:
        sim_q_gpu = GeneralizedHistoricalSimulator(
            n_sites=n_sites,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.QUANTUM,
            use_gpu=True
        )
        
        start_mem = get_memory_usage()
        start_time = time.time()
        sim_q_gpu.run_simulation(n_steps=n_steps_quantum, dt=dt)
        gpu_time = time.time() - start_time
        quantum_gpu_times.append(gpu_time)
        quantum_memory_gpu.append(get_memory_usage() - start_mem)
        print(f"N={n_sites:3d}: GPU={gpu_time:.3f}s")
    except (MemoryError, RuntimeError) as e:
        print(f"Skipping remaining quantum GPU tests due to memory constraints: {str(e)}")
        break

# CPU tests for small quantum sizes only
print("\nRunning CPU tests for quantum sizes <= 8:")
for n_sites in tqdm([s for s in quantum_sizes if s <= cpu_limit], desc="CPU Quantum"):
    print(f"\nQuantum Mode CPU Test (N={n_sites}):")
    try:
        sim_q_cpu = GeneralizedHistoricalSimulator(
            n_sites=n_sites,
            coupling_strength=J,
            field_strength=h,
            temperature=T,
            hbar_h=hbar_h,
            mode=SimulationMode.QUANTUM,
            use_gpu=False
        )
        
        start_mem = get_memory_usage()
        start_time = time.time()
        sim_q_cpu.run_simulation(n_steps=n_steps_quantum, dt=dt)
        cpu_time = time.time() - start_time
        quantum_cpu_times.append(cpu_time)
        quantum_memory_cpu.append(get_memory_usage() - start_mem)
        gpu_time = quantum_gpu_times[quantum_sizes.index(n_sites)]
        speedup = cpu_time / gpu_time
        print(f"N={n_sites:3d}: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, Speedup={speedup:.1f}x")
    except (MemoryError, RuntimeError) as e:
        print(f"Skipping remaining quantum CPU tests due to memory constraints: {str(e)}")
        break

# Create timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Plotting
plt.figure(figsize=(15, 10))

# Classical performance comparison
plt.subplot(2, 2, 1)
if classical_cpu_times:
    plt.plot(classical_sizes[:len(classical_cpu_times)], classical_cpu_times, 'o-', label='CPU')
plt.plot(classical_sizes, classical_gpu_times, 's-', label='GPU')
plt.xlabel('System Size (N)')
plt.ylabel('Execution Time (s)')
plt.title('Classical Mode Performance')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xscale('log')

# Speedup factor
plt.subplot(2, 2, 2)
if classical_cpu_times:
    classical_speedup = np.array(classical_cpu_times) / np.array(classical_gpu_times[:len(classical_cpu_times)])
    plt.plot(classical_sizes[:len(classical_cpu_times)], classical_speedup, 'o-', label='Classical')
if quantum_cpu_times:
    quantum_speedup = np.array(quantum_cpu_times) / np.array(quantum_gpu_times[:len(quantum_cpu_times)])
    plt.plot(quantum_sizes[:len(quantum_cpu_times)], quantum_speedup, 's-', label='Quantum')
plt.xlabel('System Size (N)')
plt.ylabel('Speedup Factor (CPU Time / GPU Time)')
plt.title('GPU Speedup Factor')
plt.legend()
plt.grid(True)
plt.xscale('log')

# Memory usage comparison
plt.subplot(2, 2, 3)
if memory_usage_cpu:
    plt.plot(classical_sizes[:len(memory_usage_cpu)], memory_usage_cpu, 'o-', label='CPU Classical')
plt.plot(classical_sizes, memory_usage_gpu, 's-', label='GPU Classical')
if quantum_memory_cpu:
    plt.plot(quantum_sizes[:len(quantum_memory_cpu)], quantum_memory_cpu, '^-', label='CPU Quantum')
plt.plot(quantum_sizes[:len(quantum_memory_gpu)], quantum_memory_gpu, 'v-', label='GPU Quantum')
plt.xlabel('System Size (N)')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Comparison')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

# Quantum performance comparison
plt.subplot(2, 2, 4)
if quantum_cpu_times:
    plt.plot(quantum_sizes[:len(quantum_cpu_times)], quantum_cpu_times, 'o-', label='CPU')
plt.plot(quantum_sizes[:len(quantum_gpu_times)], quantum_gpu_times, 's-', label='GPU')
plt.xlabel('System Size (N)')
plt.ylabel('Execution Time (s)')
plt.title('Quantum Mode Performance')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xscale('log')

plt.tight_layout()
plt.savefig(f'benchmarks/gpu_performance_comparison_{timestamp}.png')
plt.close()

# Save numerical results
results = {
    'timestamp': timestamp,
    'parameters': {
        'classical_sizes': classical_sizes,
        'quantum_sizes': quantum_sizes,
        'cpu_limit': cpu_limit,
        'n_steps_classical': n_steps_classical,
        'n_steps_quantum': n_steps_quantum,
        'dt': dt,
        'J': J,
        'h': h,
        'T': T,
        'hbar_h': hbar_h
    },
    'classical_results': {
        'sizes': classical_sizes,
        'cpu_sizes': classical_sizes[:len(classical_cpu_times)],
        'cpu_times': classical_cpu_times,
        'gpu_times': classical_gpu_times,
        'speedup': classical_speedup.tolist() if classical_cpu_times else [],
        'memory_cpu': memory_usage_cpu,
        'memory_gpu': memory_usage_gpu
    },
    'quantum_results': {
        'sizes': quantum_sizes[:len(quantum_gpu_times)],
        'cpu_sizes': quantum_sizes[:len(quantum_cpu_times)],
        'cpu_times': quantum_cpu_times,
        'gpu_times': quantum_gpu_times,
        'speedup': quantum_speedup.tolist() if quantum_cpu_times else [],
        'memory_cpu': quantum_memory_cpu,
        'memory_gpu': quantum_memory_gpu[:len(quantum_gpu_times)]
    }
}

with open(f'benchmarks/performance_results_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\nPerformance Summary:")
print("\nClassical Mode:")
for n, gpu_t in enumerate(classical_gpu_times):
    n_sites = classical_sizes[n]
    if n < len(classical_cpu_times):
        cpu_t = classical_cpu_times[n]
        speedup = cpu_t / gpu_t
        print(f"N={n_sites:4d}: CPU={cpu_t:.3f}s, GPU={gpu_t:.3f}s, Speedup={speedup:.1f}x")
    else:
        print(f"N={n_sites:4d}: GPU={gpu_t:.3f}s (CPU test skipped)")

print("\nQuantum Mode:")
# Print CPU vs GPU comparisons for small systems
for n in range(len(quantum_cpu_times)):
    n_sites = quantum_sizes[n]
    cpu_t = quantum_cpu_times[n]
    gpu_t = quantum_gpu_times[n]
    speedup = cpu_t / gpu_t
    print(f"N={n_sites:4d}: CPU={cpu_t:.3f}s, GPU={gpu_t:.3f}s, Speedup={speedup:.1f}x")

# Print GPU-only results for larger systems
for n in range(len(quantum_cpu_times), len(quantum_gpu_times)):
    n_sites = quantum_sizes[n]
    gpu_t = quantum_gpu_times[n]
    print(f"N={n_sites:4d}: GPU={gpu_t:.3f}s (CPU test skipped)")

print(f"\nResults saved to:")
print(f"- benchmarks/gpu_performance_comparison_{timestamp}.png")
print(f"- benchmarks/performance_results_{timestamp}.json") 
"""GPU Performance Benchmarking Tool

This module provides comprehensive performance comparison between CPU and GPU implementations:
1. Execution time analysis across system sizes
2. Memory usage tracking
3. Speedup factor measurements
4. System limits determination

For usage instructions, see benchmarks/README.md
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import psutil
import os
import json
from datetime import datetime
from pathlib import Path

from historical_fidelity_simulator.simulator import GeneralizedHistoricalSimulator, SimulationMode

def setup_output_directory() -> Path:
    """Set up output directory for benchmark results.
    
    Returns:
        Path: Path to output directory
    """
    output_dir = Path('output')
    if not output_dir.exists():
        output_dir = Path('..') / 'output'
    if not output_dir.exists():
        output_dir = Path('../..') / 'output'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = output_dir / 'benchmarks'
    benchmark_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to: {benchmark_dir.absolute()}")
    return benchmark_dir

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

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmarks() -> dict:
    """Run comprehensive performance benchmarks.
    
    Returns:
        dict: Dictionary containing all benchmark results
    """
    # Arrays to store timing results
    classical_cpu_times = []
    classical_gpu_times = []
    quantum_cpu_times = []
    quantum_gpu_times = []
    memory_usage_cpu = []
    memory_usage_gpu = []
    quantum_memory_cpu = []
    quantum_memory_gpu = []
    
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
    
    # Prepare results dictionary
    if classical_cpu_times:
        classical_speedup = np.array(classical_cpu_times) / np.array(classical_gpu_times[:len(classical_cpu_times)])
    else:
        classical_speedup = []
        
    if quantum_cpu_times:
        quantum_speedup = np.array(quantum_cpu_times) / np.array(quantum_gpu_times[:len(quantum_cpu_times)])
    else:
        quantum_speedup = []
    
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
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
            'speedup': classical_speedup.tolist() if len(classical_speedup) > 0 else [],
            'memory_cpu': memory_usage_cpu,
            'memory_gpu': memory_usage_gpu
        },
        'quantum_results': {
            'sizes': quantum_sizes[:len(quantum_gpu_times)],
            'cpu_sizes': quantum_sizes[:len(quantum_cpu_times)],
            'cpu_times': quantum_cpu_times,
            'gpu_times': quantum_gpu_times,
            'speedup': quantum_speedup.tolist() if len(quantum_speedup) > 0 else [],
            'memory_cpu': quantum_memory_cpu,
            'memory_gpu': quantum_memory_gpu[:len(quantum_gpu_times)]
        }
    }
    
    return results

def plot_results(results: dict, output_dir: Path) -> None:
    """Plot benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    # Classical performance comparison
    plt.subplot(2, 2, 1)
    if results['classical_results']['cpu_times']:
        plt.plot(results['classical_results']['cpu_sizes'],
                results['classical_results']['cpu_times'], 'o-', label='CPU')
    plt.plot(results['classical_results']['sizes'],
            results['classical_results']['gpu_times'], 's-', label='GPU')
    plt.xlabel('System Size (N)')
    plt.ylabel('Execution Time (s)')
    plt.title('Classical Mode Performance')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    
    # Speedup factor
    plt.subplot(2, 2, 2)
    if results['classical_results']['speedup']:
        plt.plot(results['classical_results']['cpu_sizes'],
                results['classical_results']['speedup'], 'o-', label='Classical')
    if results['quantum_results']['speedup']:
        plt.plot(results['quantum_results']['cpu_sizes'],
                results['quantum_results']['speedup'], 's-', label='Quantum')
    plt.xlabel('System Size (N)')
    plt.ylabel('Speedup Factor (CPU Time / GPU Time)')
    plt.title('GPU Speedup Factor')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Memory usage comparison
    plt.subplot(2, 2, 3)
    if results['classical_results']['memory_cpu']:
        plt.plot(results['classical_results']['cpu_sizes'],
                results['classical_results']['memory_cpu'], 'o-', label='CPU Classical')
    plt.plot(results['classical_results']['sizes'],
            results['classical_results']['memory_gpu'], 's-', label='GPU Classical')
    if results['quantum_results']['memory_cpu']:
        plt.plot(results['quantum_results']['cpu_sizes'],
                results['quantum_results']['memory_cpu'], '^-', label='CPU Quantum')
    plt.plot(results['quantum_results']['sizes'],
            results['quantum_results']['memory_gpu'], 'v-', label='GPU Quantum')
    plt.xlabel('System Size (N)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    
    # Quantum performance comparison
    plt.subplot(2, 2, 4)
    if results['quantum_results']['cpu_times']:
        plt.plot(results['quantum_results']['cpu_sizes'],
                results['quantum_results']['cpu_times'], 'o-', label='CPU')
    plt.plot(results['quantum_results']['sizes'],
            results['quantum_results']['gpu_times'], 's-', label='GPU')
    plt.xlabel('System Size (N)')
    plt.ylabel('Execution Time (s)')
    plt.title('Quantum Mode Performance')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'gpu_performance_comparison_{results["timestamp"]}.png')
    plt.close()

def print_summary(results: dict) -> None:
    """Print summary of benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
    """
    print("\nPerformance Summary:")
    print("\nClassical Mode:")
    for n, gpu_t in enumerate(results['classical_results']['gpu_times']):
        n_sites = results['classical_results']['sizes'][n]
        if n < len(results['classical_results']['cpu_times']):
            cpu_t = results['classical_results']['cpu_times'][n]
            speedup = cpu_t / gpu_t
            print(f"N={n_sites:4d}: CPU={cpu_t:.3f}s, GPU={gpu_t:.3f}s, Speedup={speedup:.1f}x")
        else:
            print(f"N={n_sites:4d}: GPU={gpu_t:.3f}s (CPU test skipped)")
    
    print("\nQuantum Mode:")
    # Print CPU vs GPU comparisons for small systems
    for n in range(len(results['quantum_results']['cpu_times'])):
        n_sites = results['quantum_results']['cpu_sizes'][n]
        cpu_t = results['quantum_results']['cpu_times'][n]
        gpu_t = results['quantum_results']['gpu_times'][n]
        speedup = cpu_t / gpu_t
        print(f"N={n_sites:4d}: CPU={cpu_t:.3f}s, GPU={gpu_t:.3f}s, Speedup={speedup:.1f}x")
    
    # Print GPU-only results for larger systems
    for n in range(len(results['quantum_results']['cpu_times']),
                  len(results['quantum_results']['gpu_times'])):
        n_sites = results['quantum_results']['sizes'][n]
        gpu_t = results['quantum_results']['gpu_times'][n]
        print(f"N={n_sites:4d}: GPU={gpu_t:.3f}s (CPU test skipped)")

def main() -> None:
    """Run the GPU performance benchmarking tool."""
    output_dir = setup_output_directory()
    results = run_benchmarks()
    
    # Save results
    with open(output_dir / f'performance_results_{results["timestamp"]}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot and print results
    plot_results(results, output_dir)
    print_summary(results)
    
    print(f"\nResults saved to:")
    print(f"- {(output_dir / f'gpu_performance_comparison_{results["timestamp"]}.png').absolute()}")
    print(f"- {(output_dir / f'performance_results_{results["timestamp"]}.json').absolute()}")

if __name__ == '__main__':
    main() 
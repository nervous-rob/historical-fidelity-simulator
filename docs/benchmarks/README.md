# Benchmarks Documentation

This directory contains benchmark results and performance analysis for the Historical Fidelity Simulator.

## Benchmark Results

### Performance Metrics

1. **GPU Performance Comparison**
   - File: `gpu_performance_comparison_20250117_215845.png`
   - Results: `performance_results_20250117_215845.json`
   - Comparison of CPU vs GPU performance
   - Scaling with system size

2. **Evolution Comparison**
   - File: `evolution_comparison.png`
   - Classical vs quantum evolution performance
   - Time step scaling
   - Memory usage

3. **Uncertainty Products**
   - File: `uncertainty_products.png`
   - Measurement of ΔH_f Δt
   - Bound verification
   - Statistical analysis

### Physics Results

1. **Phase Diagrams**
   - `field_phase_diagram.png`: Field strength dependence
   - `information_planck_phase_diagram.png`: ℏ_h dependence
   - `field_sweep_results.png`: Parameter sweep analysis

2. **Critical Behavior**
   - `critical_exponent.png`: Critical exponent analysis
   - `classical_scaling.png`: Scaling behavior
   - `quantum_classical_differences.png`: Comparison of approaches

3. **System Comparisons**
   - `quantum_classical_comparison.png`: Detailed comparison
   - `evolution_comparison.png`: Dynamical differences
   - `information_planck_sweep.png`: ℏ_h dependence

## Performance Analysis

### GPU Acceleration

1. **System Size Scaling**
   - Linear scaling up to N=1000 sites
   - GPU advantage above N=100
   - Memory limitations at N=10000
   - Optimal batch sizes

2. **Operation Types**
   - Matrix operations: 10-100x speedup
   - State evolution: 5-50x speedup
   - Memory transfers: Potential bottleneck
   - Kernel optimization results

### Classical vs Quantum

1. **Computational Costs**
   - Classical: O(N) scaling
   - Quantum: O(2^N) scaling
   - Memory usage patterns
   - Parallelization efficiency

2. **Algorithm Comparison**
   - Monte Carlo efficiency
   - Quantum state evolution
   - Decoherence effects
   - Numerical precision

## Key Findings

1. **Performance**
   - GPU acceleration essential for N > 100
   - Quantum simulation limited to N < 20
   - Memory optimization critical
   - Parallel scaling near-linear

2. **Physics**
   - Critical temperature T_c ≈ 2.27 J/k_B
   - Universal scaling confirmed
   - Quantum-classical correspondence
   - Information bound saturation

3. **Implementation**
   - CUDA optimization effective
   - Memory management crucial
   - Error handling robust
   - Numerical stability verified

## Running Benchmarks

### Setup

```bash
# Install dependencies
pip install -e .[benchmark]

# Set up GPU environment
export CUDA_VISIBLE_DEVICES=0
```

### Execution

```bash
# Run full benchmark suite
python benchmarks/run_benchmarks.py

# Run specific benchmark
python benchmarks/run_benchmarks.py --type gpu_comparison
```

### Analysis

```bash
# Generate performance plots
python benchmarks/analyze_results.py

# Compare with previous results
python benchmarks/compare_benchmarks.py
```

## Adding New Benchmarks

1. **Create Benchmark**
   - Add script to `benchmarks/`
   - Follow template structure
   - Include documentation
   - Add to suite

2. **Validation**
   - Test with small systems
   - Verify GPU support
   - Check memory usage
   - Validate results

3. **Documentation**
   - Update this README
   - Add result analysis
   - Document parameters
   - Include examples

## Best Practices

1. **Performance Testing**
   - Use consistent hardware
   - Warm-up runs
   - Multiple iterations
   - Statistical analysis

2. **Result Recording**
   - Standard format
   - Version control
   - Hardware details
   - Parameter sets

3. **Visualization**
   - Clear labeling
   - Error bars
   - Comparison plots
   - Trend analysis 
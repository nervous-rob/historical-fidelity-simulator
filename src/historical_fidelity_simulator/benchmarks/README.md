# Performance Benchmarking Tools

This directory contains tools and scripts for benchmarking the Historical Fidelity Simulator's performance.

## Contents

- `gpu_performance.py`: Comprehensive GPU vs CPU performance comparison tool
  - Compares execution times across different system sizes
  - Analyzes memory usage patterns
  - Measures speedup factors for both classical and quantum modes
  - Generates detailed performance plots and metrics

## Running Benchmarks

To run the GPU performance comparison:

```bash
python -m historical_fidelity_simulator.benchmarks.gpu_performance
```

## Output

Benchmark results are saved in the `output/benchmarks` directory:
- Performance comparison plots (`.png` files)
- Detailed numerical results (`.json` files)
- System configuration information

## Interpreting Results

The benchmarking tools generate several key metrics:

1. **Execution Time Scaling**
   - Shows how performance scales with system size
   - Separate curves for CPU and GPU implementations
   - Log-scale plots to show scaling behavior

2. **Memory Usage**
   - Tracks memory consumption patterns
   - Compares CPU vs GPU memory requirements
   - Helps identify memory bottlenecks

3. **Speedup Factors**
   - Ratio of CPU to GPU execution times
   - Indicates optimal system sizes for GPU acceleration
   - Separate analysis for classical and quantum modes

4. **System Limits**
   - Maximum feasible system sizes
   - Memory constraints
   - Optimal configuration recommendations 
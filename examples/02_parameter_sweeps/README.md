# GPU Performance Comparison Example

This example demonstrates the performance benefits of GPU acceleration in the Historical Fidelity Simulator by comparing CPU and GPU execution times across different system sizes and simulation modes.

## Contents

- `gpu_performance_comparison.py`: Script that runs performance benchmarks and generates comparison plots
- Generated output: `gpu_performance_comparison.png` with four subplots showing detailed performance metrics

## Running the Example

To run the performance comparison:

```bash
python gpu_performance_comparison.py
```

The script will:
1. Run classical simulations on both CPU and GPU for system sizes N = [8, 16, 32, 64, 128, 256]
2. Run quantum simulations on both CPU and GPU for small systems (N ≤ 8)
3. Generate performance plots and print detailed timing information

## Understanding the Results

The script generates a figure with four subplots:

1. **Classical Mode Performance**: Shows execution time vs system size for CPU and GPU implementations
2. **GPU Speedup Factor**: Displays the ratio of CPU time to GPU time, showing how many times faster the GPU is
3. **Memory Usage**: Compares memory consumption between CPU and GPU implementations
4. **Quantum Mode Performance**: Shows execution time comparison for quantum simulations (small systems only)

### Interpreting the Results

- **Execution Time Scaling**: Both plots use log scales to show how performance scales with system size
- **Speedup Factor**: Higher values indicate better GPU performance relative to CPU
- **Memory Usage**: Shows the trade-off between CPU and GPU memory consumption
- **Quantum vs Classical**: Note that quantum simulations are limited to smaller systems due to exponential scaling

### Expected Performance

- **Classical Mode**: GPU acceleration typically shows better speedup for larger systems
- **Quantum Mode**: GPU benefits may be more pronounced due to matrix operations
- **Memory Usage**: GPU implementation may use more memory due to CUDA overhead
- **First Run**: May be slower due to CUDA kernel compilation

## System Requirements

- NVIDIA GPU with CUDA support
- CuPy installed with appropriate CUDA toolkit
- Sufficient GPU memory for larger systems
- Python packages: numpy, matplotlib, tqdm, psutil

## Notes

- The example automatically adjusts quantum simulation sizes to prevent out-of-memory errors
- Memory measurements include both host and device memory
- Performance may vary based on your specific hardware configuration 
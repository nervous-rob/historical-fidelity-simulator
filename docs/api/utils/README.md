# Utils Package API Reference

The utils package provides shared utilities and helper functions for the Historical Fidelity Simulator.

## Modules

- [GPU Accelerator](gpu_accelerator.md) - GPU acceleration utilities
- [Visualization](visualization.md) - Plotting and visualization tools

## Overview

The utils package provides:
- GPU acceleration for large-scale simulations
- Visualization tools for results analysis
- Common numerical operations
- Performance optimization utilities

## Key Components

### GPU Acceleration
- CUDA/CuPy integration
- Matrix operations
- State evolution
- Performance profiling

### Visualization
- Phase diagram plotting
- Time series visualization
- Parameter sweep plots
- Statistical analysis plots

## Usage Examples

### GPU Acceleration

```python
from historical_fidelity_simulator.utils import GPUAccelerator

# Initialize GPU accelerator
gpu = GPUAccelerator(n_sites=10, use_gpu=True)

# Perform GPU-accelerated computation
energies = gpu.compute_classical_energies(state, J=1.0, h=0.5)
```

### Visualization

```python
from historical_fidelity_simulator.utils import plot_phase_diagram

# Plot phase diagram
plot_phase_diagram(
    temperatures=temps,
    field_strengths=fields,
    order_parameter=magnetizations,
    title="Phase Diagram",
    save_path="phase_diagram.png"
)
```

## Performance Considerations

1. **GPU Acceleration**
   - Automatic device selection
   - Memory management
   - Batch processing
   - Performance monitoring

2. **Visualization**
   - Efficient data handling
   - Memory-friendly plotting
   - High-quality output
   - Interactive capabilities

## Dependencies

### Core Dependencies
- NumPy: Numerical operations
- CuPy: GPU acceleration
- Matplotlib: Visualization
- SciPy: Scientific computing

### Optional Components
- CUDA Toolkit: GPU support
- Plotly: Interactive plots
- Seaborn: Statistical plots

## Extension Points

The utils package can be extended through:
1. Custom GPU kernels
2. New visualization styles
3. Additional optimization methods
4. Performance profiling tools 
# GPU Accelerator Module

The GPU accelerator module provides GPU-accelerated versions of core computations using:
1. Numba CUDA for classical simulations
2. CuPy for quantum matrix operations

## Functions

### matrix_exp_gpu

```python
def matrix_exp_gpu(A: Union[np.ndarray, "cp.ndarray"], order: int = 6) -> "cp.ndarray"
```

Compute matrix exponential using Padé approximation.

**Parameters:**
- `A`: Input matrix (CuPy array)
- `order`: Order of Padé approximation (higher = more accurate but slower)

**Returns:**
- Matrix exponential as a CuPy array

**Raises:**
- `ImportError`: If CuPy is not available
- `ValueError`: If unsupported order is specified

### CUDA Kernels

#### compute_local_energies_kernel

```python
@cuda.jit
def compute_local_energies_kernel(spins, J, h, energies, n_sites)
```

CUDA kernel for parallel computation of local energies.

**Parameters:**
- `spins`: Spin configuration array
- `J`: Coupling strength
- `h`: Field strength
- `energies`: Output energy array
- `n_sites`: Number of sites

#### metropolis_update_kernel

```python
@cuda.jit
def metropolis_update_kernel(spins, energies, J, h, T, rand_nums, accepted, n_sites)
```

CUDA kernel for parallel Metropolis updates.

**Parameters:**
- `spins`: Spin configuration array
- `energies`: Energy array
- `J`: Coupling strength
- `h`: Field strength
- `T`: Temperature
- `rand_nums`: Random numbers for acceptance
- `accepted`: Output acceptance array
- `n_sites`: Number of sites

### get_gpu_device

```python
def get_gpu_device() -> Optional[cuda.Device]
```

Get the GPU device if available.

**Returns:**
- CUDA device object or None if not available

## Classes

### GPUAccelerator

Class for managing GPU-accelerated computations.

#### Constructor

```python
def __init__(self, n_sites: int, use_gpu: bool = True)
```

Initialize the GPU accelerator.

**Parameters:**
- `n_sites`: Number of sites in the system
- `use_gpu`: Whether to use GPU acceleration

#### Methods

##### compute_classical_energies

```python
def compute_classical_energies(
    self,
    spins: np.ndarray,
    J: float,
    h: float
) -> np.ndarray
```

Compute local energies using GPU acceleration if available.

**Parameters:**
- `spins`: Spin configuration array
- `J`: Coupling strength
- `h`: Field strength

**Returns:**
- Array of local energies

##### metropolis_update

```python
def metropolis_update(
    self,
    spins: np.ndarray,
    energies: np.ndarray,
    J: float,
    h: float,
    T: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Perform Metropolis updates using GPU acceleration if available.

**Parameters:**
- `spins`: Spin configuration array
- `energies`: Energy array
- `J`: Coupling strength
- `h`: Field strength
- `T`: Temperature

**Returns:**
- Tuple of (updated spins, updated energies, acceptance flags)

## Usage Examples

### Basic GPU Acceleration

```python
from historical_fidelity_simulator.utils import GPUAccelerator
import numpy as np

# Initialize accelerator
gpu = GPUAccelerator(n_sites=1000, use_gpu=True)

# Create random spin configuration
spins = np.random.choice([-1, 1], size=1000)

# Compute energies
energies = gpu.compute_classical_energies(
    spins=spins,
    J=1.0,
    h=0.5
)
```

### Metropolis Updates

```python
# Perform Metropolis update
updated_spins, updated_energies, accepted = gpu.metropolis_update(
    spins=spins,
    energies=energies,
    J=1.0,
    h=0.5,
    T=0.1
)

# Check acceptance rate
acceptance_rate = accepted.mean()
print(f"Acceptance rate: {acceptance_rate}")
```

### Matrix Operations

```python
import numpy as np
from historical_fidelity_simulator.utils import matrix_exp_gpu

# Create test matrix
A = np.random.randn(100, 100)

# Compute matrix exponential on GPU
exp_A = matrix_exp_gpu(A, order=6)
```

## Implementation Details

### GPU Memory Management
1. **Device Arrays**
   - Pre-allocated device memory
   - Efficient data transfer
   - Memory reuse
   - Automatic cleanup

2. **CUDA Grid Configuration**
   - Optimal thread block size
   - Dynamic grid dimensions
   - Load balancing

### Computation Patterns
1. **Classical Computations**
   - Parallel energy calculations
   - Vectorized spin updates
   - Efficient random number usage
   - Local memory optimization

2. **Matrix Operations**
   - Padé approximation
   - Optimized linear algebra
   - Memory-efficient implementation
   - Numerical stability

## Performance Considerations

1. **Memory Transfer**
   - Minimized host-device transfers
   - Batch processing
   - Pinned memory usage
   - Asynchronous operations

2. **Computational Efficiency**
   - Coalesced memory access
   - Shared memory usage
   - Warp-level optimization
   - Load balancing

3. **Fallback Mechanisms**
   - Automatic CPU fallback
   - Performance monitoring
   - Error handling
   - Resource management

## Error Handling

The implementation includes:
- Device availability checks
- Memory allocation validation
- Numerical stability monitoring
- Graceful CPU fallback 
# Critical Analysis Module

The critical analysis module provides tools for analyzing critical behavior and phase transitions, with a focus on finite-size scaling and data collapse analysis.

## Functions

### compute_correlation_length

```python
def compute_correlation_length(
    fidelity_history: npt.NDArray[np.float64],
    lattice_spacing: float = 1.0
) -> float
```

Compute correlation length from fidelity history.

**Parameters:**
- `fidelity_history`: Time series of fidelity measurements
- `lattice_spacing`: Physical spacing between lattice sites

**Returns:**
- Estimated correlation length

### analyze_finite_size_scaling

```python
def analyze_finite_size_scaling(
    temps: npt.NDArray[np.float64],
    fidelities: Dict[int, List[float]],
    t_c: float,
    size_range: Optional[Tuple[float, float]] = None
) -> Dict[str, float]
```

Analyze finite-size scaling behavior near critical point.

**Parameters:**
- `temps`: Array of temperatures
- `fidelities`: Dict mapping system sizes to fidelity lists
- `t_c`: Critical temperature
- `size_range`: Optional tuple of (min_size, max_size) for fitting

**Returns:**
- Dict containing critical exponents and scaling parameters:
  - `nu`: Correlation length exponent
  - `beta`: Order parameter exponent
  - `quality`: Quality of data collapse

### compute_susceptibility

```python
def compute_susceptibility(
    fidelity_history: npt.NDArray[np.float64],
    temperature: float
) -> float
```

Compute fidelity susceptibility.

**Parameters:**
- `fidelity_history`: Time series of fidelity measurements
- `temperature`: System temperature

**Returns:**
- Computed susceptibility

## Usage Examples

### Correlation Length Analysis

```python
from historical_fidelity_simulator.analysis import compute_correlation_length
import numpy as np

# Generate sample fidelity history
times = np.linspace(0, 100, 1000)
fidelity = np.exp(-times/10) + 0.1*np.random.randn(len(times))

# Compute correlation length
xi = compute_correlation_length(
    fidelity_history=fidelity,
    lattice_spacing=1.0
)
print(f"Correlation length: {xi}")
```

### Finite-Size Scaling

```python
from historical_fidelity_simulator.analysis import analyze_finite_size_scaling

# Prepare data for different system sizes
temps = np.linspace(2.0, 4.0, 20)
fidelities = {
    N: simulate_system(N, temps) for N in [8, 16, 32, 64]
}

# Analyze scaling behavior
scaling_results = analyze_finite_size_scaling(
    temps=temps,
    fidelities=fidelities,
    t_c=2.27,  # Known critical temperature
    size_range=(8, 64)
)

print(f"Critical exponents: ν={scaling_results['nu']}, β={scaling_results['beta']}")
```

### Susceptibility Calculation

```python
from historical_fidelity_simulator.analysis import compute_susceptibility

# Compute susceptibility from time series
chi = compute_susceptibility(
    fidelity_history=fidelity_data,
    temperature=2.0
)
print(f"Susceptibility: {chi}")
```

## Implementation Details

### Correlation Length
1. **Computation Method**
   - Autocorrelation function
   - Exponential decay fitting
   - Noise handling
   - Error estimation

2. **Fitting Procedure**
   - Exponential model
   - Least squares optimization
   - Valid range selection
   - Robustness checks

### Finite-Size Scaling
1. **Scaling Analysis**
   - Data collapse method
   - Critical exponent extraction
   - Quality assessment
   - Size range selection

2. **Optimization**
   - Objective function
   - Parameter bounds
   - Convergence criteria
   - Error handling

### Susceptibility
1. **Computation**
   - Variance calculation
   - Temperature scaling
   - Fluctuation-dissipation
   - Statistical analysis

## Performance Considerations

1. **Numerical Efficiency**
   - Optimized array operations
   - Memory-efficient algorithms
   - Parallel computation options
   - Caching strategies

2. **Scaling Properties**
   - System size dependence
   - Temperature resolution
   - Memory requirements
   - Computational complexity

## Error Handling

The implementation includes:
- Robust fitting procedures
- NaN/Inf checking
- Exception handling
- Input validation

## Dependencies

- NumPy: Array operations
- SciPy: Optimization and fitting
- Typing: Type hints
- Optional: Parallel processing 
# Analysis Package API Reference

The analysis package provides tools for analyzing simulation results, particularly focusing on critical behavior and phase transitions.

## Modules

- [Critical Analysis](critical_analysis.md) - Phase transition and critical behavior analysis

## Overview

The analysis package provides tools for:
- Critical point detection
- Phase transition analysis
- Scaling behavior studies
- Statistical analysis
- Data processing utilities

## Key Features

### Critical Analysis
- Critical point detection
- Finite-size scaling
- Critical exponents
- Universal behavior

### Statistical Analysis
- Data aggregation
- Error estimation
- Uncertainty quantification
- Correlation analysis

### Data Processing
- Time series analysis
- Parameter sweeps
- Ensemble averaging
- Trend detection

## Usage Examples

### Critical Point Analysis

```python
from historical_fidelity_simulator.analysis import analyze_critical_point

# Analyze critical behavior
critical_data = analyze_critical_point(
    temperatures=temps,
    magnetizations=mags,
    system_sizes=sizes
)

print(f"Critical temperature: {critical_data['T_c']}")
print(f"Critical exponent β: {critical_data['beta']}")
```

### Finite-Size Scaling

```python
from historical_fidelity_simulator.analysis import finite_size_scaling

# Perform finite-size scaling analysis
scaling_results = finite_size_scaling(
    observable_values=values,
    system_sizes=sizes,
    temperatures=temps,
    critical_temp=T_c
)

# Plot scaling collapse
plot_scaling_collapse(scaling_results)
```

### Phase Transition Detection

```python
from historical_fidelity_simulator.analysis import detect_phase_transition

# Detect phase transition
transition = detect_phase_transition(
    parameter_values=params,
    order_parameter=order_param,
    method='susceptibility'
)

print(f"Transition point: {transition['point']}")
print(f"Transition width: {transition['width']}")
```

## Implementation Details

### Critical Analysis
1. **Point Detection**
   - Susceptibility peaks
   - Binder cumulant
   - Order parameter collapse
   - Correlation length

2. **Scaling Analysis**
   - Data collapse
   - Critical exponents
   - Universal ratios
   - Finite-size effects

### Statistical Methods
1. **Error Analysis**
   - Bootstrap resampling
   - Jackknife estimation
   - Error propagation
   - Confidence intervals

2. **Data Processing**
   - Binning analysis
   - Trend removal
   - Noise reduction
   - Outlier detection

## Performance Considerations

1. **Computational Efficiency**
   - Optimized algorithms
   - Parallel processing
   - Memory management
   - Caching strategies

2. **Data Handling**
   - Efficient storage
   - Lazy evaluation
   - Streaming processing
   - Memory-mapped files

## Dependencies

### Core Dependencies
- NumPy: Numerical operations
- SciPy: Scientific computing
- Pandas: Data analysis
- Statsmodels: Statistical analysis

### Optional Components
- Scikit-learn: Machine learning
- emcee: MCMC sampling
- corner: Corner plots
- h5py: Data storage

## Extension Points

The analysis package can be extended through:
1. Custom analysis methods
2. New statistical tools
3. Additional plotting functions
4. Data processing pipelines 
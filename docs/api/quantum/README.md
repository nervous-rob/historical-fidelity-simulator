# Quantum Package API Reference

The quantum package implements quantum-specific functionality for the Historical Fidelity Simulator.

## Modules

- [Evolution](evolution.md) - Quantum state evolution and dynamics
- [Operators](operators.md) - Quantum operators and measurements

## Overview

The quantum package provides implementations for:
- Quantum state evolution using the Lindblad formalism
- Hamiltonian construction and manipulation
- Decoherence effects modeling
- Quantum measurement operators
- Fidelity calculations
- Density matrix operations

## Key Concepts

### State Representation
- Pure states represented as state vectors
- Mixed states represented as density matrices
- Support for both sparse and dense representations
- GPU-accelerated operations for large systems

### Evolution Methods
- Unitary evolution under Hamiltonian
- Non-unitary evolution with Lindblad terms
- Time-dependent Hamiltonians
- Adaptive time stepping

### Measurement and Observables
- Standard quantum observables
- Custom observable definitions
- Expectation value calculations
- Uncertainty quantification

### Decoherence Models
- Temperature-dependent decoherence
- Site-specific noise models
- Environmental coupling
- Markovian approximations

## Usage Examples

### Basic State Evolution
```python
from historical_fidelity_simulator.quantum import evolve_state, create_hamiltonian

# Create Hamiltonian
H = create_hamiltonian(n_sites=5, J=1.0, h=0.5)

# Initial state
psi0 = create_initial_state(n_sites=5)

# Evolve state
psi_t = evolve_state(psi0, H, t=1.0, dt=0.01)
```

### Decoherence Effects
```python
from historical_fidelity_simulator.quantum import LindbladEvolution

# Create Lindblad operators
gamma = 0.1  # Decoherence strength
L_ops = create_lindblad_operators(n_sites=5, gamma=gamma)

# Evolution with decoherence
evolution = LindbladEvolution(H, L_ops)
rho_t = evolution.run(rho0, t=1.0)
```

### Measurement
```python
from historical_fidelity_simulator.quantum import measure_observable

# Define observable
Sz = create_spin_observable('z')

# Measure expectation value
expectation_value = measure_observable(psi_t, Sz)
```

## Performance Considerations

1. **Memory Usage**
   - Use sparse representations for large systems
   - Implement selective state storage
   - Monitor memory consumption

2. **GPU Acceleration**
   - Available for state evolution
   - Optimized matrix operations
   - Batch processing capabilities

3. **Optimization Strategies**
   - Symmetry exploitation
   - Parallel evolution
   - Adaptive algorithms

## Error Handling

The package implements comprehensive error checking:
- State normalization verification
- Hermiticity checks for operators
- Dimensional compatibility validation
- Numerical stability monitoring

## Extension Points

The quantum package can be extended through:
1. Custom Hamiltonian definitions
2. New decoherence models
3. Additional measurement operators
4. Alternative evolution schemes

## Dependencies

- NumPy: Core numerical operations
- SciPy: Sparse matrix operations
- QuTiP: Quantum toolbox
- CuPy: GPU acceleration (optional) 
# Classical Package API Reference

The classical package implements classical physics simulations for the Historical Fidelity Simulator.

## Modules

- [Ising](ising.md) - Classical Ising model implementation
- [Dynamics](dynamics.md) - Classical dynamics and time evolution

## Overview

The classical package provides implementations for:
- Classical Ising model dynamics
- Phase space trajectories
- Energy calculations
- Classical fidelity analogs
- Monte Carlo methods

## Key Concepts

### State Representation
- Discrete spin configurations
- Phase space coordinates
- Collective variables
- Statistical ensembles

### Evolution Methods
- Metropolis-Hastings dynamics
- Molecular dynamics
- Langevin dynamics
- Deterministic evolution

### Observables
- Energy calculations
- Magnetization
- Correlation functions
- Order parameters

### Thermal Effects
- Temperature coupling
- Heat bath dynamics
- Fluctuation-dissipation
- Ergodicity considerations

## Usage Examples

### Basic Ising Simulation
```python
from historical_fidelity_simulator.classical import IsingModel

# Create Ising model
model = IsingModel(
    n_sites=10,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=0.1
)

# Run simulation
results = model.run_simulation(n_steps=1000)
```

### Phase Space Evolution
```python
from historical_fidelity_simulator.classical import ClassicalDynamics

# Initialize dynamics
dynamics = ClassicalDynamics(
    n_particles=100,
    temperature=0.1,
    dt=0.01
)

# Evolve system
trajectory = dynamics.evolve(
    initial_state,
    n_steps=1000
)
```

### Observable Measurement
```python
# Compute observables
energy = model.compute_energy()
magnetization = model.compute_magnetization()
correlation = model.compute_correlation_function(r=2)
```

## Performance Considerations

1. **Computational Efficiency**
   - Vectorized operations
   - Local update schemes
   - Efficient neighbor lookup
   - Parallelization options

2. **Memory Management**
   - State compression
   - Trajectory storage
   - Observable caching
   - Memory-efficient algorithms

3. **Scaling Properties**
   - Linear with system size
   - Efficient for large systems
   - Parallelizable operations

## Error Handling

The package implements comprehensive error checking:
- Parameter validation
- State consistency checks
- Energy conservation monitoring
- Numerical stability tests

## Extension Points

The classical package can be extended through:
1. Custom dynamics implementations
2. Alternative update schemes
3. New observable definitions
4. Additional analysis methods

## Dependencies

### Core Dependencies
- NumPy: Numerical operations
- SciPy: Scientific computing
- Numba: JIT compilation (optional)
- Matplotlib: Visualization

### Optional Components
- CuPy: GPU acceleration
- MPI4Py: Parallel computing
- H5Py: Data storage 
# Quantum Module Documentation

## Overview

The quantum module implements the quantum mechanical aspects of the historical fidelity simulator. It provides functionality for:

1. Constructing and evolving quantum states
2. Computing quantum observables
3. Implementing decoherence through Lindblad operators
4. Calculating uncertainty products for the ℏ_h bound

## Core Components

### 1. Quantum Operators (`operators.py`)

#### Transverse-Field Ising Hamiltonian
```python
H = construct_ising_hamiltonian(n_sites=4, coupling_strength=1.0, field_strength=0.5)
```
Constructs the Hamiltonian H = -J Σ σ_i^x σ_{i+1}^x - h Σ σ_i^z where:
- J (`coupling_strength`): Nearest-neighbor coupling
- h (`field_strength`): Transverse field strength
- Supports both periodic and open boundary conditions

#### Lindblad Operators
```python
c_ops = construct_lindblad_operators(n_sites=4, temperature=1.0)
```
Creates decoherence operators modeling system-environment interaction:
- Dephasing (σ_z)
- Energy relaxation (σ_-)
- Energy excitation (σ_+)
- Rates satisfy detailed balance at given temperature

#### Observables
```python
magnetization, entropy = compute_observables(state, n_sites=4)
```
Computes physical observables:
- Magnetization: Average σ_z per site
- von Neumann entropy of reduced density matrix

### 2. Quantum Evolution (`evolution.py`)

#### QuantumEvolver Class
```python
evolver = QuantumEvolver(
    hamiltonian=H,
    n_sites=4,
    temperature=1.0
)
```
Handles time evolution of quantum states using QuTiP's master equation solver.

Key methods:
1. `evolve_state(initial_state, dt)`: Time evolution with decoherence
2. `compute_fidelity(state)`: Historical fidelity metric
3. `compute_uncertainty_product(initial_state, dt)`: ΔH_f Δt calculation

## Usage Examples

### 1. Basic Evolution
```python
from historical_fidelity_simulator.quantum import (
    construct_ising_hamiltonian,
    QuantumEvolver
)
import qutip as qt

# Create Hamiltonian
H = construct_ising_hamiltonian(
    n_sites=2,
    coupling_strength=1.0,
    field_strength=0.5
)

# Initialize evolver
evolver = QuantumEvolver(
    hamiltonian=H,
    n_sites=2,
    temperature=1.0
)

# Create initial state |↑↑⟩
up = qt.basis([2], 0)
initial_state = qt.tensor(up, up)

# Evolve state
final_state, _ = evolver.evolve_state(
    initial_state=initial_state,
    dt=0.1
)
```

### 2. Computing Uncertainty Products
```python
# Compute ΔH_f Δt
uncertainty = evolver.compute_uncertainty_product(
    initial_state=initial_state,
    dt=0.1,
    n_samples=10
)
```

### 3. Analyzing Temperature Dependence
```python
# Compare decoherence at different temperatures
cold_evolver = QuantumEvolver(H, n_sites=2, temperature=0.5)
hot_evolver = QuantumEvolver(H, n_sites=2, temperature=2.0)

cold_uncertainty = cold_evolver.compute_uncertainty_product(initial_state, dt=0.1)
hot_uncertainty = hot_evolver.compute_uncertainty_product(initial_state, dt=0.1)
```

## Error Handling

The module includes comprehensive input validation:

1. System Parameters
   - `n_sites` must be positive integer
   - `coupling_strength` and `field_strength` must be numeric
   - `temperature` must be positive

2. Quantum States
   - Must be QuTiP Qobj with correct dimensions
   - Pure states: dims = [[2,...,2], [1,...,1]]
   - Mixed states: dims = [[2,...,2], [2,...,2]]

3. Evolution Parameters
   - `dt` must be positive
   - `n_samples` must be ≥ 2 for uncertainty calculations

## Implementation Notes

1. **State Evolution**
   - Uses QuTiP's `mesolve` for Lindblad master equation
   - Includes both unitary evolution and decoherence
   - Preserves state normalization

2. **Decoherence Model**
   - Temperature-dependent rates
   - Satisfies detailed balance
   - Includes dephasing and energy exchange

3. **Performance Considerations**
   - System size limited by QuTiP's tensor product capabilities
   - Memory usage scales exponentially with n_sites
   - Recommended maximum: n_sites ≤ 10

## Dependencies

- NumPy: Array operations and linear algebra
- QuTiP: Quantum dynamics and operator manipulation
- SciPy: Scientific computing utilities 
# Quantum Operators Module

The quantum operators module provides the quantum operators needed for the simulation, including the transverse-field Ising Hamiltonian and Lindblad operators for decoherence.

## Functions

### validate_system_params

```python
def validate_system_params(n_sites: int, coupling_strength: float, field_strength: float) -> None
```

Validate system parameters for quantum operators.

**Parameters:**
- `n_sites`: Number of sites in the system
- `coupling_strength`: J, the coupling strength between neighboring spins
- `field_strength`: h, the transverse field strength

**Raises:**
- `ValueError`: If any parameters are invalid
- `TypeError`: If parameters have wrong type

### validate_temperature

```python
def validate_temperature(temperature: float) -> None
```

Validate temperature parameter.

**Parameters:**
- `temperature`: Temperature of the system

**Raises:**
- `ValueError`: If temperature is invalid
- `TypeError`: If temperature has wrong type

### construct_ising_hamiltonian

```python
def construct_ising_hamiltonian(
    n_sites: int,
    coupling_strength: float,
    field_strength: float,
    periodic: bool = True
) -> qt.Qobj
```

Construct the transverse-field Ising Hamiltonian:
H = -J Σ σ_i^x σ_{i+1}^x - h Σ σ_i^z

**Parameters:**
- `n_sites`: Number of sites in the system
- `coupling_strength`: J, the coupling strength between neighboring spins
- `field_strength`: h, the transverse field strength
- `periodic`: Whether to use periodic boundary conditions

**Returns:**
- The Hamiltonian as a QuTiP Qobj

**Raises:**
- `ValueError`: If parameters are invalid
- `TypeError`: If parameters have wrong type

### construct_lindblad_operators

```python
def construct_lindblad_operators(
    n_sites: int,
    temperature: float,
    decoherence_strength: Optional[float] = None
) -> List[qt.Qobj]
```

Construct Lindblad operators for decoherence. Creates operators that model the system's interaction with a thermal bath, causing decoherence proportional to temperature.

**Parameters:**
- `n_sites`: Number of sites in the system
- `temperature`: Temperature of the environment
- `decoherence_strength`: Optional scaling factor for decoherence rate. If None, uses sqrt(temperature)

**Returns:**
- List of Lindblad operators

**Raises:**
- `ValueError`: If parameters are invalid
- `TypeError`: If parameters have wrong type

### compute_observables

```python
def compute_observables(
    state: qt.Qobj,
    n_sites: int
) -> Tuple[float, float]
```

Compute relevant observables from the quantum state.

**Parameters:**
- `state`: The quantum state as a QuTiP Qobj
- `n_sites`: Number of sites in the system

**Returns:**
- Tuple of (magnetization, entropy)

**Raises:**
- `ValueError`: If parameters are invalid
- `TypeError`: If parameters have wrong type

## Usage Examples

### Creating Ising Hamiltonian

```python
import qutip as qt
from historical_fidelity_simulator.quantum import construct_ising_hamiltonian

# Create Hamiltonian for 3-site system
H = construct_ising_hamiltonian(
    n_sites=3,
    coupling_strength=1.0,  # J
    field_strength=0.5,     # h
    periodic=True
)

# Print Hamiltonian
print(H)
```

### Setting Up Decoherence

```python
from historical_fidelity_simulator.quantum import construct_lindblad_operators

# Create Lindblad operators for thermal bath
c_ops = construct_lindblad_operators(
    n_sites=3,
    temperature=0.1,
    decoherence_strength=0.05
)

# Use in master equation solver
result = qt.mesolve(H, psi0, times, c_ops=c_ops)
```

### Computing Observables

```python
from historical_fidelity_simulator.quantum import compute_observables

# Create initial state (all spins up)
psi0 = qt.basis([2, 2, 2], [0, 0, 0])

# Compute observables
magnetization, entropy = compute_observables(psi0, n_sites=3)
print(f"Magnetization: {magnetization}")
print(f"Entropy: {entropy}")
```

## Implementation Details

### Hamiltonian Construction

The transverse-field Ising Hamiltonian is constructed as:
1. Nearest-neighbor coupling terms (-J σ_i^x σ_{i+1}^x)
2. Transverse field terms (-h σ_i^z)
3. Optional periodic boundary term

### Decoherence Model

The decoherence model includes three types of Lindblad operators:
1. Dephasing (σ_z)
2. Energy relaxation (σ_-)
3. Energy excitation (σ_+)

Rates satisfy detailed balance at the given temperature.

### Observable Calculations

1. **Magnetization**
   - Computed as average σ_z per site
   - Normalized by system size

2. **Entropy**
   - Von Neumann entropy of reduced density matrix
   - Traces out all but one site
   - Handles both pure and mixed states

## Performance Considerations

1. **Memory Usage**
   - Operators stored in sparse format
   - System size scales exponentially
   - Careful memory management needed

2. **Computational Efficiency**
   - Uses QuTiP's optimized tensor operations
   - Exploits symmetries where possible
   - Caches common operators

3. **Scaling**
   - Hamiltonian size: 2^n × 2^n
   - Number of Lindblad operators: 3n
   - Observable calculations: O(2^n) 
# Quantum Evolution Module

The quantum evolution module handles the time evolution of quantum states using QuTiP's master equation solver, including both unitary evolution and decoherence effects.

## Functions

### validate_evolution_params

```python
def validate_evolution_params(dt: float, n_samples: Optional[int] = None) -> None
```

Validate evolution parameters.

**Parameters:**
- `dt`: Time step for evolution
- `n_samples`: Optional number of samples for uncertainty calculation

**Raises:**
- `ValueError`: If parameters are invalid
- `TypeError`: If parameters have wrong type

### validate_quantum_state

```python
def validate_quantum_state(state: qt.Qobj, n_sites: int) -> None
```

Validate a quantum state.

**Parameters:**
- `state`: Quantum state to validate
- `n_sites`: Expected number of sites

**Raises:**
- `ValueError`: If state is invalid
- `TypeError`: If state has wrong type

## Classes

### QuantumEvolver

Class to handle quantum state evolution.

#### Constructor

```python
def __init__(
    self,
    hamiltonian: qt.Qobj,
    n_sites: int,
    temperature: float,
    decoherence_strength: Optional[float] = None
)
```

Initialize the quantum evolver.

**Parameters:**
- `hamiltonian`: The system Hamiltonian
- `n_sites`: Number of sites in the system
- `temperature`: Temperature of the environment
- `decoherence_strength`: Optional scaling factor for decoherence. If None, uses sqrt(temperature)

**Raises:**
- `ValueError`: If parameters are invalid
- `TypeError`: If parameters have wrong type

#### Methods

##### evolve_state

```python
def evolve_state(
    self,
    initial_state: qt.Qobj,
    dt: float,
    store_states: bool = False
) -> Tuple[qt.Qobj, Optional[Dict[str, Any]]]
```

Evolve the quantum state forward in time.

**Parameters:**
- `initial_state`: Initial quantum state
- `dt`: Time step for evolution
- `store_states`: Whether to store intermediate states

**Returns:**
- Tuple of (final state, optional dict with evolution data)

**Raises:**
- `ValueError`: If parameters are invalid
- `TypeError`: If parameters have wrong type

##### compute_fidelity

```python
def compute_fidelity(self, state: qt.Qobj) -> float
```

Compute the historical fidelity metric for a quantum state.

For quantum systems, we define fidelity as the negative expectation value of the Hamiltonian (normalized by system size), analogous to the classical case.

**Parameters:**
- `state`: Quantum state to compute fidelity for

**Returns:**
- The fidelity metric

**Raises:**
- `ValueError`: If state is invalid
- `TypeError`: If state has wrong type

##### compute_uncertainty_product

```python
def compute_uncertainty_product(
    self,
    initial_state: qt.Qobj,
    dt: float,
    n_samples: int = 10
) -> float
```

Compute the uncertainty product ΔH_f Δt.

**Parameters:**
- `initial_state`: Initial quantum state
- `dt`: Time interval
- `n_samples`: Number of intermediate points to sample

**Returns:**
- The uncertainty product

**Raises:**
- `ValueError`: If parameters are invalid
- `TypeError`: If parameters have wrong type

## Usage Examples

### Basic Evolution

```python
from historical_fidelity_simulator.quantum import QuantumEvolver
import qutip as qt

# Create Hamiltonian for 3 sites
n_sites = 3
sx = qt.sigmax()
sz = qt.sigmaz()
H = sum([-J * qt.tensor([sx if i==j or i==j+1 else qt.qeye(2) 
         for j in range(n_sites)]) for i in range(n_sites-1)])

# Initialize evolver
evolver = QuantumEvolver(
    hamiltonian=H,
    n_sites=n_sites,
    temperature=0.1
)

# Create initial state (all spins up)
psi0 = qt.basis([2]*n_sites, [0]*n_sites)

# Evolve state
final_state, _ = evolver.evolve_state(psi0, dt=0.1)

# Compute fidelity
fidelity = evolver.compute_fidelity(final_state)
```

### Evolution with State History

```python
# Evolve and store intermediate states
final_state, evolution_data = evolver.evolve_state(
    psi0,
    dt=0.1,
    store_states=True
)

# Access evolution data
times = evolution_data['times']
states = evolution_data['states']
```

### Uncertainty Product Calculation

```python
# Compute uncertainty product
uncertainty = evolver.compute_uncertainty_product(
    initial_state=psi0,
    dt=0.1,
    n_samples=20
)
print(f"Uncertainty product: {uncertainty}")
```

## Implementation Notes

1. **State Validation**
   - Pure states must have dimensions `[[2]*n_sites, [1]*n_sites]`
   - Mixed states must have dimensions `[[2]*n_sites, [2]*n_sites]`
   - Automatic validation of state normalization

2. **Evolution Method**
   - Uses QuTiP's `mesolve` for master equation evolution
   - Includes both unitary (Hamiltonian) and non-unitary (Lindblad) terms
   - Adaptive time stepping for numerical stability

3. **Decoherence**
   - Temperature-dependent decoherence through Lindblad operators
   - Optional custom decoherence strength scaling
   - Site-local decoherence effects

4. **Performance**
   - Efficient sparse matrix operations
   - Optional state history storage
   - Configurable sampling for uncertainty calculations 
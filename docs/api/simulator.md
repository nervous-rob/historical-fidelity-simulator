# Simulator Module API Reference

The simulator module provides the core functionality for running historical fidelity simulations in both classical and quantum modes.

## Classes

### SimulationMode

An enumeration defining the available simulation modes.

```python
class SimulationMode(Enum):
    CLASSICAL = "classical"  # Classical physics simulation
    QUANTUM = "quantum"      # Quantum physics simulation
```

### GeneralizedHistoricalSimulator

The main simulator class implementing both classical and quantum approaches.

#### Constructor

```python
def __init__(
    self,
    n_sites: int,
    coupling_strength: float,
    field_strength: float,
    temperature: float,
    hbar_h: float,
    mode: SimulationMode = SimulationMode.CLASSICAL,
    f_scaling: Optional[Callable[[float, float], float]] = None,
    use_gpu: bool = True
) -> None
```

**Parameters:**
- `n_sites`: Number of sites in the system
- `coupling_strength`: Interaction strength between sites (J)
- `field_strength`: External field strength (h)
- `temperature`: System temperature (T)
- `hbar_h`: Information Planck constant
- `mode`: Simulation mode (classical or quantum)
- `f_scaling`: Custom scaling function f(E, T)
- `use_gpu`: Whether to use GPU acceleration if available

**Raises:**
- `ValueError`: If any parameters are invalid

#### Public Methods

##### compute_generalized_bound

```python
def compute_generalized_bound(self) -> float
```

Compute ℏ_h * f(⟨E⟩, T).

**Returns:**
- Current value of the generalized bound

##### run_simulation

```python
def run_simulation(
    self,
    n_steps: int,
    dt: float,
    measure_interval: int = 100
) -> List[Dict]
```

Run the simulation for a specified number of steps.

**Parameters:**
- `n_steps`: Number of time steps
- `dt`: Time step size
- `measure_interval`: Number of steps between measurements

**Returns:**
- List of dictionaries containing measurement results at each interval

#### Internal Methods

##### _default_scaling_function

```python
def _default_scaling_function(self, energy: float, temperature: float) -> float
```

Default implementation of the scaling function: f(E, T) = 1 + E/T + α(T/Tₖ)^β

**Parameters:**
- `energy`: System energy
- `temperature`: Current temperature

**Returns:**
- Scaling factor

##### _construct_quantum_hamiltonian

```python
def _construct_quantum_hamiltonian(self) -> qt.Qobj
```

Construct the quantum Hamiltonian using QuTiP.

**Returns:**
- QuTiP Quantum object representing the Hamiltonian

##### _compute_local_energy

```python
def _compute_local_energy(self, site: int) -> float
```

Compute local energy contribution from a single site.

**Parameters:**
- `site`: Site index

**Returns:**
- Local energy contribution

**Raises:**
- `ValueError`: If called in quantum mode

##### _compute_quantum_fidelity

```python
def _compute_quantum_fidelity(self) -> float
```

Compute fidelity for quantum state.

**Returns:**
- Quantum fidelity measure

##### _compute_fidelity

```python
def _compute_fidelity(self) -> float
```

Compute current historical fidelity metric.

**Returns:**
- Current fidelity value

##### _evolve_quantum_state

```python
def _evolve_quantum_state(self, dt: float) -> None
```

Evolve quantum state with Hamiltonian + Lindblad decoherence.

**Parameters:**
- `dt`: Time step

## Usage Example

```python
from historical_fidelity_simulator import GeneralizedHistoricalSimulator, SimulationMode

# Create a quantum simulator
simulator = GeneralizedHistoricalSimulator(
    n_sites=10,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=0.1,
    hbar_h=1.0,
    mode=SimulationMode.QUANTUM
)

# Run simulation
results = simulator.run_simulation(
    n_steps=1000,
    dt=0.01,
    measure_interval=10
)

# Compute bound
bound = simulator.compute_generalized_bound() 
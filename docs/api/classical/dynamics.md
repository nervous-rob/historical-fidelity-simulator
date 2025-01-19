# Classical Dynamics Module

The classical dynamics module implements Metropolis-Hastings dynamics for the classical Ising model.

## Classes

### MetropolisDynamics

Metropolis-Hastings dynamics for classical Ising model.

#### Constructor

```python
def __init__(
    self,
    model: IsingModel,
    random_seed: Optional[int] = None
) -> None
```

Initialize the dynamics.

**Parameters:**
- `model`: IsingModel instance to evolve
- `random_seed`: Optional seed for random number generator

#### Methods

##### step

```python
def step(self, dt: float, n_flips: Optional[int] = None) -> Dict
```

Perform one time step of evolution.

**Parameters:**
- `dt`: Time step size
- `n_flips`: Number of spin flips to attempt (default: n_sites)

**Returns:**
- Dictionary with step results containing:
  - `time`: Current simulation time
  - `energy`: Total system energy
  - `magnetization`: System magnetization
  - `energy_change`: Total energy change in step
  - `acceptance_rate`: Fraction of accepted flips

##### run

```python
def run(
    self,
    total_time: float,
    dt: float,
    measure_interval: int = 1
) -> List[Dict]
```

Run dynamics for specified time.

**Parameters:**
- `total_time`: Total simulation time
- `dt`: Time step size
- `measure_interval`: Number of steps between measurements

**Returns:**
- List of measurement dictionaries

##### compute_fidelity

```python
def compute_fidelity(self) -> float
```

Compute current historical fidelity.

**Returns:**
- Historical fidelity metric based on energy (negative energy per site)

##### compute_generalized_bound

```python
def compute_generalized_bound(
    self,
    hbar_h: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    t_c: Optional[float] = None
) -> float
```

Compute generalized uncertainty bound.

**Parameters:**
- `hbar_h`: Information Planck constant
- `alpha`: Scaling prefactor
- `beta`: Critical exponent
- `t_c`: Critical temperature (default: current T)

**Returns:**
- Value of generalized bound

## Usage Examples

### Basic Evolution

```python
from historical_fidelity_simulator.classical import IsingModel, MetropolisDynamics

# Create model
model = IsingModel(
    n_sites=10,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=0.1
)

# Initialize dynamics
dynamics = MetropolisDynamics(model, random_seed=42)

# Run simulation
results = dynamics.run(
    total_time=100.0,
    dt=0.1,
    measure_interval=10
)

# Analyze results
for step in results:
    print(f"t={step['time']}: E={step['energy']}, M={step['magnetization']}")
```

### Single Step Evolution

```python
# Perform single time step
step_result = dynamics.step(dt=0.1, n_flips=20)

print(f"Energy change: {step_result['energy_change']}")
print(f"Acceptance rate: {step_result['acceptance_rate']}")
```

### Computing Bounds

```python
# Compute fidelity and bound
fidelity = dynamics.compute_fidelity()
bound = dynamics.compute_generalized_bound(
    hbar_h=1.0,
    alpha=0.5,
    beta=-0.8,
    t_c=3.0
)

print(f"Fidelity: {fidelity}")
print(f"Generalized bound: {bound}")
```

## Implementation Details

### Time Evolution
1. **Monte Carlo Steps**
   - Random site selection
   - Metropolis acceptance criterion
   - Multiple flip attempts per time step
   - Configurable number of flips

2. **Measurements**
   - Energy tracking
   - Magnetization computation
   - Acceptance rate monitoring
   - Time series recording

### Fidelity Calculation
1. **Historical Fidelity**
   - Based on energy per site
   - Negative sign convention
   - System size normalization

2. **Generalized Bound**
   - Temperature-dependent scaling
   - Critical behavior near T_c
   - Configurable exponents

## Performance Considerations

1. **Computational Efficiency**
   - Local updates only
   - Efficient random sampling
   - Optional measurement intervals
   - History storage management

2. **Memory Usage**
   - Selective history recording
   - Measurement aggregation
   - Configurable recording frequency

3. **Scaling Properties**
   - Time step cost: O(n_flips)
   - Memory usage: O(n_measurements)
   - Overall scaling: O(N) per time step

## Error Handling

The implementation includes:
- Parameter validation
- Random seed management
- Time step consistency checks
- Measurement validation 
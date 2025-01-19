# Classical Ising Model Module

The classical Ising model module implements a classical Ising model with nearest-neighbor interactions for historical fidelity simulation.

## Classes

### IsingModel

Classical Ising model with nearest-neighbor interactions.

#### Constructor

```python
def __init__(
    self,
    n_sites: int,
    coupling_strength: float,
    field_strength: float,
    temperature: float,
    periodic: bool = True
) -> None
```

Initialize the Ising model.

**Parameters:**
- `n_sites`: Number of sites in the system
- `coupling_strength`: J parameter for neighbor interactions
- `field_strength`: h parameter for external field
- `temperature`: System temperature T
- `periodic`: Whether to use periodic boundary conditions

#### Methods

##### get_neighbors

```python
def get_neighbors(self, site: int) -> List[int]
```

Get indices of neighboring sites.

**Parameters:**
- `site`: Index of the site

**Returns:**
- List of neighbor indices, ordered [left_neighbor, right_neighbor]

##### local_energy

```python
def local_energy(self, site: int) -> float
```

Compute local energy contribution from a site.

**Parameters:**
- `site`: Index of the site

**Returns:**
- Local energy contribution

##### total_energy

```python
def total_energy(self) -> float
```

Compute total energy of the system.

**Returns:**
- Total system energy

##### magnetization

```python
def magnetization(self) -> float
```

Compute system magnetization (order parameter).

**Returns:**
- Average magnetization per site

##### flip_spin

```python
def flip_spin(self, site: int) -> Tuple[float, bool]
```

Attempt to flip a spin using Metropolis algorithm.

**Parameters:**
- `site`: Index of the site to flip

**Returns:**
- Tuple of (energy change, whether flip was accepted)

##### get_state

```python
def get_state(self) -> npt.NDArray[np.int8]
```

Get current spin configuration.

**Returns:**
- Array of spin values

## Usage Examples

### Basic Model Setup

```python
from historical_fidelity_simulator.classical import IsingModel

# Create model
model = IsingModel(
    n_sites=10,
    coupling_strength=1.0,  # J
    field_strength=0.5,     # h
    temperature=0.1,        # T
    periodic=True
)

# Get initial state
state = model.get_state()
```

### Monte Carlo Steps

```python
import numpy as np

# Perform Monte Carlo steps
n_steps = 1000
for _ in range(n_steps):
    site = np.random.randint(model.n_sites)
    delta_e, accepted = model.flip_spin(site)
    
    if accepted:
        print(f"Accepted spin flip at site {site}, ΔE = {delta_e}")
```

### Observable Measurements

```python
# Measure observables
E = model.total_energy()
M = model.magnetization()

print(f"Energy per site: {E/model.n_sites}")
print(f"Magnetization: {M}")

# Get local properties
for site in range(model.n_sites):
    neighbors = model.get_neighbors(site)
    local_E = model.local_energy(site)
    print(f"Site {site}: neighbors = {neighbors}, energy = {local_E}")
```

## Implementation Details

### State Representation
- Spins stored as ±1 values in NumPy array
- Efficient memory layout for fast access
- Integer type for spin values

### Energy Calculations
1. **Local Energy**
   - Nearest-neighbor interactions (-J Σ s_i s_j)
   - External field contribution (-h s_i)
   - Boundary conditions handled automatically

2. **Total Energy**
   - Sum of local energy contributions
   - Avoids double-counting of bonds
   - O(N) computation time

### Monte Carlo Dynamics
1. **Metropolis Algorithm**
   - Single-spin flip proposals
   - Energy-based acceptance criterion
   - Temperature-dependent acceptance rates
   - Special handling for T=0 case

2. **Boundary Conditions**
   - Optional periodic boundaries
   - Neighbor lookup handles boundaries
   - Consistent energy computation

## Performance Considerations

1. **Memory Efficiency**
   - Compact state representation
   - No redundant storage
   - Efficient neighbor lists

2. **Computational Speed**
   - Local updates O(1)
   - Vectorized operations where possible
   - Efficient random number usage

3. **Scaling**
   - Memory: O(N)
   - Energy computation: O(N)
   - Single update: O(1)

## Error Handling

The implementation includes:
- Parameter validation in constructor
- Boundary checking in neighbor lookup
- Type checking for NumPy arrays
- Temperature regime handling 
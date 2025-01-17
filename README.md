# Historical Fidelity Simulator

A unified simulator for studying historical fidelity through both classical and quantum approaches, investigating the concept of an information Planck constant (ℏ_h) and its generalized uncertainty relation.

## Features

- Dual-mode simulation (classical/quantum)
- Customizable scaling function for critical phenomena
- Phase transition analysis
- Historical fidelity tracking
- Support for system sizes up to 100 sites (classical) or 10 sites (quantum)
- Comprehensive visualization tools
- Advanced quantum features:
  - Temperature-dependent decoherence
  - Lindblad master equation evolution
  - Von Neumann entropy tracking
  - Quantum uncertainty products
  - Transverse-field Ising model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nervousrob/historical-fidelity-simulator.git
cd historical-fidelity-simulator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode with all dependencies:
```bash
pip install -e ".[dev]"  # Includes development dependencies
# OR
pip install -e .         # Only runtime dependencies
```

## Quick Start

### Classical Simulation
```python
from historical_fidelity_simulator import GeneralizedHistoricalSimulator, SimulationMode
from historical_fidelity_simulator.utils import plot_simulation_history

# Create a classical simulator
sim = GeneralizedHistoricalSimulator(
    n_sites=20,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=2.0,
    hbar_h=1.0,
    mode=SimulationMode.CLASSICAL
)

# Run simulation
history = sim.run_simulation(n_steps=5000, dt=0.1)

# Plot results
fig, ax = plot_simulation_history(history, "Classical Simulation")
```

### Quantum Simulation with Decoherence
```python
from historical_fidelity_simulator.quantum import (
    construct_ising_hamiltonian,
    QuantumEvolver
)
import qutip as qt

# Create Hamiltonian
H = construct_ising_hamiltonian(
    n_sites=4,
    coupling_strength=1.0,
    field_strength=0.5,
    periodic=True  # Use periodic boundary conditions
)

# Initialize quantum evolver with temperature-dependent decoherence
evolver = QuantumEvolver(
    hamiltonian=H,
    n_sites=4,
    temperature=2.0
)

# Create initial state |↑↑↑↑⟩
up = qt.basis([2], 0)
initial_state = qt.tensor([up] * 4)

# Evolve with decoherence
final_state, evolution_data = evolver.evolve_state(
    initial_state=initial_state,
    dt=0.1,
    store_states=True
)

# Compute observables
magnetization, entropy = compute_observables(final_state, n_sites=4)

# Calculate uncertainty product
uncertainty = evolver.compute_uncertainty_product(
    initial_state=initial_state,
    dt=0.1,
    n_samples=10
)
```

### Custom Scaling Function with Critical Behavior
```python
def custom_scaling(energy: float, temperature: float) -> float:
    """Custom critical scaling function."""
    T_c = 2.0  # Critical temperature
    alpha = 0.5  # Scaling prefactor
    beta = -0.8  # Critical exponent
    
    return 1.0 + energy/temperature + alpha * (temperature/T_c)**beta

sim = GeneralizedHistoricalSimulator(
    n_sites=20,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=2.0,
    hbar_h=1.0,
    f_scaling=custom_scaling
)
```

## Project Structure
```
historical-fidelity-simulator/
├── historical_fidelity_simulator/  # Main package directory
│   ├── simulator.py               # Core simulation framework
│   ├── quantum/                   # Quantum-specific implementations
│   │   ├── operators.py          # Quantum operators and observables
│   │   ├── evolution.py          # Quantum state evolution
│   │   └── README.md             # Quantum module documentation
│   ├── classical/                # Classical physics implementations
│   └── utils/                    # Shared utilities and visualization
├── tests/                        # Test suite
│   ├── test_quantum.py          # Quantum module tests
│   ├── test_simulator.py        # Core simulator tests
│   └── test_visualization.py    # Visualization tests
├── examples/                     # Example notebooks
├── docs/                        # Documentation
└── benchmarks/                  # Performance benchmarks
```

## Theory

The simulator implements a generalized uncertainty relation for historical fidelity:

```
ΔH_f Δt ≥ ℏ_h f(⟨E⟩, T)
```

where:
- `ΔH_f`: Change in historical fidelity
- `Δt`: Time interval
- `ℏ_h`: Information Planck constant
- `f(E,T)`: Critical scaling function
- `⟨E⟩`: Average energy/distortion
- `T`: Information temperature

The default scaling function includes critical behavior:

```
f(E,T) = 1 + E/T + α(T/T_c)^β
```

### Quantum Implementation

The quantum mode implements:
1. Transverse-field Ising Hamiltonian:
   ```
   H = -J Σ σ_i^x σ_{i+1}^x - h Σ σ_i^z
   ```

2. Temperature-dependent decoherence through Lindblad operators:
   - Dephasing (σ_z)
   - Energy relaxation (σ_-)
   - Energy excitation (σ_+)
   - Rates satisfy detailed balance

3. Quantum observables:
   - Magnetization
   - Von Neumann entropy
   - Historical fidelity
   - Uncertainty products

## Development

- Follow PEP 8 style guide
- Use type hints
- Maintain test coverage above 90%
- Run tests: `pytest tests/`
- Format code: `black src/ tests/`
- Type checking: `mypy src/`

### Performance Notes

- Classical mode: Efficient for systems up to 100 sites
- Quantum mode: Limited by QuTiP tensor operations
  - Recommended maximum: 10 sites
  - Memory usage scales exponentially with system size
  - Use periodic boundary conditions for better physics

## License

MIT License

## Citation

If you use this simulator in your research, please cite:
[Citation information to be added] 
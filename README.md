# Historical Fidelity Simulator

A unified simulator for studying historical fidelity through both classical and quantum approaches, investigating the concept of an information Planck constant (ℏ_h) and its generalized uncertainty relation.

## Features

- Dual-mode simulation (classical/quantum)
- Customizable scaling function for critical phenomena
- Phase transition analysis
- Historical fidelity tracking
- Support for system sizes up to 100 sites
- Comprehensive visualization tools
- Quantum decoherence modeling
- Critical scaling analysis

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

### Quantum Simulation
```python
# Create a quantum simulator
sim = GeneralizedHistoricalSimulator(
    n_sites=4,  # Smaller system size for quantum simulation
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=2.0,
    hbar_h=1.0,
    mode=SimulationMode.QUANTUM
)

# Run simulation with decoherence
history = sim.run_simulation(n_steps=1000, dt=0.1)
```

### Custom Scaling Function
```python
def custom_scaling(energy: float, temperature: float) -> float:
    """Custom critical scaling function."""
    T_c = 2.0
    return 1.0 + energy/temperature + 0.5 * (temperature/T_c)**(-0.8)

sim = GeneralizedHistoricalSimulator(
    n_sites=20,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=2.0,
    hbar_h=1.0,
    f_scaling=custom_scaling
)
```

## Development

- Follow PEP 8 style guide
- Use type hints
- Maintain test coverage above 90%
- Run tests: `pytest tests/`
- Format code: `black src/ tests/`
- Type checking: `mypy src/`

### Project Structure
```
historical-fidelity-simulator/
├── historical_fidelity_simulator/  # Main package directory
│   ├── simulator.py               # Core simulation framework
│   ├── quantum/                  # Quantum-specific implementations
│   ├── classical/                # Classical physics implementations
│   └── utils/                    # Shared utilities and visualization
├── tests/                       # Test suite
├── examples/                    # Example notebooks
├── docs/                       # Documentation
└── benchmarks/                 # Performance benchmarks
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

## License

MIT License

## Citation

If you use this simulator in your research, please cite:
[Citation information to be added] 
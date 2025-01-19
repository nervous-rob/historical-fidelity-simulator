# Getting Started with Historical Fidelity Simulator

This guide will help you get started with using the Historical Fidelity Simulator package for both classical and quantum simulations.

## Installation

```bash
pip install historical-fidelity-simulator
```

For GPU acceleration support:
```bash
pip install historical-fidelity-simulator[gpu]
```

## Basic Usage

### 1. Simple Classical Simulation

```python
from historical_fidelity_simulator import GeneralizedHistoricalSimulator, SimulationMode

# Create a classical simulator
simulator = GeneralizedHistoricalSimulator(
    n_sites=5,                # Number of sites
    coupling_strength=1.0,    # J parameter
    field_strength=0.5,       # h parameter
    temperature=0.1,          # Temperature T
    hbar_h=1.0,              # Information Planck constant
    mode=SimulationMode.CLASSICAL
)

# Run simulation
results = simulator.run_simulation(
    n_steps=1000,            # Total time steps
    dt=0.01,                 # Time step size
    measure_interval=10      # Measure every 10 steps
)

# Analyze results
for step in results:
    print(f"Time: {step['time']}, Fidelity: {step['fidelity']}")
```

### 2. Quantum Simulation with GPU Acceleration

```python
# Create a quantum simulator with GPU acceleration
simulator = GeneralizedHistoricalSimulator(
    n_sites=10,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=0.1,
    hbar_h=1.0,
    mode=SimulationMode.QUANTUM,
    use_gpu=True
)

# Run simulation
results = simulator.run_simulation(n_steps=1000, dt=0.01)

# Compute generalized bound
bound = simulator.compute_generalized_bound()
print(f"Generalized Bound: {bound}")
```

### 3. Custom Scaling Function

```python
def custom_scaling(energy: float, temperature: float) -> float:
    """Custom scaling function for the generalized bound."""
    return 1.0 + energy/temperature

# Create simulator with custom scaling
simulator = GeneralizedHistoricalSimulator(
    n_sites=5,
    coupling_strength=1.0,
    field_strength=0.5,
    temperature=0.1,
    hbar_h=1.0,
    f_scaling=custom_scaling
)
```

## Common Tasks

### Parameter Sweeps

```python
# Perform parameter sweep over field strength
field_strengths = np.linspace(0.1, 2.0, 20)
results = []

for h in field_strengths:
    simulator = GeneralizedHistoricalSimulator(
        n_sites=10,
        coupling_strength=1.0,
        field_strength=h,
        temperature=0.1,
        hbar_h=1.0
    )
    sim_results = simulator.run_simulation(n_steps=1000, dt=0.01)
    results.append({
        'h': h,
        'final_fidelity': sim_results[-1]['fidelity']
    })
```

### Phase Transition Detection

```python
# Scan temperature range near critical point
temperatures = np.linspace(2.0, 4.0, 20)
fidelities = []

for T in temperatures:
    simulator = GeneralizedHistoricalSimulator(
        n_sites=20,
        coupling_strength=1.0,
        field_strength=0.1,
        temperature=T,
        hbar_h=1.0
    )
    results = simulator.run_simulation(n_steps=1000, dt=0.01)
    fidelities.append(results[-1]['fidelity'])

# Plot results
plt.plot(temperatures, fidelities)
plt.xlabel('Temperature')
plt.ylabel('Fidelity')
plt.title('Phase Transition Detection')
plt.show()
```

## Best Practices

1. **System Size**
   - Start with small systems (n_sites ≤ 10) for testing
   - Use GPU acceleration for larger systems
   - Consider memory constraints for quantum simulations

2. **Time Evolution**
   - Choose dt based on energy scales (J, h)
   - Verify convergence by varying dt
   - Monitor conservation laws

3. **Performance**
   - Use GPU acceleration for large systems
   - Adjust measurement interval based on needs
   - Profile memory usage for quantum simulations

4. **Error Handling**
   - Always check parameter validity
   - Monitor numerical stability
   - Verify physical constraints

## Next Steps

- Check out the [examples](examples/) directory for more complex scenarios
- Read the [API documentation](api/README.md) for detailed reference
- Review the [architecture documentation](architecture.md) for system design
- See [development standards](development_standards_and_project_goals.md) for contributing 
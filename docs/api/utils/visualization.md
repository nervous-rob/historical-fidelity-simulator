# Visualization Module

The visualization module provides plotting utilities for analyzing and visualizing simulation results.

## Functions

### plot_simulation_history

```python
def plot_simulation_history(
    history: List[Dict],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple[Figure, Axes]
```

Plot simulation history including fidelity and bounds.

**Parameters:**
- `history`: List of simulation history dictionaries
- `title`: Optional plot title
- `figsize`: Figure size (width, height)

**Returns:**
- Tuple of (Figure, Axes) for further customization

### plot_phase_diagram

```python
def plot_phase_diagram(
    parameter_values: np.ndarray,
    fidelities: np.ndarray,
    uncertainty_products: np.ndarray,
    bounds: np.ndarray,
    param_name: str = 'Temperature',
    figsize: Tuple[float, float] = (12, 4)
) -> Tuple[Figure, Axes]
```

Plot phase diagram showing fidelity, uncertainty product, and bounds.

**Parameters:**
- `parameter_values`: Array of parameter values (e.g., temperature)
- `fidelities`: Array of fidelity values
- `uncertainty_products`: Array of ΔH_f Δt products
- `bounds`: Array of ℏ_h bounds
- `param_name`: Name of the parameter being varied
- `figsize`: Figure size (width, height)

**Returns:**
- Tuple of (Figure, Axes) for further customization

### plot_state_evolution

```python
def plot_state_evolution(
    states: List[np.ndarray],
    times: List[float],
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple[Figure, Axes]
```

Plot evolution of classical spin states.

**Parameters:**
- `states`: List of spin state arrays
- `times`: List of corresponding times
- `figsize`: Figure size (width, height)

**Returns:**
- Tuple of (Figure, Axes) for further customization

### plot_quantum_observables

```python
def plot_quantum_observables(
    history: List[Dict],
    observables: Dict[str, List[float]],
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple[Figure, Axes]
```

Plot quantum observables over time.

**Parameters:**
- `history`: Simulation history
- `observables`: Dictionary of observable names and their values
- `figsize`: Figure size (width, height)

**Returns:**
- Tuple of (Figure, Axes) for further customization

## Usage Examples

### Simulation History

```python
from historical_fidelity_simulator.utils import plot_simulation_history

# Plot simulation results
fig, ax = plot_simulation_history(
    history=simulation_results,
    title="System Evolution",
    figsize=(12, 6)
)

# Customize plot
ax.set_ylim(-2, 2)
ax.set_xlabel("Time (ℏ/J)")
plt.savefig("simulation_history.png")
```

### Phase Diagram

```python
from historical_fidelity_simulator.utils import plot_phase_diagram

# Plot phase diagram
fig, (ax1, ax2, ax3) = plot_phase_diagram(
    parameter_values=temperatures,
    fidelities=fidelity_values,
    uncertainty_products=uncertainty_values,
    bounds=bound_values,
    param_name="Temperature (J/k_B)",
    figsize=(15, 5)
)

# Add critical temperature line
for ax in (ax1, ax2, ax3):
    ax.axvline(x=T_c, color='k', linestyle='--', alpha=0.5)
plt.savefig("phase_diagram.png")
```

### State Evolution

```python
from historical_fidelity_simulator.utils import plot_state_evolution

# Plot spin state evolution
fig, ax = plot_state_evolution(
    states=spin_states,
    times=time_points,
    figsize=(10, 8)
)

# Customize colormap
ax.set_title("Spin Dynamics")
plt.savefig("state_evolution.png")
```

### Quantum Observables

```python
from historical_fidelity_simulator.utils import plot_quantum_observables

# Plot quantum observables
fig, ax = plot_quantum_observables(
    history=simulation_history,
    observables={
        "Magnetization": mag_values,
        "Energy": energy_values,
        "Entropy": entropy_values
    }
)

# Add annotations
ax.annotate("Critical Point", xy=(t_c, 0))
plt.savefig("quantum_observables.png")
```

## Implementation Details

### Plot Components

1. **Simulation History**
   - Fidelity vs time
   - Uncertainty bounds
   - Shaded uncertainty regions
   - Customizable styling

2. **Phase Diagrams**
   - Three-panel layout
   - Parameter dependence
   - Ratio analysis
   - Bound comparison

3. **State Evolution**
   - Heatmap visualization
   - Time-site representation
   - Spin configuration
   - Color-coded states

4. **Quantum Observables**
   - Multiple observable tracking
   - Time evolution
   - Expectation values
   - Legend and labels

## Style Guidelines

1. **Colors**
   - Blue: Fidelity/primary data
   - Red: Bounds/limits
   - Green: Uncertainty products
   - Gray: Grid/guidelines

2. **Layout**
   - Clear axes labels
   - Informative titles
   - Proper scaling
   - Grid for readability

3. **Customization**
   - Returnable figure objects
   - Modifiable properties
   - Saveable outputs
   - High-quality export

## Performance Considerations

1. **Memory Usage**
   - Efficient data handling
   - Array operations
   - Figure cleanup
   - Resource management

2. **Rendering**
   - Vector graphics
   - Resolution control
   - Interactive vs static
   - Export optimization

## Dependencies

- Matplotlib: Core plotting
- NumPy: Data handling
- SciPy: Optional analysis
- Seaborn: Optional styling 
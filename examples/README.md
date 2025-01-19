# Historical Fidelity Simulator Examples

This directory contains example scripts and notebooks demonstrating various features and use cases of the Historical Fidelity Simulator.

## Directory Structure

- `01_basic_usage/` - Basic examples to get started with the simulator
  - `getting_started.py` - Simple example demonstrating core functionality
  
- `02_parameter_sweeps/` - Examples of parameter space exploration
  - `field_strength_sweep.py` - Analysis of system behavior under varying field strengths
  - `information_planck_sweep.py` - Study of quantum effects with varying ℏ_h
  - `temperature_coupling_sweep.py` - Phase space exploration with temperature and coupling
  
- `03_phase_transitions/` - Critical phenomena and phase transition analysis
  - `critical_behavior.py` - Analysis of phase transitions and critical behavior
  
- `04_classical_vs_quantum/` - Comparative analysis between classical and quantum modes
  - `fidelity_comparison.py` - Detailed comparison of classical and quantum behaviors

## Running the Examples

Each example can be run directly using Python:

```bash
python examples/scripts/<category>/<script_name>.py
```

All examples automatically create and manage their output in the `output/` directory, with each script creating its own subdirectory for results.

## Example Categories

### Basic Usage
Demonstrates fundamental features of the simulator, including:
- Basic simulation setup and configuration
- Running classical and quantum simulations
- Plotting and analyzing results
- Output handling and data management

### Parameter Sweeps
Shows how to explore parameter spaces and analyze system behavior:
- Field strength dependencies
- Temperature and coupling relationships
- Information Planck constant effects
- Performance optimization techniques

### Phase Transitions
Explores critical phenomena and phase transitions:
- Finite-size scaling analysis
- Critical exponent estimation
- Correlation function behavior
- Order parameter scaling

### Classical vs Quantum Comparison
Provides detailed comparisons between classical and quantum behaviors:
- Time evolution characteristics
- Response to temperature and coupling
- Uncertainty relations and bounds
- GPU-accelerated analysis

## Output Structure

Each example script creates its output in a dedicated subdirectory under `output/`:
```
output/
├── getting_started/
├── field_strength_sweep/
├── information_planck_sweep/
├── temperature_coupling_sweep/
├── critical_behavior/
└── fidelity_comparison/
```

## Converting to Notebooks

You can convert any script to a Jupyter notebook using the provided utility:

```bash
python examples/convert_to_notebooks.py
```

This will create equivalent notebooks in the `notebooks/` directory. 
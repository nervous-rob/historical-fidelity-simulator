# Historical Fidelity Simulator: Developer Guide

## Project Overview
This project implements a unified simulator for studying historical fidelity through both classical and quantum approaches. The core functionality is built around the concept of an information Planck constant (ℏ_h) and its generalized uncertainty relation.

## Core Components

### GeneralizedHistoricalSimulator Class
Location: src/simulator.py (lines 119-299)

Key features:
- Dual-mode operation (classical/quantum)
- Customizable scaling function
- Phase transition analysis
- Historical fidelity tracking

### System Requirements
- Python 3.8+
- NumPy
- SciPy
- QuTiP (for quantum simulations)
- Matplotlib (for visualization)

## Implementation Guidelines

### State Initialization
- Classical Mode: Random ±1 spins
- Quantum Mode: Initial |0...0⟩ product state
- System size (n_sites) should be configurable from 10 to 100

### Energy Calculations
Classical Mode:
- Local energy = -h*σᵢ - J*Σⱼ σᵢσⱼ
- Total energy normalized by system size
- Metropolis updates with β = 1/T

Quantum Mode:
- Transverse field Ising Hamiltonian
- QuTiP-based evolution
- Lindblad operators for decoherence

### Critical Scaling Function
Default implementation:
f(E, T) = 1 + E/T + α(T/Tₖ)^β
where:
- E: system energy
- T: temperature
- Tₖ: critical temperature (default = 3.0)
- α: scaling prefactor (default = 0.5)
- β: critical exponent (default = -0.8)

## Testing Protocol

1. Basic Validation
   - Verify energy conservation in classical mode
   - Check quantum state normalization
   - Test boundary conditions (T → 0, T → ∞)

2. Phase Transition Analysis
   - Temperature range: 0.5 to 5.0
   - Coupling strength: 0.1 to 2.0
   - Measure fidelity, uncertainty product, and bounds

3. Finite-Size Scaling
   - Test system sizes: 10, 20, 50, 100
   - Track convergence of observables
   - Document finite-size effects

## Data Collection

### Required Measurements
- Time series of fidelity
- Uncertainty products (ΔHf Δt)
- Generalized bounds
- Phase transition indicators

### Output Format
JSON structure per simulation:
{
    "parameters": {
        "n_sites": int,
        "temperature": float,
        "coupling": float,
        "mode": str
    },
    "measurements": [
        {
            "time": float,
            "fidelity": float,
            "bound": float,
            "step": int
        }
    ]
}

## Performance Considerations
- Use vectorized operations where possible
- Implement parallel temperature sweeps
- Cache Hamiltonian construction in quantum mode
- Limit history storage based on measurement interval

## Development Timeline
Month 1: Classical implementation
Month 2: Quantum extension
Month 3: Parameter sweeps
Month 4: Phase transition analysis
Month 5: Documentation and optimization

## Known Limitations
- QuTiP memory constraints above 100 sites
- Classical mode local minima at low T
- Periodic boundary conditions only
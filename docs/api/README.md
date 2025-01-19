# Historical Fidelity Simulator API Reference

This directory contains detailed API documentation for the Historical Fidelity Simulator package.

## Core Components

- [Simulator](simulator.md) - Core simulation framework
- [Quantum](quantum/README.md) - Quantum physics implementations
- [Classical](classical/README.md) - Classical physics implementations
- [Analysis](analysis/README.md) - Analysis tools and utilities
- [Utils](utils/README.md) - Shared utilities

## Getting Started

For a quick introduction to using the API, see our [Getting Started Guide](../getting_started.md).

## Module Organization

### Core Simulator
The `simulator.py` module provides the main simulation framework, handling both quantum and classical simulations.

### Quantum Package
The `quantum/` package implements quantum-specific functionality:
- State evolution
- Hamiltonian construction
- Decoherence effects
- Measurement operators

### Classical Package
The `classical/` package implements classical physics simulations:
- Phase space dynamics
- Energy calculations
- Trajectory tracking

### Analysis Package
The `analysis/` package provides tools for:
- Data processing
- Visualization
- Statistical analysis
- Phase transition detection

### Utils Package
The `utils/` package contains shared utilities:
- Numerical methods
- I/O operations
- Parameter validation
- Performance optimization

## API Stability

This API reference covers the stable public API. Components marked as "experimental" may change in future releases. 
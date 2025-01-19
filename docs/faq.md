# Frequently Asked Questions

## General Questions

### What is the Historical Fidelity Simulator?
The Historical Fidelity Simulator is a Python package for studying quantum-classical correspondence through fidelity measurements. It implements both quantum and classical versions of the Ising model to explore phase transitions and dynamical properties.

### What can I use it for?
- Studying quantum-classical correspondence
- Investigating phase transitions
- Analyzing fidelity bounds
- Teaching statistical mechanics
- Research in quantum dynamics

### What are the system requirements?
- Python 3.8+
- CUDA-capable GPU (optional, for acceleration)
- 4GB RAM minimum (16GB recommended for large systems)
- Linux, macOS, or Windows

## Installation Questions

### How do I install the package?
```bash
# Basic installation
pip install historical-fidelity-simulator

# With all optional dependencies
pip install "historical-fidelity-simulator[all]"

# Development installation
git clone https://github.com/historical-fidelity-simulator
cd historical-fidelity-simulator
pip install -e ".[dev]"
```

### Why isn't GPU acceleration working?
1. Check CUDA installation
2. Verify GPU compatibility
3. Install CUDA toolkit
4. Set environment variables
See [Troubleshooting Guide](troubleshooting.md) for details.

## Usage Questions

### What system sizes can I simulate?
- Classical systems: Up to ~10⁶ sites
- Quantum systems: Up to ~20 qubits
- GPU acceleration helps with larger systems
- Memory is the main limitation

### How do I choose parameters?
- Temperature (T): 0 to 5 J/k_B
- Coupling (J): Typically set to 1.0
- Field (h): 0 to 5 J
- Critical temperature: T_c ≈ 2.27 J/k_B

### How long should I run simulations?
- Thermalization: ~10⁴ steps per site
- Measurements: ~10⁵ steps for good statistics
- Check autocorrelation times
- Monitor observables for convergence

### What are typical error bounds?
- Statistical errors: ~1/√N for N measurements
- Systematic errors: ~1/L for system size L
- Quantum errors: Controlled by time step
- Fidelity bounds: Set by ℏ_h

## Technical Questions

### How does the classical-quantum correspondence work?
1. Classical system: Phase space trajectories
2. Quantum system: Unitary evolution
3. Fidelity measures overlap/distance
4. Bounds relate uncertainties

### What is the historical fidelity?
- Measures similarity of trajectories
- Quantum analog of classical overlap
- Bounded by uncertainty relations
- Key to quantum-classical correspondence

### How are phase transitions detected?
1. Order parameter (magnetization)
2. Susceptibility peaks
3. Finite-size scaling
4. Fidelity metrics

### What numerical methods are used?
- Classical: Metropolis algorithm
- Quantum: Trotter decomposition
- Integration: Symplectic methods
- Analysis: Finite-size scaling

## Performance Questions

### How can I improve performance?
1. Enable GPU acceleration
2. Optimize system size
3. Use batch processing
4. Parallelize analysis
See [Performance Guide](performance.md) for details.

### What are typical run times?
- Classical (N=1000): ~1s per 10⁶ steps
- Quantum (N=10): ~1s per 10³ steps
- GPU speedup: 10-100x
- Scales with system size

### How much memory is needed?
- Classical: O(N) memory
- Quantum: O(2^N) memory
- GPU memory: 2-8GB recommended
- Disk space: ~100MB for results

## Development Questions

### How can I contribute?
1. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

### How do I report bugs?
1. Check existing issues
2. Provide minimal example
3. Include system information
4. Describe expected behavior

### How do I add new features?
1. Discuss in issues first
2. Follow development standards
3. Add tests and documentation
4. Submit pull request

### How do I run tests?
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_ising.py

# Check coverage
pytest --cov=historical_fidelity_simulator
```

## Research Questions

### What papers should I read?
1. Original fidelity papers
2. Quantum chaos literature
3. Phase transition reviews
4. Implementation details
See [References](references.md)

### How do I cite this work?
```bibtex
@software{historical_fidelity_simulator,
  title = {Historical Fidelity Simulator},
  year = {2025},
  url = {https://github.com/historical-fidelity-simulator}
}
```

### Where can I find examples?
1. Basic examples in `examples/`
2. Jupyter notebooks
3. Documentation tutorials
4. Research examples

### How do I extend the model?
1. Subclass base classes
2. Implement new Hamiltonians
3. Add custom measurements
4. Contribute back! 
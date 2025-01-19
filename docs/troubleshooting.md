# Troubleshooting Guide

This guide helps resolve common issues encountered when using the Historical Fidelity Simulator.

## Installation Issues

### GPU Support Not Working

**Symptoms:**
- CUDA errors when running GPU-accelerated simulations
- Unexpected fallback to CPU
- Poor performance on large systems

**Solutions:**
1. Verify CUDA installation:
   ```bash
   nvidia-smi  # Should show GPU info
   ```
2. Check CUDA version compatibility:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
3. Ensure GPU is visible:
   ```bash
   echo $CUDA_VISIBLE_DEVICES  # Should show available GPU indices
   ```

### Package Dependencies

**Symptoms:**
- Import errors
- Version conflicts
- Missing optional features

**Solutions:**
1. Install with all optional dependencies:
   ```bash
   pip install -e ".[all]"
   ```
2. Check dependency versions:
   ```bash
   pip list | grep -E "numpy|scipy|torch|matplotlib"
   ```
3. Create fresh environment:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install -e ".[all]"
   ```

## Runtime Issues

### Memory Errors

**Symptoms:**
- Out of memory errors
- System slowdown
- Unexpected crashes

**Solutions:**
1. Reduce system size
2. Enable GPU memory optimization:
   ```python
   simulator.config.optimize_memory = True
   ```
3. Use batch processing for large parameter sweeps
4. Monitor memory usage:
   ```python
   from historical_fidelity_simulator.utils import print_memory_usage
   print_memory_usage()
   ```

### Numerical Instability

**Symptoms:**
- NaN values in results
- Unexpected divergence
- Energy conservation violation

**Solutions:**
1. Check parameter ranges:
   - Temperature: T > 0
   - Coupling strength: |J| < 100
   - Field strength: |h| < 100
2. Reduce time step size
3. Enable stability checks:
   ```python
   simulator.config.check_stability = True
   ```
4. Use double precision:
   ```python
   simulator.config.dtype = np.float64
   ```

### Performance Issues

**Symptoms:**
- Slow execution
- High memory usage
- Poor scaling

**Solutions:**
1. Enable GPU acceleration:
   ```python
   simulator.use_gpu = True
   ```
2. Optimize batch size:
   ```python
   simulator.config.batch_size = 1000  # Adjust based on GPU memory
   ```
3. Use parallel processing:
   ```python
   simulator.config.n_workers = 4  # Adjust based on CPU cores
   ```
4. Profile code:
   ```python
   from historical_fidelity_simulator.utils import profile_execution
   with profile_execution():
       simulator.run()
   ```

## Analysis Issues

### Phase Transition Detection

**Symptoms:**
- Missing critical points
- Noisy phase diagrams
- Incorrect scaling behavior

**Solutions:**
1. Increase temperature resolution:
   ```python
   temps = np.linspace(1.0, 4.0, 100)  # More points near T_c
   ```
2. Use larger system sizes
3. Increase sampling:
   ```python
   simulator.config.n_samples = 10000
   ```
4. Enable finite-size scaling:
   ```python
   analysis.use_finite_size_scaling = True
   ```

### Fidelity Measurement

**Symptoms:**
- Unexpected fidelity values
- Poor bound satisfaction
- High fluctuations

**Solutions:**
1. Check normalization:
   ```python
   simulator.check_normalization()
   ```
2. Increase measurement frequency
3. Use error mitigation:
   ```python
   simulator.config.use_error_mitigation = True
   ```
4. Verify initial states:
   ```python
   simulator.verify_initial_state()
   ```

## Common Error Messages

### "CUDA out of memory"
1. Reduce system size
2. Clear GPU memory
3. Use CPU fallback
4. Enable memory optimization

### "Invalid parameter range"
1. Check parameter bounds
2. Verify units
3. Use parameter validation
4. Enable debug logging

### "Convergence failed"
1. Increase max iterations
2. Adjust convergence criteria
3. Check initial conditions
4. Use alternative solver

## Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](faq.md)
2. Search [GitHub Issues](https://github.com/historical-fidelity-simulator/issues)
3. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. Create a minimal example that reproduces the issue
5. Open a new issue with:
   - Python version
   - Package version
   - Error message
   - Minimal example
   - System information 
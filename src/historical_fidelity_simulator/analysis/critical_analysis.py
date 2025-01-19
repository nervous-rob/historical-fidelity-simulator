"""
Critical Analysis Tools for Historical Fidelity Simulator

This module provides tools for analyzing critical behavior and phase transitions,
with a focus on finite-size scaling and data collapse analysis.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from typing import Dict, List, Tuple, Optional
import numpy.typing as npt

def compute_correlation_length(
    fidelity_history: npt.NDArray[np.float64],
    lattice_spacing: float = 1.0
) -> float:
    """Compute correlation length from fidelity history.

    Args:
        fidelity_history: Time series of fidelity measurements
        lattice_spacing: Physical spacing between lattice sites

    Returns:
        float: Estimated correlation length
    """
    # Compute autocorrelation function
    mean = np.mean(fidelity_history)
    fluctuations = fidelity_history - mean
    autocorr = np.correlate(fluctuations, fluctuations, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Fit exponential decay to get correlation length
    x = np.arange(len(autocorr)) * lattice_spacing
    valid_range = len(x) // 3  # Use first third for fitting
    
    def exp_decay(x: npt.NDArray[np.float64], xi: float, a: float) -> npt.NDArray[np.float64]:
        return a * np.exp(-x / xi)
    
    try:
        popt, _ = curve_fit(exp_decay, x[:valid_range], 
                           autocorr[:valid_range] / autocorr[0],
                           p0=[1.0, 1.0])
        return abs(popt[0])  # Return correlation length
    except:
        return np.nan

def analyze_finite_size_scaling(
    temps: npt.NDArray[np.float64],
    fidelities: Dict[int, List[float]],
    t_c: float,
    size_range: Optional[Tuple[float, float]] = None
) -> Dict[str, float]:
    """Analyze finite-size scaling behavior near critical point.

    Args:
        temps: Array of temperatures
        fidelities: Dict mapping system sizes to fidelity lists
        t_c: Critical temperature
        size_range: Optional tuple of (min_size, max_size) for fitting

    Returns:
        Dict containing critical exponents and scaling parameters
    """
    reduced_t = (temps - t_c) / t_c
    sizes = np.array(list(fidelities.keys()))
    
    if size_range:
        min_size, max_size = size_range
        mask = (sizes >= min_size) & (sizes <= max_size)
        sizes = sizes[mask]
    
    # Compute scaling collapse
    def scaling_function(x: npt.NDArray[np.float64], nu: float, beta: float) -> npt.NDArray[np.float64]:
        return x * np.power(sizes[:, np.newaxis], beta/nu)
    
    def objective(params: Tuple[float, float]) -> float:
        nu, beta = params
        scaled_data = []
        for N in sizes:
            x = reduced_t * np.power(N, 1/nu)
            y = np.array(fidelities[N]) * np.power(N, beta/nu)
            scaled_data.extend(list(zip(x, y)))
        scaled_data = np.array(scaled_data)
        return np.std(scaled_data[:, 1])  # Minimize spread in collapsed data
    
    result = minimize(objective, x0=[1.0, 0.125], bounds=[(0.1, 10), (0.01, 1.0)])
    nu_fit, beta_fit = result.x
    
    return {
        'nu': nu_fit,
        'beta': beta_fit,
        'quality': -result.fun  # Higher is better
    }

def compute_susceptibility(
    fidelity_history: npt.NDArray[np.float64],
    temperature: float
) -> float:
    """Compute fidelity susceptibility.

    Args:
        fidelity_history: Time series of fidelity measurements
        temperature: System temperature

    Returns:
        float: Computed susceptibility
    """
    variance = np.var(fidelity_history)
    return variance / temperature  # Fluctuation-dissipation relation 
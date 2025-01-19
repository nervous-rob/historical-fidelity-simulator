"""Analysis tools for the Historical Fidelity Simulator."""

from .critical_analysis import (
    compute_correlation_length,
    analyze_finite_size_scaling,
    compute_susceptibility
)

__all__ = [
    'compute_correlation_length',
    'analyze_finite_size_scaling',
    'compute_susceptibility'
] 
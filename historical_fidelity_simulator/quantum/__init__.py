"""Quantum module for the historical fidelity simulator.

This module provides quantum-specific implementations for the simulator,
including Hamiltonian construction, Lindblad operators, and state evolution.
"""

from .operators import (
    construct_ising_hamiltonian,
    construct_lindblad_operators,
    compute_observables
)
from .evolution import QuantumEvolver

__all__ = [
    'construct_ising_hamiltonian',
    'construct_lindblad_operators',
    'compute_observables',
    'QuantumEvolver'
] 
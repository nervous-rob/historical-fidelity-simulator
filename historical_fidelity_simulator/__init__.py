"""Historical Fidelity Simulator package.

A unified simulator for studying historical fidelity through both classical and quantum approaches,
investigating the concept of an information Planck constant (ℏ_h) and its generalized uncertainty relation.
"""

__version__ = "0.1.0"

from .simulator import GeneralizedHistoricalSimulator, SimulationMode

__all__ = ["GeneralizedHistoricalSimulator", "SimulationMode"] 
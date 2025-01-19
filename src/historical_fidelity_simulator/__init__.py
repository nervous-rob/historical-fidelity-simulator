"""Historical Fidelity Simulator Package

A package for simulating and analyzing historical fidelity in both classical and quantum systems.
"""

from .simulator import GeneralizedHistoricalSimulator, SimulationMode
from . import analysis

__all__ = [
    'GeneralizedHistoricalSimulator',
    'SimulationMode',
    'analysis'
]

__version__ = '0.1.0' 
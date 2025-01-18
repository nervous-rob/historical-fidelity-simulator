"""Classical simulation components.

This package contains classical physics implementations including:
- Ising model with nearest-neighbor interactions
- Metropolis-Hastings dynamics
- Local energy calculations
- Classical state evolution
"""

from .ising import IsingModel
from .dynamics import MetropolisDynamics

__all__ = ['IsingModel', 'MetropolisDynamics'] 
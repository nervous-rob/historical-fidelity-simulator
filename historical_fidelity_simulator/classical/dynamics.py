"""Classical dynamics implementation.

This module implements Metropolis-Hastings dynamics for the classical Ising model.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .ising import IsingModel


class MetropolisDynamics:
    """Metropolis-Hastings dynamics for classical Ising model."""
    
    def __init__(
        self,
        model: IsingModel,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the dynamics.
        
        Args:
            model: IsingModel instance to evolve
            random_seed: Optional seed for random number generator
        """
        self.model = model
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.time = 0.0
        self.history: List[Dict] = []
        
    def step(self, dt: float, n_flips: Optional[int] = None) -> Dict:
        """Perform one time step of evolution.
        
        Args:
            dt: Time step size
            n_flips: Number of spin flips to attempt (default: n_sites)
            
        Returns:
            Dictionary with step results
        """
        n_flips = n_flips or self.model.n_sites
        total_de = 0.0
        accepted = 0
        
        for _ in range(n_flips):
            site = np.random.randint(0, self.model.n_sites)
            de, was_accepted = self.model.flip_spin(site)
            total_de += de
            accepted += int(was_accepted)
        
        self.time += dt
        
        # Record step results
        results = {
            'time': self.time,
            'energy': self.model.total_energy(),
            'magnetization': self.model.magnetization(),
            'energy_change': total_de,
            'acceptance_rate': accepted / n_flips
        }
        self.history.append(results)
        return results
    
    def run(
        self,
        total_time: float,
        dt: float,
        measure_interval: int = 1
    ) -> List[Dict]:
        """Run dynamics for specified time.
        
        Args:
            total_time: Total simulation time
            dt: Time step size
            measure_interval: Number of steps between measurements
            
        Returns:
            List of measurement dictionaries
        """
        n_steps = int(total_time / dt)
        results = []
        
        for step in range(n_steps):
            step_result = self.step(dt)
            if step % measure_interval == 0:
                results.append(step_result)
        
        return results
    
    def compute_fidelity(self) -> float:
        """Compute current historical fidelity.
        
        Returns:
            Historical fidelity metric based on energy
        """
        # Historical fidelity defined as negative energy per site
        return -self.model.total_energy() / self.model.n_sites
    
    def compute_generalized_bound(
        self,
        hbar_h: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        t_c: Optional[float] = None
    ) -> float:
        """Compute generalized uncertainty bound.
        
        Args:
            hbar_h: Information Planck constant
            alpha: Scaling prefactor
            beta: Critical exponent
            t_c: Critical temperature (default: current T)
            
        Returns:
            Value of generalized bound
        """
        t_c = t_c or self.model.T
        avg_energy = abs(self.compute_fidelity())
        
        # Compute f(E,T) = 1 + E/T + α(T/Tc)^β
        f_scale = 1.0 + avg_energy / self.model.T
        if t_c > 0:
            f_scale += alpha * (self.model.T / t_c) ** beta
            
        return hbar_h * f_scale 
"""Classical Ising model implementation.

This module implements a classical Ising model with nearest-neighbor interactions
for historical fidelity simulation.
"""

from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt


class IsingModel:
    """Classical Ising model with nearest-neighbor interactions."""
    
    def __init__(
        self,
        n_sites: int,
        coupling_strength: float,
        field_strength: float,
        temperature: float,
        periodic: bool = True
    ) -> None:
        """Initialize the Ising model.
        
        Args:
            n_sites: Number of sites in the system
            coupling_strength: J parameter for neighbor interactions
            field_strength: h parameter for external field
            temperature: System temperature T
            periodic: Whether to use periodic boundary conditions
        """
        self.n_sites = n_sites
        self.J = coupling_strength
        self.h = field_strength
        self.T = temperature
        self.beta = 1.0 / temperature
        self.periodic = periodic
        
        # Initialize random spin configuration
        self.state = np.random.choice([-1, 1], size=n_sites)
        
    def get_neighbors(self, site: int) -> List[int]:
        """Get indices of neighboring sites.
        
        Args:
            site: Index of the site
            
        Returns:
            List of neighbor indices, ordered [left_neighbor, right_neighbor]
        """
        neighbors = []
        # Left neighbor
        if site > 0:
            neighbors.append(site - 1)
        elif self.periodic:
            neighbors.append(self.n_sites - 1)
            
        # Right neighbor
        if site < self.n_sites - 1:
            neighbors.append(site + 1)
        elif self.periodic:
            neighbors.append(0)
            
        return neighbors
    
    def local_energy(self, site: int) -> float:
        """Compute local energy contribution from a site.
        
        Args:
            site: Index of the site
            
        Returns:
            Local energy contribution
        """
        neighbors = self.get_neighbors(site)
        # Interaction term
        e_interaction = -self.J * sum(
            self.state[site] * self.state[n] for n in neighbors
        )
        # Field term
        e_field = -self.h * self.state[site]
        return e_interaction + e_field
    
    def total_energy(self) -> float:
        """Compute total energy of the system.
        
        Returns:
            Total system energy
        """
        return sum(self.local_energy(i) for i in range(self.n_sites))
    
    def magnetization(self) -> float:
        """Compute system magnetization (order parameter).
        
        Returns:
            Average magnetization per site
        """
        return np.mean(self.state)
    
    def flip_spin(self, site: int) -> Tuple[float, bool]:
        """Attempt to flip a spin using Metropolis algorithm.
        
        Args:
            site: Index of the site to flip
            
        Returns:
            Tuple of (energy change, whether flip was accepted)
        """
        old_energy = self.local_energy(site)
        self.state[site] *= -1  # Flip
        new_energy = self.local_energy(site)
        delta_e = new_energy - old_energy
        
        # Metropolis acceptance criterion
        # At T=0, only accept if energy decreases
        if self.T == 0.0:
            accept = delta_e < 0
        else:
            accept = delta_e <= 0 or np.random.random() < np.exp(-self.beta * delta_e)
            
        if not accept:
            self.state[site] *= -1  # Flip back
            return 0.0, False
            
        return delta_e, True
    
    def get_state(self) -> npt.NDArray[np.int8]:
        """Get current spin configuration.
        
        Returns:
            Array of spin values
        """
        return self.state.copy() 
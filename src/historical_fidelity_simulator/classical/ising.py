"""Classical Ising model implementation.

This module implements a classical Ising model with nearest-neighbor interactions
for historical fidelity simulation. The model supports both periodic and open
boundary conditions, with configurable coupling strength (J) and external field (h).

Example:
    ```python
    # Create a 10-site Ising model at T=2.27 (critical temperature)
    model = IsingModel(
        n_sites=10,
        coupling_strength=1.0,
        field_strength=0.0,
        temperature=2.27
    )
    
    # Run Metropolis updates
    for _ in range(1000):
        site = np.random.randint(10)
        model.flip_spin(site)
    
    # Measure observables
    energy = model.total_energy()
    mag = model.magnetization()
    ```
"""

from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt


class IsingModel:
    """Classical Ising model with nearest-neighbor interactions.
    
    The Hamiltonian is given by:
    H = -J ∑_<i,j> s_i s_j - h ∑_i s_i
    
    where J is the coupling strength, h is the external field,
    s_i = ±1 are the spin values, and <i,j> denotes nearest neighbors.
    
    Example:
        ```python
        # Initialize model in paramagnetic phase
        model = IsingModel(
            n_sites=100,
            coupling_strength=1.0,
            field_strength=0.0,
            temperature=5.0
        )
        
        # Measure magnetization
        mag = model.magnetization()  # Should be close to 0
        
        # Initialize model in ferromagnetic phase
        model_fm = IsingModel(
            n_sites=100,
            coupling_strength=1.0,
            field_strength=0.0,
            temperature=1.0
        )
        
        # Measure magnetization
        mag_fm = model_fm.magnetization()  # Should be close to ±1
        ```
    """
    
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
            
        Example:
            ```python
            # Create 1D Ising chain with open boundaries
            model = IsingModel(
                n_sites=50,
                coupling_strength=1.0,
                field_strength=0.5,
                temperature=2.0,
                periodic=False
            )
            ```
        """
        # Store model parameters
        self.n_sites = n_sites
        self.J = coupling_strength
        self.h = field_strength
        self.T = temperature
        self.beta = 1.0 / temperature  # Inverse temperature for Metropolis
        self.periodic = periodic
        
        # Initialize random spin configuration
        # Each spin is ±1 with equal probability
        self.state = np.random.choice([-1, 1], size=n_sites)
        
    def get_neighbors(self, site: int) -> List[int]:
        """Get indices of neighboring sites.
        
        For periodic boundaries, sites 0 and N-1 are neighbors.
        For open boundaries, edge sites have only one neighbor.
        
        Args:
            site: Index of the site
            
        Returns:
            List of neighbor indices, ordered [left_neighbor, right_neighbor]
            
        Example:
            ```python
            model = IsingModel(n_sites=5, coupling_strength=1.0,
                             field_strength=0.0, temperature=1.0)
            
            # Middle site neighbors
            neighbors = model.get_neighbors(2)  # Returns [1, 3]
            
            # Edge site with periodic boundaries
            neighbors = model.get_neighbors(0)  # Returns [4, 1]
            
            # Edge site without periodic boundaries
            model.periodic = False
            neighbors = model.get_neighbors(0)  # Returns [1]
            ```
        """
        neighbors = []
        
        # Handle left neighbor
        if site > 0:
            neighbors.append(site - 1)
        elif self.periodic:
            neighbors.append(self.n_sites - 1)
            
        # Handle right neighbor
        if site < self.n_sites - 1:
            neighbors.append(site + 1)
        elif self.periodic:
            neighbors.append(0)
            
        return neighbors
    
    def local_energy(self, site: int) -> float:
        """Compute local energy contribution from a site.
        
        The local energy includes:
        1. Interaction terms with neighbors (-J s_i s_j)
        2. External field term (-h s_i)
        
        Args:
            site: Index of the site
            
        Returns:
            Local energy contribution
            
        Example:
            ```python
            model = IsingModel(n_sites=10, coupling_strength=1.0,
                             field_strength=0.5, temperature=1.0)
            
            # Energy of middle site
            e_local = model.local_energy(5)
            
            # Compare with total energy
            e_total = model.total_energy()
            # e_total ≈ sum of local energies / 2 (to avoid double counting)
            ```
        """
        # Get neighboring sites
        neighbors = self.get_neighbors(site)
        
        # Calculate interaction energy with neighbors
        # E_int = -J ∑_j s_i s_j where j are neighbors
        e_interaction = -self.J * sum(
            self.state[site] * self.state[n] for n in neighbors
        )
        
        # Calculate field energy
        # E_field = -h s_i
        e_field = -self.h * self.state[site]
        
        return e_interaction + e_field
    
    def total_energy(self) -> float:
        """Compute total energy of the system.
        
        The total energy is the sum of all local energies,
        divided by 2 to avoid double counting bonds.
        
        Returns:
            Total system energy
            
        Example:
            ```python
            model = IsingModel(n_sites=10, coupling_strength=1.0,
                             field_strength=0.0, temperature=1.0)
            
            # Ground state energy (all spins aligned)
            # Should be close to -10 (N_bonds * J = 9 for periodic)
            model.state = np.ones(10)
            e_ground = model.total_energy()
            
            # Random state energy
            model.state = np.random.choice([-1, 1], size=10)
            e_random = model.total_energy()
            # e_random > e_ground (ground state minimizes energy)
            ```
        """
        return sum(self.local_energy(i) for i in range(self.n_sites)) / 2.0
    
    def magnetization(self) -> float:
        """Compute system magnetization (order parameter).
        
        The magnetization is the average spin value:
        m = (1/N) ∑_i s_i
        
        Returns:
            Average magnetization per site
            
        Example:
            ```python
            model = IsingModel(n_sites=1000, coupling_strength=1.0,
                             field_strength=0.0, temperature=1.0)
            
            # Low temperature: expect |m| ≈ 1 (spontaneous magnetization)
            mag_low_t = model.magnetization()
            
            # High temperature: expect m ≈ 0 (paramagnetic)
            model.T = 5.0
            model.beta = 0.2
            for _ in range(10000):  # Thermalize
                site = np.random.randint(1000)
                model.flip_spin(site)
            mag_high_t = model.magnetization()
            ```
        """
        return np.mean(self.state)
    
    def flip_spin(self, site: int) -> Tuple[float, bool]:
        """Attempt to flip a spin using Metropolis algorithm.
        
        The Metropolis acceptance probability is:
        P(accept) = min(1, exp(-β ΔE))
        
        Args:
            site: Index of the site to flip
            
        Returns:
            Tuple of (energy change, whether flip was accepted)
            
        Example:
            ```python
            model = IsingModel(n_sites=100, coupling_strength=1.0,
                             field_strength=0.0, temperature=2.27)
            
            # Metropolis update
            n_accepted = 0
            n_steps = 1000
            
            for _ in range(n_steps):
                site = np.random.randint(100)
                delta_e, accepted = model.flip_spin(site)
                if accepted:
                    n_accepted += 1
            
            acceptance_rate = n_accepted / n_steps
            # Should be around 0.5 at critical temperature
            ```
        """
        # Store old energy before flip
        old_energy = self.local_energy(site)
        
        # Flip the spin: s_i -> -s_i
        self.state[site] *= -1
        
        # Calculate new energy and energy change
        new_energy = self.local_energy(site)
        delta_e = new_energy - old_energy
        
        # Metropolis acceptance criterion
        # Special case: At T=0, only accept if energy decreases
        if self.T == 0.0:
            accept = delta_e < 0
        else:
            # Accept with probability min(1, exp(-β ΔE))
            # Always accept if ΔE ≤ 0 (energy decreases)
            # Accept with probability exp(-β ΔE) if ΔE > 0
            accept = delta_e <= 0 or np.random.random() < np.exp(-self.beta * delta_e)
            
        # If rejected, restore old state
        if not accept:
            self.state[site] *= -1  # Flip back
            return 0.0, False
            
        return delta_e, True
    
    def get_state(self) -> npt.NDArray[np.int8]:
        """Get current spin configuration.
        
        Returns:
            Array of spin values
            
        Example:
            ```python
            model = IsingModel(n_sites=10, coupling_strength=1.0,
                             field_strength=0.0, temperature=1.0)
            
            # Get current state
            state = model.get_state()
            
            # Verify it's a copy (modifying doesn't affect model)
            state[0] *= -1
            assert np.any(state != model.state)
            ```
        """
        return self.state.copy() 
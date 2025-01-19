"""GPU acceleration utilities for the historical fidelity simulator.

This module provides GPU-accelerated versions of core computations using:
1. Numba CUDA for classical simulations
2. CuPy for quantum matrix operations
"""

import numpy as np
from numba import cuda, float32, int32, jit
import math
from typing import Optional, Union
import warnings

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def matrix_exp_gpu(A: Union[np.ndarray, "cp.ndarray"], order: int = 6) -> "cp.ndarray":
    """Compute matrix exponential using Padé approximation.
    
    Args:
        A: Input matrix (CuPy array)
        order: Order of Padé approximation (higher = more accurate but slower)
        
    Returns:
        Matrix exponential as a CuPy array
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is required for GPU matrix operations")
    
    if not isinstance(A, cp.ndarray):
        A = cp.asarray(A)
    
    # Identity matrix
    n = A.shape[0]
    I = cp.eye(n, dtype=A.dtype)
    
    # Compute powers of A
    A_powers = [I]  # A^0
    A_current = I
    for _ in range(order):
        A_current = cp.matmul(A_current, A)
        A_powers.append(A_current)
    
    # Padé coefficients for the given order
    if order == 4:
        p_coeffs = [1, 1/2, 1/12, 1/120]
        q_coeffs = [1, -1/2, 1/12, -1/120]
    elif order == 6:
        p_coeffs = [1, 1/2, 1/10, 1/72, 1/720, 1/7200]
        q_coeffs = [1, -1/2, 1/10, -1/72, 1/720, -1/7200]
    else:
        raise ValueError("Only orders 4 and 6 are supported")
    
    # Compute numerator and denominator
    P = cp.zeros_like(I)
    Q = cp.zeros_like(I)
    for i in range(len(p_coeffs)):
        P += p_coeffs[i] * A_powers[i]
        Q += q_coeffs[i] * A_powers[i]
    
    # Solve the system (Q)⋅R = P
    return cp.linalg.solve(Q, P)

# CUDA kernel for computing local energies in parallel
@cuda.jit
def compute_local_energies_kernel(spins, J, h, energies, n_sites):
    """CUDA kernel for parallel computation of local energies."""
    idx = cuda.grid(1)
    if idx < n_sites:
        # Compute neighbors with periodic boundary conditions
        left = (idx - 1) % n_sites
        right = (idx + 1) % n_sites
        
        # Compute local energy contribution
        spin = spins[idx]
        neighbor_sum = spins[left] + spins[right]
        energies[idx] = -J * spin * neighbor_sum - h * spin

@cuda.jit
def metropolis_update_kernel(spins, energies, J, h, T, rand_nums, accepted, n_sites):
    """CUDA kernel for parallel Metropolis updates."""
    idx = cuda.grid(1)
    if idx < n_sites:
        # Propose spin flip
        old_energy = energies[idx]
        spins[idx] *= -1
        
        # Recompute local energy
        left = (idx - 1) % n_sites
        right = (idx + 1) % n_sites
        spin = spins[idx]
        neighbor_sum = spins[left] + spins[right]
        new_energy = -J * spin * neighbor_sum - h * spin
        
        # Metropolis acceptance
        delta_E = new_energy - old_energy
        if delta_E > 0 and math.exp(-delta_E / T) < rand_nums[idx]:
            # Reject move
            spins[idx] *= -1
            accepted[idx] = 0
        else:
            # Accept move
            energies[idx] = new_energy
            accepted[idx] = 1

def get_gpu_device():
    """Get the GPU device if available."""
    try:
        device = cuda.get_current_device()
        return device
    except:
        return None

class GPUAccelerator:
    """Class for managing GPU-accelerated computations."""
    
    def __init__(self, n_sites: int, use_gpu: bool = True):
        """Initialize the GPU accelerator.
        
        Args:
            n_sites: Number of sites in the system
            use_gpu: Whether to use GPU acceleration
        """
        self.n_sites = n_sites
        self.device = get_gpu_device() if use_gpu else None
        self.use_gpu = use_gpu and self.device is not None
        
        if self.use_gpu:
            # Determine optimal CUDA grid dimensions
            self.threads_per_block = min(32, n_sites)
            self.blocks_per_grid = (n_sites + self.threads_per_block - 1) // self.threads_per_block
            
            # Allocate GPU memory for common arrays
            self.d_spins = cuda.device_array(n_sites, dtype=np.int32)
            self.d_energies = cuda.device_array(n_sites, dtype=np.float32)
            self.d_accepted = cuda.device_array(n_sites, dtype=np.int32)
    
    def compute_classical_energies(self, spins: np.ndarray, J: float, h: float) -> np.ndarray:
        """Compute local energies using GPU acceleration if available."""
        if not self.use_gpu:
            return self._compute_classical_energies_cpu(spins, J, h)
        
        # Copy spins to GPU
        cuda.to_device(spins, to=self.d_spins)
        
        # Launch kernel
        compute_local_energies_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_spins, J, h, self.d_energies, self.n_sites
        )
        
        # Copy results back to CPU
        energies = np.empty(self.n_sites, dtype=np.float32)
        self.d_energies.copy_to_host(energies)
        return energies
    
    def metropolis_update(self, spins: np.ndarray, energies: np.ndarray, J: float, h: float, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Metropolis updates using GPU acceleration if available."""
        if not self.use_gpu:
            return self._metropolis_update_cpu(spins, energies, J, h, T)
        
        # Generate random numbers on CPU (numpy's random is better than CUDA's)
        rand_nums = np.random.random(self.n_sites).astype(np.float32)
        d_rand_nums = cuda.to_device(rand_nums)
        
        # Copy data to GPU
        cuda.to_device(spins, to=self.d_spins)
        cuda.to_device(energies, to=self.d_energies)
        
        # Launch kernel
        metropolis_update_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_spins, self.d_energies, J, h, T, d_rand_nums, self.d_accepted, self.n_sites
        )
        
        # Copy results back to CPU
        updated_spins = np.empty(self.n_sites, dtype=np.int32)
        updated_energies = np.empty(self.n_sites, dtype=np.float32)
        accepted = np.empty(self.n_sites, dtype=np.int32)
        
        self.d_spins.copy_to_host(updated_spins)
        self.d_energies.copy_to_host(updated_energies)
        self.d_accepted.copy_to_host(accepted)
        
        return updated_spins, updated_energies, accepted
    
    def _compute_classical_energies_cpu(self, spins: np.ndarray, J: float, h: float) -> np.ndarray:
        """CPU fallback for computing classical energies."""
        energies = np.zeros(self.n_sites, dtype=np.float32)
        for i in range(self.n_sites):
            left = (i - 1) % self.n_sites
            right = (i + 1) % self.n_sites
            spin = spins[i]
            neighbor_sum = spins[left] + spins[right]
            energies[i] = -J * spin * neighbor_sum - h * spin
        return energies
    
    def _metropolis_update_cpu(self, spins: np.ndarray, energies: np.ndarray, J: float, h: float, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU fallback for Metropolis updates."""
        accepted = np.zeros(self.n_sites, dtype=np.int32)
        for i in range(self.n_sites):
            old_energy = energies[i]
            spins[i] *= -1
            
            left = (i - 1) % self.n_sites
            right = (i + 1) % self.n_sites
            spin = spins[i]
            neighbor_sum = spins[left] + spins[right]
            new_energy = -J * spin * neighbor_sum - h * spin
            
            delta_E = new_energy - old_energy
            if delta_E > 0 and np.exp(-delta_E / T) < np.random.random():
                spins[i] *= -1
                accepted[i] = 0
            else:
                energies[i] = new_energy
                accepted[i] = 1
        
        return spins, energies, accepted 
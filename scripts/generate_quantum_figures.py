"""Script to generate quantum-specific figures for the documentation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import qutip as qt
from historical_fidelity_simulator.quantum.operators import (
    construct_ising_hamiltonian,
    construct_lindblad_operators,
    compute_observables
)
from historical_fidelity_simulator.quantum.evolution import QuantumEvolver

# Create figures directory if it doesn't exist
FIGURE_DIR = Path('docs/theory/figures')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def plot_quantum_magnetization():
    """Plot quantum magnetization vs field strength."""
    n_sites = 6  # Small system for quantum simulation
    coupling_strength = 1.0
    field_strengths = np.linspace(0, 2, 20)
    temperatures = [0.1, 0.5, 1.0]
    
    plt.figure(figsize=(10, 6))
    
    for T in temperatures:
        mags = []
        for h in field_strengths:
            # Construct Hamiltonian
            H = construct_ising_hamiltonian(n_sites, coupling_strength, h)
            
            # Get eigenstates and energies
            eigvals, eigvecs = H.eigenstates()
            
            if T > 0.1:  # For finite temperature
                # Compute partition function and thermal state
                Z = sum(np.exp(-E/T) for E in eigvals)
                rho = sum(np.exp(-E/T) * v * v.dag() for E, v in zip(eigvals, eigvecs)) / Z
            else:  # For near-zero temperature
                # Just use ground state
                rho = eigvecs[0] * eigvecs[0].dag()
            
            # Compute magnetization
            mag, _ = compute_observables(rho, n_sites)
            mags.append(mag)
        
        plt.plot(field_strengths, mags, label=f'T = {T:.1f}J/k_B')
    
    plt.xlabel('Field Strength h/J')
    plt.ylabel('Magnetization ⟨σ_z⟩')
    plt.title('Quantum Magnetization vs Field')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'quantum_magnetization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_entanglement_entropy():
    """Plot entanglement entropy vs field strength."""
    n_sites = 6
    coupling_strength = 1.0
    field_strengths = np.linspace(0, 2, 20)
    
    plt.figure(figsize=(10, 6))
    
    entropies = []
    for h in field_strengths:
        # Construct Hamiltonian
        H = construct_ising_hamiltonian(n_sites, coupling_strength, h)
        
        # Get ground state
        ground_state = H.groundstate()[1]
        
        # Compute entanglement entropy
        _, entropy = compute_observables(ground_state, n_sites)
        entropies.append(entropy)
    
    plt.plot(field_strengths, entropies, 'b-')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Critical Point')
    
    plt.xlabel('Field Strength h/J')
    plt.ylabel('Entanglement Entropy S')
    plt.title('Ground State Entanglement Entropy')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'entanglement_entropy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_quantum_dynamics():
    """Plot quantum state evolution."""
    n_sites = 4
    coupling_strength = 1.0
    field_strength = 0.5
    temperature = 0.1
    
    # Construct Hamiltonian and evolver
    H = construct_ising_hamiltonian(n_sites, coupling_strength, field_strength)
    evolver = QuantumEvolver(H, n_sites, temperature)
    
    # Initial state: all spins up
    psi0 = qt.basis([2]*n_sites, [0]*n_sites)
    
    # Evolve for different times
    times = np.linspace(0, 5, 50)
    dt = times[1] - times[0]  # Use proper time step
    mags = []
    fids = []
    
    state = psi0
    for t in times[1:]:  # Skip t=0 since we already have initial state
        state, _ = evolver.evolve_state(state, dt)
        mag, _ = compute_observables(state, n_sites)
        fid = evolver.compute_fidelity(state)
        mags.append(mag)
        fids.append(fid)
    
    # Add initial point
    mag0, _ = compute_observables(psi0, n_sites)
    fid0 = evolver.compute_fidelity(psi0)
    mags.insert(0, mag0)
    fids.insert(0, fid0)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, mags, 'b-', label='Magnetization')
    plt.ylabel('⟨σ_z⟩')
    plt.title('Quantum Dynamics')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(times, fids, 'r-', label='Fidelity')
    plt.xlabel('Time t (ℏ/J)')
    plt.ylabel('F(t)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'quantum_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_decoherence_effects():
    """Plot effects of decoherence on quantum evolution."""
    n_sites = 4
    coupling_strength = 1.0
    field_strength = 0.5
    temperature = 0.5
    
    # Construct Hamiltonian
    H = construct_ising_hamiltonian(n_sites, coupling_strength, field_strength)
    
    # Different decoherence strengths
    decoherence_strengths = [0.0, 0.1, 0.5]
    times = np.linspace(0, 5, 50)
    dt = times[1] - times[0]  # Use proper time step
    
    plt.figure(figsize=(10, 6))
    
    # Initial state: all spins up
    psi0 = qt.basis([2]*n_sites, [0]*n_sites)
    
    for gamma in decoherence_strengths:
        evolver = QuantumEvolver(H, n_sites, temperature, decoherence_strength=gamma)
        state = psi0
        mags = [compute_observables(state, n_sites)[0]]  # Initial magnetization
        
        # Evolve state step by step
        for _ in range(len(times)-1):
            state, _ = evolver.evolve_state(state, dt)
            mag, _ = compute_observables(state, n_sites)
            mags.append(mag)
        
        plt.plot(times, mags, label=f'γ = {gamma:.1f}')
    
    plt.xlabel('Time t (ℏ/J)')
    plt.ylabel('Magnetization ⟨σ_z⟩')
    plt.title('Decoherence Effects')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'decoherence_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_energy_spectrum():
    """Plot energy spectrum vs field strength."""
    n_sites = 4  # Small system for clear spectrum
    coupling_strength = 1.0
    field_strengths = np.linspace(0, 2, 50)
    
    plt.figure(figsize=(10, 6))
    
    # Store energies for each field strength
    all_energies = []
    for h in field_strengths:
        H = construct_ising_hamiltonian(n_sites, coupling_strength, h)
        eigvals = H.eigenenergies()
        all_energies.append(eigvals)
    
    # Plot each energy level
    all_energies = np.array(all_energies)
    for i in range(len(all_energies[0])):
        plt.plot(field_strengths, all_energies[:, i] / n_sites, 'b-', alpha=0.5)
    
    plt.axvline(x=1.0, color='r', linestyle='--', label='Critical Point')
    plt.xlabel('Field Strength h/J')
    plt.ylabel('Energy per Site E/N')
    plt.title('Energy Spectrum')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'energy_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_quantum_correlations():
    """Plot quantum correlation functions."""
    n_sites = 4
    coupling_strength = 1.0
    field_strengths = [0.1, 1.0, 2.0]  # Different phases
    
    plt.figure(figsize=(10, 6))
    
    for h in field_strengths:
        # Construct Hamiltonian and get ground state
        H = construct_ising_hamiltonian(n_sites, coupling_strength, h)
        ground_state = H.groundstate()[1]
        
        # Compute correlations <σ_i^z σ_j^z>
        correlations = []
        for r in range(n_sites//2):  # Up to half system size
            # Create operators for correlation measurement
            op_i = [qt.qeye(2)] * n_sites
            op_j = [qt.qeye(2)] * n_sites
            op_i[0] = qt.sigmaz()
            op_j[r] = qt.sigmaz()
            
            # Compute expectation value
            corr = qt.expect(qt.tensor(op_i), ground_state) * \
                   qt.expect(qt.tensor(op_j), ground_state)
            correlations.append(corr)
        
        plt.plot(range(n_sites//2), correlations, 'o-', label=f'h/J = {h:.1f}')
    
    plt.xlabel('Distance r')
    plt.ylabel('Correlation ⟨σ_0^z σ_r^z⟩')
    plt.title('Quantum Correlations in Ground State')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'quantum_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_quench_dynamics():
    """Plot dynamics after a quantum quench."""
    n_sites = 4
    coupling_strength = 1.0
    initial_field = 0.1
    final_field = 2.0
    temperature = 0.1
    
    # Prepare initial state (ground state of initial Hamiltonian)
    H_initial = construct_ising_hamiltonian(n_sites, coupling_strength, initial_field)
    initial_state = H_initial.groundstate()[1]
    
    # Evolve under final Hamiltonian
    H_final = construct_ising_hamiltonian(n_sites, coupling_strength, final_field)
    evolver = QuantumEvolver(H_final, n_sites, temperature)
    
    # Time evolution
    times = np.linspace(0, 10, 100)
    dt = times[1] - times[0]
    
    # Observables to track
    mags = [compute_observables(initial_state, n_sites)[0]]
    energies = [qt.expect(H_final, initial_state) / n_sites]
    
    state = initial_state
    for _ in range(len(times)-1):
        state, _ = evolver.evolve_state(state, dt)
        mag, _ = compute_observables(state, n_sites)
        energy = qt.expect(H_final, state) / n_sites
        mags.append(mag)
        energies.append(energy)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, mags, 'b-', label='Magnetization')
    plt.ylabel('⟨σ_z⟩')
    plt.title('Quantum Quench Dynamics')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(times, energies, 'r-', label='Energy')
    plt.axhline(y=H_final.groundstate()[0]/n_sites, color='k', linestyle='--',
                label='Ground State Energy')
    plt.xlabel('Time t (ℏ/J)')
    plt.ylabel('Energy per Site E/N')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'quench_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_relation():
    """Plot energy-time uncertainty relation."""
    n_sites = 4
    coupling_strength = 1.0
    field_strength = 1.0  # At critical point
    temperature = 0.1
    
    # Prepare system
    H = construct_ising_hamiltonian(n_sites, coupling_strength, field_strength)
    evolver = QuantumEvolver(H, n_sites, temperature)
    
    # Initial state: all spins up
    psi0 = qt.basis([2]*n_sites, [0]*n_sites)
    
    # Compute uncertainty product for different time intervals
    times = np.logspace(-2, 1, 20)
    uncertainty_products = []
    
    for dt in times:
        product = evolver.compute_uncertainty_product(psi0, dt)
        uncertainty_products.append(product)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(times, uncertainty_products, 'b-', label='ΔE Δt')
    plt.axhline(y=0.5, color='r', linestyle='--', label='ℏ/2')
    
    plt.xlabel('Time t (ℏ/J)')
    plt.ylabel('Uncertainty Product ΔE Δt/ℏ')
    plt.title('Energy-Time Uncertainty Relation')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'uncertainty_relation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all quantum figures."""
    print("Generating quantum magnetization plot...")
    plot_quantum_magnetization()
    
    print("Generating entanglement entropy plot...")
    plot_entanglement_entropy()
    
    print("Generating quantum dynamics plot...")
    plot_quantum_dynamics()
    
    print("Generating decoherence effects plot...")
    plot_decoherence_effects()
    
    print("Generating energy spectrum plot...")
    plot_energy_spectrum()
    
    print("Generating quantum correlations plot...")
    plot_quantum_correlations()
    
    print("Generating quench dynamics plot...")
    plot_quench_dynamics()
    
    print("Generating uncertainty relation plot...")
    plot_uncertainty_relation()
    
    print("Done! Quantum figures saved in", FIGURE_DIR)

if __name__ == "__main__":
    main() 
"""Generate figures for theory documentation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from historical_fidelity_simulator.classical import IsingModel

# Set up paths
FIGURE_DIR = Path("docs/theory/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def plot_phase_diagram():
    """Plot phase diagram showing magnetization vs temperature."""
    temps = np.linspace(1.0, 4.0, 50)
    sizes = [10, 20, 50, 100]
    plt.figure(figsize=(10, 6))
    
    for N in sizes:
        mags = []
        for T in temps:
            model = IsingModel(n_sites=N, coupling_strength=1.0,
                             field_strength=0.0, temperature=T)
            # Thermalize
            for _ in range(10000):
                site = np.random.randint(N)
                model.flip_spin(site)
            # Measure
            mags.append(abs(model.magnetization()))
        plt.plot(temps, mags, label=f'N = {N}')
    
    plt.axvline(x=2.27, color='k', linestyle='--', label='T_c')
    plt.xlabel('Temperature (J/k_B)')
    plt.ylabel('|Magnetization|')
    plt.title('Phase Diagram of Classical Ising Model')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fidelity_decay():
    """Plot fidelity decay for different perturbation strengths."""
    times = np.linspace(0, 10, 100)
    epsilons = [0.01, 0.05, 0.1]
    hbar = 1.0
    
    plt.figure(figsize=(10, 6))
    
    for eps in epsilons:
        # Combine short-time and long-time behavior
        F = np.where(
            times < 2/eps,  # Short time
            1 - (eps * times/hbar)**2,  # Quadratic decay
            0.1 * np.exp(-(times - 2/eps)/2)  # Exponential decay
        )
        plt.plot(times, F, label=f'ε = {eps}')
    
    plt.xlabel('Time (ℏ/J)')
    plt.ylabel('Fidelity F(t)')
    plt.title('Fidelity Decay Under Perturbation')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'fidelity_decay.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_bound():
    """Plot uncertainty relation bound on fidelity."""
    times = np.linspace(0, 5, 100)
    dH = 1.0  # Energy uncertainty
    hbar = 1.0
    
    plt.figure(figsize=(10, 6))
    
    # Exact fidelity (example)
    F_exact = np.cos(dH * times/hbar)**2
    
    # Bounds
    F_bound1 = np.exp(-(dH * times/hbar)**2)
    F_bound2 = np.cos(dH * times/hbar)**2
    
    plt.plot(times, F_exact, 'k-', label='Exact')
    plt.plot(times, F_bound1, 'r--', label='Gaussian bound')
    plt.plot(times, F_bound2, 'b:', label='Cosine bound')
    
    plt.xlabel('Time (ℏ/ΔE)')
    plt.ylabel('Fidelity F(t)')
    plt.title('Uncertainty Bounds on Fidelity')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'uncertainty_bound.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_spin_configurations():
    """Plot example spin configurations at different temperatures."""
    N = 50
    temps = [1.0, 2.27, 4.0]
    fig, axes = plt.subplots(len(temps), 1, figsize=(10, 8))
    
    for i, T in enumerate(temps):
        model = IsingModel(n_sites=N, coupling_strength=1.0,
                          field_strength=0.0, temperature=T)
        # Thermalize
        for _ in range(10000):
            site = np.random.randint(N)
            model.flip_spin(site)
        
        # Plot spins
        spins = model.get_state()
        axes[i].imshow(spins.reshape(1, -1), cmap='binary', aspect='auto')
        axes[i].set_title(f'T = {T:.2f} J/k_B')
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'spin_configurations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_function():
    """Plot spin-spin correlation function at different temperatures."""
    N = 100
    max_distance = 20
    temps = [1.0, 2.27, 4.0]
    plt.figure(figsize=(10, 6))
    
    for T in temps:
        model = IsingModel(n_sites=N, coupling_strength=1.0,
                          field_strength=0.0, temperature=T)
        # Thermalize
        for _ in range(20000):
            site = np.random.randint(N)
            model.flip_spin(site)
        
        # Compute correlation function
        correlations = []
        center = N // 2
        state = model.get_state()
        for r in range(max_distance):
            if center + r < N:
                corr = np.mean(state[center] * state[center + r])
                correlations.append(corr)
        
        plt.plot(range(len(correlations)), correlations, 
                marker='o', label=f'T = {T:.2f} J/k_B')
    
    plt.xlabel('Distance r')
    plt.ylabel('Correlation G(r)')
    plt.title('Spin-Spin Correlation Function')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(FIGURE_DIR / 'correlation_function.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_susceptibility():
    """Plot magnetic susceptibility vs temperature."""
    temps = np.linspace(1.0, 4.0, 40)
    sizes = [20, 50, 100]
    plt.figure(figsize=(10, 6))
    
    for N in sizes:
        chis = []
        for T in temps:
            model = IsingModel(n_sites=N, coupling_strength=1.0,
                             field_strength=0.0, temperature=T)
            # Thermalize
            for _ in range(10000):
                site = np.random.randint(N)
                model.flip_spin(site)
            
            # Measure magnetization fluctuations
            mags = []
            for _ in range(1000):
                site = np.random.randint(N)
                model.flip_spin(site)
                mags.append(model.magnetization())
            
            # Compute susceptibility
            chi = N * np.var(mags) / T
            chis.append(chi)
        
        plt.plot(temps, chis, label=f'N = {N}')
    
    plt.axvline(x=2.27, color='k', linestyle='--', label='T_c')
    plt.xlabel('Temperature (J/k_B)')
    plt.ylabel('Susceptibility χ')
    plt.title('Magnetic Susceptibility')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'susceptibility.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_energy_histogram():
    """Plot energy distribution at different temperatures."""
    N = 50
    temps = [1.0, 2.27, 4.0]
    n_samples = 10000
    plt.figure(figsize=(10, 6))
    
    for T in temps:
        model = IsingModel(n_sites=N, coupling_strength=1.0,
                          field_strength=0.0, temperature=T)
        energies = []
        
        # Sample energies
        for _ in range(n_samples):
            site = np.random.randint(N)
            model.flip_spin(site)
            if _ > 1000:  # Skip thermalization
                energies.append(model.total_energy())
        
        plt.hist(energies, bins=50, density=True, alpha=0.5,
                label=f'T = {T:.2f} J/k_B')
    
    plt.xlabel('Energy (J)')
    plt.ylabel('Probability Density')
    plt.title('Energy Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'energy_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_relaxation_dynamics():
    """Plot relaxation of magnetization from different initial states."""
    N = 100
    T = 2.27  # Critical temperature
    n_steps = 1000
    plt.figure(figsize=(10, 6))
    
    # Different initial states
    initial_states = {
        'All up': np.ones(N),
        'All down': -np.ones(N),
        'Random': np.random.choice([-1, 1], size=N),
        'Alternating': np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    }
    
    for name, init_state in initial_states.items():
        model = IsingModel(n_sites=N, coupling_strength=1.0,
                          field_strength=0.0, temperature=T)
        model.state = init_state.copy()
        
        # Track magnetization
        mags = [model.magnetization()]
        for _ in range(n_steps):
            site = np.random.randint(N)
            model.flip_spin(site)
            if _ % 10 == 0:  # Record every 10 steps
                mags.append(model.magnetization())
        
        plt.plot(range(0, n_steps + 1, 10), mags, label=name)
    
    plt.xlabel('Monte Carlo Steps / 10')
    plt.ylabel('Magnetization')
    plt.title('Relaxation Dynamics at T_c')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURE_DIR / 'relaxation_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures."""
    print("Generating phase diagram...")
    plot_phase_diagram()
    
    print("Generating fidelity decay plot...")
    plot_fidelity_decay()
    
    print("Generating uncertainty bound plot...")
    plot_uncertainty_bound()
    
    print("Generating spin configurations plot...")
    plot_spin_configurations()
    
    print("Generating correlation function plot...")
    plot_correlation_function()
    
    print("Generating susceptibility plot...")
    plot_susceptibility()
    
    print("Generating energy histogram...")
    plot_energy_histogram()
    
    print("Generating relaxation dynamics plot...")
    plot_relaxation_dynamics()
    
    print("Done! Figures saved in", FIGURE_DIR)

if __name__ == "__main__":
    main() 
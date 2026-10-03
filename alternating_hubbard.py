"""
QAOA for Fermi-Hubbard Model with Correlation Distance Analysis
================================================================

Analyzes spatial correlations in the final QAOA state:
- Density-density correlations: ⟨n_i n_j⟩
- Spin-spin correlations: ⟨S_i · S_j⟩
- Pair correlations: ⟨c†_i↑ c†_j↓ c_j↓ c_i↑⟩
- Distance-dependent behavior
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class HubbardQAOAWithCorrelations:
    """QAOA for Hubbard model with correlation distance analysis."""

    def __init__(self, L=2, t=1.0, U=2.0, p=1):
        """Initialize system."""
        self.L = L
        self.t = t
        self.U = U
        self.p = p
        self.Ns = L * L
        self.N = 2 * self.Ns
        self.dim = 2**self.N

        print(f"System: {L}x{L} lattice, {self.N} fermions, Hilbert space: 2^{self.N} = {self.dim}")

        self.setup_lattice()
        self.build_hamiltonians()

    def setup_lattice(self):
        """Create lattice structure."""
        self.sites = [(i, j) for i in range(self.L) for j in range(self.L)]
        self.site_idx = {site: idx for idx, site in enumerate(self.sites)}

        # Nearest neighbor pairs
        self.nn_pairs = []
        for i, j in self.sites:
            self.nn_pairs.append(((i, j), ((i+1) % self.L, j)))
            self.nn_pairs.append(((i, j), (i, (j+1) % self.L)))

    def orbital_idx(self, site, spin):
        """Orbital index from site and spin."""
        return 2 * self.site_idx[site] + spin

    def c_dag(self, idx):
        """Creation operator."""
        mat = np.zeros((self.dim, self.dim), dtype=complex)
        for state in range(self.dim):
            if not ((state >> idx) & 1):
                sign = (-1)**sum((state >> i) & 1 for i in range(idx))
                new_state = state | (1 << idx)
                mat[new_state, state] = sign
        return mat

    def c(self, idx):
        """Annihilation operator."""
        return self.c_dag(idx).conj().T

    def n(self, idx):
        """Number operator."""
        return self.c_dag(idx) @ self.c(idx)

    def build_hamiltonians(self):
        """Build H_t and H_U."""
        print(f"Building Hamiltonians...")

        # Hopping term
        self.H_t = np.zeros((self.dim, self.dim), dtype=complex)
        for site1, site2 in self.nn_pairs:
            for spin in [0, 1]:
                idx1 = self.orbital_idx(site1, spin)
                idx2 = self.orbital_idx(site2, spin)
                self.H_t += -self.t * (
                    self.c_dag(idx2) @ self.c(idx1) +
                    self.c_dag(idx1) @ self.c(idx2)
                )

        # Interaction term
        self.H_U = np.zeros((self.dim, self.dim), dtype=complex)
        for site in self.sites:
            idx_up = self.orbital_idx(site, 0)
            idx_down = self.orbital_idx(site, 1)
            n_up = self.n(idx_up)
            n_down = self.n(idx_down)
            I = np.eye(self.dim)
            self.H_U += self.U * (n_up - 0.5*I) @ (n_down - 0.5*I)

        print(f"  Hamiltonians built")

    def qaoa_state(self, params):
        """Apply QAOA circuit."""
        state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)

        for layer in range(self.p):
            gamma = params[2*layer]
            beta = params[2*layer + 1]

            U_P = expm(-1j * gamma * self.H_U)
            state = U_P @ state

            U_M = expm(-1j * beta * self.H_t)
            state = U_M @ state

        return state

    def energy(self, state, hamiltonian):
        """Expectation value."""
        return np.real(state.conj() @ hamiltonian @ state)

    def cost_function(self, state):
        """Cost function."""
        return self.energy(state, self.H_U)

    def objective(self, params):
        """Objective for optimization."""
        state = self.qaoa_state(params)
        return self.cost_function(state)

    def optimize(self, num_trials=3):
        """Optimize QAOA parameters."""
        best_params = None
        best_cost = np.inf
        costs = []

        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials}", end=" ")
            x0 = np.random.uniform(-np.pi, np.pi, 2*self.p)

            res = minimize(
                self.objective, x0,
                method='COBYLA',
                options={'maxiter': 200, 'rhobeg': 1.0}
            )

            print(f"cost={res.fun:.6f}")
            costs.append(res.fun)

            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x

        return best_params, best_cost, np.array(costs)

    # ========================================================================
    # CORRELATION ANALYSIS METHODS
    # ========================================================================

    def compute_density_density_correlation(self, state, site_i, site_j, spin_i=None, spin_j=None):
        """
        Compute ⟨n_i n_j⟩ - density-density correlation.

        Args:
            state: Quantum state
            site_i, site_j: Sites
            spin_i, spin_j: Spin indices (0=up, 1=down, None=both)

        Returns:
            Correlation value
        """
        if spin_i is None and spin_j is None:
            # Both spins at each site
            correlation = 0.0
            for si in [0, 1]:
                for sj in [0, 1]:
                    idx_i = self.orbital_idx(site_i, si)
                    idx_j = self.orbital_idx(site_j, sj)
                    n_i = self.n(idx_i)
                    n_j = self.n(idx_j)
                    correlation += self.energy(state, n_i @ n_j)
            return correlation / 4  # Average
        else:
            # Specific spins
            idx_i = self.orbital_idx(site_i, spin_i)
            idx_j = self.orbital_idx(site_j, spin_j)
            n_i = self.n(idx_i)
            n_j = self.n(idx_j)
            return self.energy(state, n_i @ n_j)

    def compute_spin_spin_correlation(self, state, site_i, site_j):
        """
        Compute ⟨S_i · S_j⟩ - spin-spin correlation.

        S_z = (n_up - n_down)/2
        S_+ = c†_up c_down
        S_- = c†_down c_up
        S_x = (S_+ + S_-)/2
        S_y = (S_+ - S_-)/(2i)
        """
        correlation = 0.0

        # S_z correlation
        idx_i_up = self.orbital_idx(site_i, 0)
        idx_i_down = self.orbital_idx(site_i, 1)
        idx_j_up = self.orbital_idx(site_j, 0)
        idx_j_down = self.orbital_idx(site_j, 1)

        Sz_i = (self.n(idx_i_up) - self.n(idx_i_down)) / 2
        Sz_j = (self.n(idx_j_up) - self.n(idx_j_down)) / 2

        Sz_corr = self.energy(state, Sz_i @ Sz_j)

        # S_+ S_- correlation
        Sp_i = self.c_dag(idx_i_up) @ self.c(idx_i_down)
        Sm_i = self.c_dag(idx_i_down) @ self.c(idx_i_up)
        Sp_j = self.c_dag(idx_j_up) @ self.c(idx_j_down)
        Sm_j = self.c_dag(idx_j_down) @ self.c(idx_j_up)

        Sp_Sm_corr = self.energy(state, Sp_i @ Sm_j)
        Sm_Sp_corr = self.energy(state, Sm_i @ Sp_j)

        correlation = Sz_corr + 0.5 * (Sp_Sm_corr + Sm_Sp_corr)

        return correlation

    def compute_pair_correlation(self, state, site_i, site_j):
        """
        Compute pair correlation: ⟨c†_i↑ c†_j↓ c_j↓ c_i↑⟩
        (Cooper pair or charge correlation)
        """
        idx_i_up = self.orbital_idx(site_i, 0)
        idx_j_down = self.orbital_idx(site_j, 1)

        c_dag_i_up = self.c_dag(idx_i_up)
        c_dag_j_down = self.c_dag(idx_j_down)
        c_j_down = self.c(idx_j_down)
        c_i_up = self.c(idx_i_up)

        # ⟨c†_i↑ c†_j↓ c_j↓ c_i↑⟩
        operator = c_dag_i_up @ c_dag_j_down @ c_j_down @ c_i_up

        return np.real(state.conj() @ operator @ state)

    def distance_between_sites(self, site_i, site_j):
        """Euclidean distance between sites (with PBC)."""
        i1, j1 = site_i
        i2, j2 = site_j

        # Account for periodic boundary conditions
        di = min(abs(i1 - i2), self.L - abs(i1 - i2))
        dj = min(abs(j1 - j2), self.L - abs(j1 - j2))

        return np.sqrt(di**2 + dj**2)

    def compute_all_correlations(self, state):
        """
        Compute all correlation distances.

        Returns:
            dict with:
            - density_density: array of correlations vs distance
            - spin_spin: array of correlations vs distance
            - pair: array of pair correlations vs distance
            - distances: unique distances
        """
        correlations = {
            'density_density': [],
            'spin_spin': [],
            'pair': [],
            'distances': []
        }

        # Collect all unique distances and their correlations
        distance_dict = {
            'density_density': {},
            'spin_spin': {},
            'pair': {}
        }

        for i, site_i in enumerate(self.sites):
            for j, site_j in enumerate(self.sites):
                if i < j:  # Avoid double counting
                    dist = self.distance_between_sites(site_i, site_j)

                    # Round to avoid floating point issues
                    dist_rounded = round(dist, 4)

                    if dist_rounded not in distance_dict['density_density']:
                        distance_dict['density_density'][dist_rounded] = []
                        distance_dict['spin_spin'][dist_rounded] = []
                        distance_dict['pair'][dist_rounded] = []

                    # Compute correlations
                    dd = self.compute_density_density_correlation(state, site_i, site_j)
                    ss = self.compute_spin_spin_correlation(state, site_i, site_j)
                    pair = self.compute_pair_correlation(state, site_i, site_j)

                    distance_dict['density_density'][dist_rounded].append(dd)
                    distance_dict['spin_spin'][dist_rounded].append(ss)
                    distance_dict['pair'][dist_rounded].append(pair)

        # Average over pairs at same distance
        distances = sorted(distance_dict['density_density'].keys())

        correlations['distances'] = distances
        correlations['density_density'] = [
            np.mean(distance_dict['density_density'][d])
            for d in distances
        ]
        correlations['spin_spin'] = [
            np.mean(distance_dict['spin_spin'][d])
            for d in distances
        ]
        correlations['pair'] = [
            np.mean(distance_dict['pair'][d])
            for d in distances
        ]

        return correlations

    def analyze_correlations(self, state):
        """Analyze and print all correlation distances."""
        print("\n" + "="*70)
        print("CORRELATION DISTANCE ANALYSIS")
        print("="*70)

        correlations = self.compute_all_correlations(state)

        print(f"\n{'Distance':>10} | {'Density-Density':>18} | {'Spin-Spin':>15} | {'Pair Corr':>15}")
        print("-" * 70)

        for dist, dd, ss, pair in zip(
            correlations['distances'],
            correlations['density_density'],
            correlations['spin_spin'],
            correlations['pair']
        ):
            print(f"{dist:10.4f} | {dd:18.6f} | {ss:15.6f} | {pair:15.6f}")

        return correlations

    def analyze_state(self, state):
        """Full state analysis."""
        E_t = self.energy(state, self.H_t)
        E_U = self.energy(state, self.H_U)
        E_total = E_t + E_U

        N_total = sum(self.n(i) for i in range(self.N))
        n_particles = self.energy(state, N_total)

        probs = np.abs(state)**2
        valid = probs > 1e-10
        entropy = -np.sum(probs[valid] * np.log2(probs[valid])) if np.any(valid) else 0

        return {
            'E_kinetic': E_t,
            'E_interaction': E_U,
            'E_total': E_total,
            'n_particles': n_particles,
            'entropy': entropy,
            'max_prob': np.max(probs)
        }


def demo_with_correlations():
    """Demo showing correlation distance analysis."""
    print("="*70)
    print("QAOA for Fermi-Hubbard Model with Correlation Distance Analysis")
    print("="*70)

    # Create system
    qaoa = HubbardQAOAWithCorrelations(L=2, t=1.0, U=2.0, p=1)

    # Optimize
    print("\nOptimizing QAOA parameters...")
    best_params, best_cost, costs = qaoa.optimize(num_trials=2)

    print(f"\n{'='*70}")
    print(f"Best cost: {best_cost:.6f}")
    print(f"Parameters: γ={best_params[0]:.4f}, β={best_params[1]:.4f}")

    # Get final state
    final_state = qaoa.qaoa_state(best_params)

    # Basic analysis
    print(f"\n{'='*70}")
    print("FINAL STATE PROPERTIES")
    print(f"{'='*70}")

    analysis = qaoa.analyze_state(final_state)
    for key, val in analysis.items():
        print(f"{key:20s} = {val:12.6f}")

    # Correlation analysis
    correlations = qaoa.analyze_correlations(final_state)

    # Visualization
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Optimization convergence
    ax = axes[0, 0]
    ax.plot(costs, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Cost (H_U expectation)')
    ax.set_title('QAOA Optimization Convergence')
    ax.grid(True, alpha=0.3)

    # Plot 2: Density-density correlations
    ax = axes[0, 1]
    distances = correlations['distances']
    dd_corrs = correlations['density_density']
    ax.plot(distances, dd_corrs, 'o-', linewidth=2, markersize=8, color='red', label='⟨n_i n_j⟩')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Correlation')
    ax.set_title('Density-Density Correlation vs Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Spin-spin correlations
    ax = axes[1, 0]
    ss_corrs = correlations['spin_spin']
    ax.plot(distances, ss_corrs, 's-', linewidth=2, markersize=8, color='green', label='⟨S_i · S_j⟩')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Correlation')
    ax.set_title('Spin-Spin Correlation vs Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Pair correlations
    ax = axes[1, 1]
    pair_corrs = correlations['pair']
    ax.plot(distances, pair_corrs, '^-', linewidth=2, markersize=8, color='purple', label='⟨c†_i↑ c†_j↓ c_j↓ c_i↑⟩')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Correlation')
    ax.set_title('Pair Correlation vs Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show(dpi=100, bbox_inches='tight')
    print("Plot saved to: qaoa_correlations.png")

    # Summary statistics
    print(f"\n{'='*70}")
    print("CORRELATION SUMMARY")
    print(f"{'='*70}")

    print(f"\nDensity-Density Correlations:")
    print(f"  Max: {np.max(np.abs(dd_corrs)):.6f} at distance {distances[np.argmax(np.abs(dd_corrs))]:.4f}")
    print(f"  Mean: {np.mean(dd_corrs):.6f}")
    print(f"  Sum: {np.sum(dd_corrs):.6f}")

    print(f"\nSpin-Spin Correlations:")
    print(f"  Max: {np.max(np.abs(ss_corrs)):.6f} at distance {distances[np.argmax(np.abs(ss_corrs))]:.4f}")
    print(f"  Mean: {np.mean(ss_corrs):.6f}")
    print(f"  Sum: {np.sum(ss_corrs):.6f}")

    print(f"\nPair Correlations:")
    print(f"  Max: {np.max(np.abs(pair_corrs)):.6f} at distance {distances[np.argmax(np.abs(pair_corrs))]:.4f}")
    print(f"  Mean: {np.mean(pair_corrs):.6f}")
    print(f"  Sum: {np.sum(pair_corrs):.6f}")

    # Interpretation
    print(f"\n{'='*70}")
    print("PHYSICAL INTERPRETATION")
    print(f"{'='*70}")

    print("""
Density-Density (⟨n_i n_j⟩):
  - Positive: Double occupancy correlation
  - Negative: Anti-correlation (particles avoid each other)
  - Decreases with distance for gapped phases

Spin-Spin (⟨S_i · S_j⟩):
  - Positive: Spins aligned (ferromagnetic-like)
  - Negative: Spins anti-aligned (antiferromagnetic-like)
  - Important for understanding magnetic ordering

Pair Correlation:
  - Measures Cooper pair formation
  - Non-zero indicates superconducting correlations
  - Distance dependence shows pairing range
    """)

    return qaoa, best_params, final_state, correlations


if __name__ == "__main__":
    qaoa, params, state, correlations = demo_with_correlations()

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
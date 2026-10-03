import numpy as np
from neal import SimulatedAnnealingSampler
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import eval_legendre


class LegendreStep:
    def __init__(self, degree=1):
        self.degree = degree
        self.coeffs = self._compute_coeffs()

    def _compute_coeffs(self):
        coeffs = [0.5]  # l=0 term
        for l in range(1, self.degree + 1):
            p_lm1 = eval_legendre(l - 1, 0)
            p_lp1 = eval_legendre(l + 1, 0)
            coeffs.append(0.5 * (p_lm1 + p_lp1))
        return coeffs

    def theta_linear(self, v, v_c, v_max):
        x = 2 * (v - v_c) / v_max - 1
        return self.coeffs[0] + self.coeffs[1] * x

    def one_minus_theta_linear(self, v, v_c, v_max):
        """1 - Θ(v - v_c) - this is what we need for b(v)"""
        return 1 - self.theta_linear(v, v_c, v_max)


class FinancialNetwork:
    def __init__(self, n=5, m=3):
        self.n = n
        self.m = m

        # Cross-holdings - chain structure
        self.C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if abs(i - j) == 1:
                    self.C[i, j] = 0.25
                elif i == j:
                    self.C[i, j] = 0.0  # No self-ownership in C

        # Self-ownership - what's left after cross-holdings
        self.C_tilde = np.diag(1 - np.sum(self.C, axis=0))

        # Asset ownership matrix - THIS WAS MISSING!
        self.D = np.zeros((n, m))
        for i in range(n):
            for k in range(m):
                self.D[i, k] = 0.2  # Equal ownership for simplicity
        # Normalize so each asset is fully owned
        self.D = self.D / np.sum(self.D, axis=0)

        # Propagation matrix
        I = np.eye(n)
        self.A = self.C_tilde @ np.linalg.inv(I - self.C)

        # Failure parameters - tuned for cascade
        self.beta = 0.8
        self.v_c = 0.5


class Encoder:
    def __init__(self, q=2):
        self.q = q
        self.weights = np.array([2 ** a for a in range(-q, q + 1)])
        self.n_bits = len(self.weights)
        self.v_max = np.sum(self.weights)
        self.v_min = 0

    def var(self, i, b):
        return i * self.n_bits + b

    def decode(self, sample, n):
        v = []
        for i in range(n):
            value = 0
            for b in range(self.n_bits):
                if sample.get(self.var(i, b), 0) == 1:
                    value += self.weights[b]
            v.append(value)
        return np.array(v)

    def encode_polynomial(self, i):
        coeffs = {}
        for b in range(self.n_bits):
            coeffs[self.var(i, b)] = self.weights[b]
        return coeffs


def build_qubo(net, enc, p):
    Q = defaultdict(float)
    step = LegendreStep(degree=1)
    Dp = net.D @ p

    for i in range(net.n):

        coeffs = {}

        # Term 1: + v_i
        v_i_coeffs = enc.encode_polynomial(i)
        for var, val in v_i_coeffs.items():
            coeffs[var] = coeffs.get(var, 0) + val

        # Term 2: - Σ_j A_ij * Dp_j  (constant term)
        const_term = -np.sum(net.A[i] * Dp)
        if abs(const_term) > 1e-10:
            coeffs['const'] = coeffs.get('const', 0) + const_term

        for j in range(net.n):

            c1 = step.coeffs[1]

            # Constant part from this term
            const_part = net.A[i, j] * net.beta * (0.5 + c1 * (2 * net.v_c / enc.v_max + 1))
            coeffs['const'] = coeffs.get('const', 0) + const_part

            # v_j part: - (2c1/v_max) * v_j
            v_j_coeffs = enc.encode_polynomial(j)
            factor = -net.A[i, j] * net.beta * (2 * c1 / enc.v_max)
            for var, val in v_j_coeffs.items():
                coeffs[var] = coeffs.get(var, 0) + factor * val

        # Square the expression to get Hamiltonian
        # First, convert to list of (var, coeff) including constant
        terms = [(var, coeff) for var, coeff in coeffs.items() if var != 'const']
        const_val = coeffs.get('const', 0)

        # Square: (Σ coeff_i * x_i + C)^2 = Σ coeff_i^2 x_i + 2Σ_{i<j} coeff_i coeff_j x_i x_j + 2C Σ coeff_i x_i + C^2
        for a, coeff_a in terms:
            # Diagonal terms
            Q[(a, a)] += coeff_a * coeff_a
            # Cross terms with constant
            Q[(a, a)] += 2 * const_val * coeff_a

        for idx_a, (a, coeff_a) in enumerate(terms):
            for idx_b, (b, coeff_b) in enumerate(terms):
                if a < b:
                    Q[(a, b)] += 2 * coeff_a * coeff_b

        # Constant term (ignored for QUBO)

    return dict(Q)


class Solver:
    def __init__(self, n=5, m=3, q=2):
        self.net = FinancialNetwork(n=n, m=m)
        self.enc = Encoder(q=q)
        self.sampler = SimulatedAnnealingSampler()

    def solve(self, p, num_reads=2000):
        Q = build_qubo(self.net, self.enc, p)
        sampleset = self.sampler.sample_qubo(Q, num_reads=num_reads)
        best = sampleset.first.sample
        v = self.enc.decode(best, self.net.n)
        failures = v < self.net.v_c
        return v, failures, sampleset

    def sweep(self, p0, delta_max=10.5, points=12):
        deltas = np.linspace(0, delta_max, points)
        fail_counts = []
        all_values = []

        print("\n" + "=" * 60)
        print("PERTURBATION SWEEP")
        print("=" * 60)

        for d in deltas:
            p = p0 - d * np.ones_like(p0)
            p = np.maximum(p, 0.1)
            v, f, _ = self.solve(p)
            n_failed = np.sum(f)
            fail_counts.append(n_failed)
            all_values.append(v)

            status = "✓ CRASH" if n_failed >= self.net.n / 2 else "   "
            print(f"δ={d:.2f}  v={np.round(v, 2)}  failed={n_failed}/{self.net.n} {status}")

        return deltas, np.array(fail_counts), np.array(all_values)


def main():
    print("=" * 60)
    print("FINANCIAL CRASH PREDICTION VIA QUANTUM ANNEALING")
    print("Simplified implementation of arXiv:1810.07690")
    print("=" * 60)

    # Initialize solver
    solver = Solver(n=5, m=3, q=2)

    print(f"\nNetwork parameters:")
    print(f"  Institutions: {solver.net.n}")
    print(f"  Assets: {solver.net.m}")
    print(f"  β = {solver.net.beta}")
    print(f"  v_c = {solver.net.v_c}")
    print(f"  Binary encoding: q={solver.enc.q}, bits={solver.enc.n_bits}")
    print(f"  Max value: {solver.enc.v_max}")
    print("Correlation matrix")
    print(C)
    print("Self ownership matrix")
    print(FinancialNetwork().D)

    # Initial asset prices
    p0 = np.ones(solver.net.m) * 5.0

    # Linear equilibrium test
    v_linear = solver.net.A @ solver.net.D @ p0
    print(f"\nLinear equilibrium (no failures):")
    print(f"  Asset prices: {p0}")
    print(f"  Market values: {np.round(v_linear, 2)}")
    print(f"  Would fail: {v_linear < solver.net.v_c}")

    # Sweep perturbations
    deltas, failures, values = solver.sweep(
        p0=p0,
        delta_max=1.8,
        points=15
    )

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Number of failures
    ax1.plot(deltas, failures, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Perturbation Amplitude δ', fontsize=12)
    ax1.set_ylabel('Number of Failed Institutions', fontsize=12)
    ax1.set_title('Financial Crash Detection via Quantum Annealing', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, solver.net.n + 0.5)
    ax1.set_yticks(range(solver.net.n + 1))

    # Highlight crash region
    for i, n_fail in enumerate(failures):
        if n_fail >= solver.net.n * 0.6:
            ax1.axvspan(deltas[i], deltas[-1], alpha=0.2, color='red')
            ax1.text(deltas[i] + 0.05, solver.net.n * 0.7,
                     'CRASH REGION', fontsize=12, color='red', fontweight='bold')
            break

    # Plot 2: Market values
    for i in range(solver.net.n):
        ax2.plot(deltas, values[:, i], 'o-', linewidth=2, label=f'Inst {i}')
    ax2.axhline(y=solver.net.v_c, color='r', linestyle='--',
                label='Failure threshold', linewidth=2)
    ax2.set_xlabel('Perturbation Amplitude δ', fontsize=12)
    ax2.set_ylabel('Market Value', fontsize=12)
    ax2.set_title('Institution Market Values vs Perturbation', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig('financial_crash_corrected.png', dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("Results saved to 'financial_crash_corrected.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
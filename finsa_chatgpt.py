def build_linearized_bqm(p, D, C, Cbar, v_c, beta, eps=0.5, scale=1.0):
    import dimod
    import numpy as np

    n = len(v_c)

    A = Cbar @ np.linalg.inv(np.eye(n) - C)

    # nominal equilibrium
    V0 = A @ D @ p

    # smooth failure
    sigma = 1/(1+np.exp(-(v_c - V0)/eps))
    b0 = beta * sigma

    # derivative
    B = -beta/eps * sigma*(1-sigma)
    Bmat = np.diag(B)

    # linearized system
    M = np.eye(n) + A @ Bmat
    g = A @ b0

    Q = M.T @ M
    h = 2 * g @ M

    # convert to spin BQM
    linear = {}
    quadratic = {}

    for i in range(n):
        linear[i] = scale*h[i]

    for i in range(n):
        for j in range(i+1, n):
            quadratic[(i,j)] = scale*2*Q[i,j]

        linear[i] += scale*Q[i,i]

    return dimod.BinaryQuadraticModel(linear, quadratic, float(g@g), vartype=dimod.SPIN), V0

def generate_financial_system(n, m, seed=None):
    rng = np.random.default_rng(seed)

    # ---------------------------
    # Cross holding matrix C
    # ---------------------------
    C = rng.uniform(0.0, 0.2, size=(n, n))
    np.fill_diagonal(C, 0.0)

    # normalize rows so sum < 1
    row_sum = C.sum(axis=1)
    scale = rng.uniform(0.4, 0.9, size=n) / (row_sum + 1e-12)
    C = (C.T * scale).T

    # ensure spectral radius < 1
    eig = max(abs(np.linalg.eigvals(C)))
    if eig >= 0.98:
        C *= 0.98 / eig

    Cbar = np.diag(1 - C.sum(axis=1))

    # ---------------------------
    # Asset matrix D
    # ---------------------------
    D = rng.dirichlet(np.ones(m), size=n)

    # ---------------------------
    # Asset prices p
    # ---------------------------
    p = rng.uniform(5, 25, size=m)

    # ---------------------------
    # Compute equilibrium scale
    # ---------------------------
    A = Cbar @ np.linalg.inv(np.eye(n) - C)
    V_nominal = A @ D @ p

    # ---------------------------
    # Default thresholds
    # ---------------------------
    v_c = rng.uniform(0.6, 0.95, size=n) * V_nominal

    # ---------------------------
    # Failure penalties
    # ---------------------------
    beta = rng.uniform(0.2, 0.8, size=n) * V_nominal

    return p, D, C, Cbar, v_c, beta

import numpy as np
import neal
n = 20
m = 10
q = 3

p, D, C, Cbar, v_c, beta = generate_financial_system(n, m, seed=42)
bqm, V0 = build_linearized_bqm(p, D, C, Cbar, v_c, beta)

sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=1000, num_sweeps=10000)

#best = sampleset.first.sample
#delta = np.array([best[i] for i in range(len(best))])
#V = V0 + delta

#print("Equilibrium:", V)
#print("Critical", v_c)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ---- extract samples ----
samples = sampleset.record.sample
energies = sampleset.record.energy
n = len(sampleset.variables)

# convert structured array → simple numpy array
states = np.array([list(s) for s in samples])

# ---- count frequencies ----
tuple_states = [tuple(s) for s in states]
freq = Counter(tuple_states)

keys = [str(k) for k in freq.keys()]
values = list(freq.values())

# most frequent state
max_key = max(freq, key=freq.get)
max_count = freq[max_key]
best_spin = np.array(max_key)

print("Most frequent spin state:", best_spin)
print("Frequency:", max_count)

# ---- compute equilibrium V ----
delta = best_spin
V = V0 + delta

print("Equilibrium values:", V)
print("Critical values:", v_c)

plt.figure(figsize=(10,4))
plt.bar(keys, values)
plt.xlabel('Spin configurations')
plt.ylabel('Frequency')
plt.title('Sampling frequencies')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6,4))
plt.bar(range(n), V, alpha=0.6, color='red', label="Equilibrium Market Values")
plt.bar(range(n), v_c, alpha=0.4, color='blue', label="Critical Market Values")
plt.xlabel('Institutions')
plt.ylabel('Value')
plt.legend()
plt.title('Financial Equilibrium vs Default Threshold')
plt.show()
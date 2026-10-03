import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from mpi4py import MPI
import dimod
import neal

# ==============================
# MPI INIT
# ==============================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ==============================
# PARAMETERS
# ==============================

n = 5
m = 3
Q = 3
r = 3
TOTAL_READS = 1000
NUM_SWEEPS = 100000

reads_per_rank = TOTAL_READS // size

# ==============================
# DATA
# ==============================

p = np.array([10, 15, 20])

D = np.array([
    [0.3, 0.2, 0.1],
    [0.2, 0.3, 0.2],
    [0.1, 0.2, 0.3],
    [0.2, 0.1, 0.2],
    [0.2, 0.2, 0.2]
])

C = np.array([
    [0.0, 0.1, 0.05, 0.0, 0.0],
    [0.1, 0.0, 0.1, 0.05, 0.0],
    [0.05, 0.1, 0.0, 0.1, 0.05],
    [0.0, 0.05, 0.1, 0.0, 0.1],
    [0.0, 0.0, 0.05, 0.1, 0.0]
])

Cbar = np.diag(1 - np.sum(C, axis=1))
v_c = np.array([5, 6, 5.5, 6, 5])
beta = np.array([2, 2.5, 2, 2.5, 2])

# ==============================
# HUBO UTILITIES
# ==============================

def bit_index(inst, alpha):
    return inst * (2 * Q + 1) + Q + alpha

def expand_vi(inst):
    terms = {}
    for alpha in range(-Q, Q + 1):
        terms[(bit_index(inst, alpha),)] = 2 ** alpha
    return terms

def multi_poly(A, B):
    result = defaultdict(float)
    for bitsA, coeffA in A.items():
        for bitsB, coeffB in B.items():
            result[tuple(sorted(bitsA + bitsB))] += coeffA * coeffB
    return result

def add_term(source, target, scale=1.0):
    for bits, coeff in source.items():
        target[bits] += scale * coeff

def poly_power(poly, k):
    if k == 0:
        return {(): 1.0}
    result = poly
    for _ in range(k - 1):
        result = multi_poly(result, poly)
    return result

def failure_poly(i):
    result = defaultdict(float)
    vi = expand_vi(i)

    for l in range(1, r + 1):
        coeff = 0.5 * (
            np.polynomial.legendre.legval(0, [0]*(l-1)+[1]) -
            np.polynomial.legendre.legval(0, [0]*(l+1)+[1])
        )
        scale = -beta[i] * coeff
        v_power = poly_power(vi, l)
        add_term(v_power, result, scale)

    return result

# ==============================
# BUILD BQM (ROOT ONLY)
# ==============================

def build_bqm():

    HUBO = defaultdict(float)
    A = Cbar @ np.linalg.inv(np.eye(n) - C)

    for i in range(n):

        Ri = defaultdict(float)

        # Linear vi term
        add_term(expand_vi(i), Ri)

        # Constant part
        const = 0.0
        for j in range(n):
            for k in range(m):
                const += A[i, j] * D[j, k] * p[k]

        Ri[()] += -const

        # Failure coupling
        for j in range(n):
            bj = failure_poly(j)
            add_term(bj, Ri, scale=A[i, j])

        Ri_sq = multi_poly(Ri, Ri)
        add_term(Ri_sq, HUBO)

    poly = dimod.BinaryPolynomial(HUBO, vartype="BINARY")
    strength = 2 * max(abs(v) for v in HUBO.values()) if HUBO else 1.0
    bqm = dimod.make_quadratic(poly, strength=strength, vartype=dimod.BINARY)

    return bqm

if rank == 0:
    print("Building BQM...")
    bqm = build_bqm()
else:
    bqm = None

bqm = comm.bcast(bqm, root=0)

# ==============================
# SAMPLING
# ==============================

sampler = neal.SimulatedAnnealingSampler()

sampleset = sampler.sample(
    bqm,
    num_reads=reads_per_rank,
    num_sweeps=NUM_SWEEPS
)

local_results = [tuple(s) for s in sampleset.record.sample]

# ==============================
# GATHER RESULTS
# ==============================

all_results = comm.gather(local_results, root=0)

# ==============================
# ROOT ANALYSIS
# ==============================

if rank == 0:

    flat = [item for sublist in all_results for item in sublist]

    freq = Counter(flat)

    max_key = max(freq, key=freq.get)
    max_count = freq[max_key]

    print("\nMost frequent bitstring:")
    print(max_key)
    print("Frequency:", max_count)

    # Decode values
    def decode(bitstring):
        v = np.zeros(n)
        for i in range(n):
            for alpha in range(-Q, Q + 1):
                idx = i * (2 * Q + 1) + Q + alpha
                v[i] += bitstring[idx] * (2 ** alpha)
        return v

    v = decode(max_key)

    print("\nDecoded equilibrium values:")
    print(v)
    print("\nCritical values:")
    print(v_c)

    # ==============================
    # PLOTS
    # ==============================

    plt.figure(figsize=(12,4))
    plt.bar(range(n), v, alpha=0.6, label="Equilibrium")
    plt.bar(range(n), v_c, alpha=0.4, label="Critical")
    plt.legend()
    plt.xlabel("Institution")
    plt.ylabel("Value")
    plt.title("Equilibrium vs Critical Values")
    plt.show()

    plt.figure(figsize=(10,4))
    plt.bar([str(k) for k in freq.keys()], list(freq.values()))
    plt.xticks(rotation=90)
    plt.title("Bitstring Frequencies")
    plt.show()
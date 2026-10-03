"""
Maximum Cascade Problem — QUBO Builder + dimod Simulated Annealing
===================================================================
Paper: "Finance Cascading Problem for QUBOs" (Nagpal, Jan 2026)

Pipeline:
    market dict
        │
        ├─► Q_obj  +  Q_cascade  +  Q_shock
        │                │
        │           Q = sum of three terms
        │                │
        │           Q_sym = (Q + Q.T) / 2
        │                │
        │      ┌─────────┴──────────┐
        │    h_i = row_sum(Q_sym)   J_ij = -Q_sym[i,j]/2
        │                │
        │         dimod BinaryQuadraticModel (SPIN)
        │                │
        └─►     SimulatedAnnealingSampler
                         │
                    ground state s*  →  maximum cascade config
"""

import numpy as np
import networkx as nx
import dimod
from dimod import SimulatedAnnealingSampler

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — MARKET SETUP
# ══════════════════════════════════════════════════════════════════

n = 5
m = 5

names  = ["Bank_1", "Bank_2", "Bank_3", "Bank_4", "Bank_5"]
assets = ["Equity", "Corp_Bond", "Govt_Bond", "Real_Estate", "Commodity"]

# Asset prices  p ∈ R^m
p = np.ones(m, dtype=float)

# Asset holdings  D (n x m)  —  D[i,j] = units of asset j held by institution i
D = np.array([
    #  Equity  CorpBond  GovtBond  RealEst  Commodity
    [  40.0,    60.0,     80.0,    20.0,     10.0 ],   # Bank_1
    [  30.0,    70.0,     60.0,    15.0,      5.0 ],   # Bank_2
    [  80.0,    40.0,     20.0,    10.0,     25.0 ],   # Bank_3
    [  20.0,    50.0,    100.0,    40.0,      5.0 ],   # Bank_4
    [  10.0,    30.0,    120.0,    50.0,      5.0 ],   # Bank_5
], dtype=float)

# Cross-holdings  C (n x n)  —  C[i,j] = fraction of institution j owned by i
# Constraints: diagonal = 0,  column sums < 1
C = np.array([
    # B1    B2    B3    B4    B5
    [0.00, 0.20, 0.00, 0.00, 0.00],   # Bank_1 owns 20% of Bank_2
    [0.15, 0.00, 0.20, 0.00, 0.00],   # Bank_2 owns 15% of B1, 20% of B3
    [0.00, 0.10, 0.00, 0.15, 0.00],   # Bank_3 owns 10% of B2, 15% of B4
    [0.00, 0.00, 0.08, 0.00, 0.10],   # Bank_4 owns 8%  of B3, 10% of B5
    [0.00, 0.00, 0.00, 0.05, 0.00],   # Bank_5 owns 5%  of Bank_4
], dtype=float)

assert np.all(np.diag(C) == 0),      "C diagonal must be zero"
assert np.all(C.sum(axis=0) < 1.0),  "C column sums must be < 1"
assert np.all(C >= 0),               "C must be non-negative"

# Derived quantities — paper Definitions 2-3, equations (3)-(4)
ext_own   = 1.0 - C.sum(axis=0)          # external ownership fractions
C_hat     = np.diag(ext_own)             # diagonal matrix
IminC_inv = np.linalg.inv(np.eye(n) - C) # (I - C)^{-1}
A         = C_hat @ IminC_inv            # dependency matrix  A = C_hat (I-C)^{-1}
V         = IminC_inv @ D @ p            # book value   V = (I-C)^{-1} D p
v         = A @ D @ p                    # market value  v = A D p

assert np.allclose(A.sum(axis=0), 1.0, atol=1e-8), "A not column-stochastic"

# Failure thresholds and costs — paper Definition 4
v_bar = 0.08 * v        # institution i fails if v_i < v_bar_i
beta  = 0.20 * V        # discontinuous loss upon failure

# QUBO hyper-parameters — paper Section 4.2, eq (25)
THETA   = 0.10    # raised threshold: cuts weak edges to Bank_4, Bank_5
w       = 1.0     # objective weight  (maximize failures)
lambda_ = 10.0    # cascade constraint penalty weight
M       = 100.0   # initial shock enforcement penalty
M_iso   = 50.0    # isolation penalty: blocks unjustified failures for
                  # nodes with no upstream cascade edge above THETA
SHOCK   = 2       # index of institution forced to fail (Bank_3)

# Pack into market dict for reference
market = {
    "n": n, "m": m, "names": names, "assets": assets,
    "p": p, "D": D, "C": C,
    "C_hat": C_hat, "ext_own": ext_own, "IminC_inv": IminC_inv,
    "A": A, "V": V, "v": v, "v_bar": v_bar, "beta": beta,
    "theta": THETA, "w": w, "lambda_": lambda_, "M": M, "M_iso": M_iso, "shock": SHOCK,
}


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — QUBO MATRIX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════
# Q is n x n.  Three additive contributions:
#   Q_obj     : objective  — maximize total failures
#   Q_cascade : cascade consistency constraint
#   Q_shock   : force shock institution to fail
#
# Full formula (paper eq 25):
#   Q[i,i] = -w  +  Σ_{j: A[j,i]>θ} λ·A[j,i]  +  (-M)·δ(i, SHOCK)
#   Q[i,j] = -λ·A[j,i]   if A[j,i] > θ,  i ≠ j
#   Q[i,j] = 0            otherwise

def build_qubo(A, n, w, lambda_, M, M_iso, theta, shock):
    """
    Build the raw (possibly asymmetric) QUBO matrix Q.

    Parameters
    ----------
    A       : (n,n) dependency matrix, column-stochastic
    n       : number of institutions
    w       : objective weight
    lambda_ : cascade penalty weight
    M       : shock enforcement penalty
    theta   : edge inclusion threshold on A
    shock   : index of institution forced to fail

    Returns
    -------
    Q       : (n,n) raw QUBO matrix
    Q_sym   : (n,n) symmetrized QUBO  =  (Q + Q.T) / 2
    """

    Q = np.zeros((n, n), dtype=float)

    # ── Term 1 : Objective  Q_obj[i,i] = -w  (paper eq 18) ──────
    for i in range(n):
        Q[i, i] += -w

    # ── Term 2 : Cascade constraint  (paper eqs 20-22) ───────────
    # For each directed dependency edge j → i  (A[i,j] > theta):
    #   penalize xj=1, xi=0  via  λ·A[i,j]·xj(1 - xi)
    #   expanding: λ·A[i,j]·xj  -  λ·A[i,j]·xi·xj
    #
    #   Q_cascade[j,j] += λ·A[i,j]   (linear term on j's diagonal)
    #   Q_cascade[i,j] += -λ·A[i,j]  (off-diagonal penalty)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > theta:
                Q[j, j] += lambda_ * A[i, j]   # diagonal of j
                Q[i, j] += -lambda_ * A[i, j]  # off-diagonal (i,j)

    # ── Term 3 : Shock constraint  (paper eq 23-24) ──────────────
    # Enforce x_shock = 1 via penalty M(1 - x_shock)^2
    # Expanding: M - 2M·x + M·x^2 = M - 2M·x + M·x  (x^2=x binary)
    #          = M - M·x  →  linear term -M on shock diagonal
    # (constant M drops out of optimisation)
    Q[shock, shock] += -M

    # ── Term 4 : Isolation penalty ────────────────────────────────
    # Nodes with NO upstream edge above theta (except the shock node)
    # have no cascade justification for failing.
    # Penalise their failure with M_iso on the diagonal:
    #   Q[i,i] += M_iso  →  adds M_iso cost to x_i = 1 if no upstream
    # This ensures Bank_4 and Bank_5 only fail if their energy benefit
    # from couplings exceeds M_iso, which it won't given w=1.
    for i in range(n):
        if i == shock:
            continue
        upstream = [j for j in range(n)
                    if j != i and A[i, j] > theta]
        if not upstream:
            Q[i, i] += M_iso

    # ── Symmetrize  Q_sym = (Q + Q.T) / 2  (paper Section 3.3) ──
    Q_sym = (Q + Q.T) / 2.0

    return Q, Q_sym


Q, Q_sym = build_qubo(A, n, w, lambda_, M, M_iso, THETA, SHOCK)


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — ISING PARAMETER EXTRACTION
# ══════════════════════════════════════════════════════════════════
# From Q_sym, extract Ising local fields h_i and couplings J_ij.
# These are used for reporting and interpretation only.
# The BQM in Section 4 is built directly from Q in BINARY vartype
# to avoid sign-convention issues between QUBO and Ising.
#
# Paper equations (16)-(17):
#   h_i  = row_sum_i(Q_sym)
#   J_ij = -Q_sym[i,j] / 2   (i ≠ j, upper triangle)

def extract_ising(Q_sym, n):
    """
    Extract local fields h and couplings J from symmetrized QUBO.
    For interpretation purposes only — BQM uses BINARY vartype directly.
    """
    h = Q_sym.sum(axis=1)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = -Q_sym[i, j] / 2.0
    return h, J


h, J = extract_ising(Q_sym, n)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD dimod BinaryQuadraticModel  (BINARY vartype)
# ══════════════════════════════════════════════════════════════════
# We use BINARY vartype (x in {0,1}) so dimod minimises x^T Q x
# directly — no Ising sign conversion needed.
#
# Why BINARY not SPIN:
#   The QUBO Q matrix is constructed so that x=1 means fail and
#   the minimum of x^T Q x is the maximum cascade configuration.
#   Converting to Ising (s = 2x-1) introduces sign flips in h_i
#   that cause the cascade penalty diagonal to oppose failure.
#   Using BINARY bypasses this entirely.
#
# dimod BINARY convention:
#   add_variable(i, bias)        →  bias * x_i        in energy
#   add_interaction(i, j, bias)  →  bias * x_i * x_j  in energy
#
# We load the upper triangle of Q_sym:
#   diagonal    → linear  biases (Q_sym[i,i])
#   off-diagonal → quadratic biases (Q_sym[i,j] + Q_sym[j,i] = 2*Q_sym[i,j])

def build_bqm(Q_sym, n, names):
    """
    Construct a dimod BinaryQuadraticModel in BINARY vartype
    directly from the symmetrized QUBO matrix.

    Parameters
    ----------
    Q_sym : (n,n) symmetrized QUBO matrix
    n     : number of institutions
    names : list of institution names (variable labels)

    Returns
    -------
    bqm : dimod.BinaryQuadraticModel (BINARY)
    """

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    # Linear biases from diagonal of Q_sym
    for i in range(n):
        bqm.add_variable(names[i], Q_sym[i, i])

    # Quadratic biases from upper triangle of Q_sym
    # Q_sym is symmetric so Q_sym[i,j]+Q_sym[j,i] = 2*Q_sym[i,j]
    for i in range(n):
        for j in range(i + 1, n):
            if Q_sym[i, j] != 0.0:
                bqm.add_interaction(names[i], names[j], 2.0 * Q_sym[i, j])

    return bqm


bqm = build_bqm(Q_sym, n, names)


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — SIMULATED ANNEALING VIA dimod
# ══════════════════════════════════════════════════════════════════
# dimod's SimulatedAnnealingSampler takes the BQM directly.
# Key parameters:
#   num_reads       : number of independent SA runs (take best)
#   num_sweeps      : total spin-flip sweeps per run
#   beta_range      : [beta_start, beta_end]  (inverse temperature)
#                     beta_start low  = high T  (explore freely)
#                     beta_end   high = low T   (exploit / freeze)
#   beta_schedule_type : 'geometric' matches standard SA cooling

def run_sa(bqm, num_reads=1000, num_sweeps=2000,
           beta_start=0.1, beta_end=10.0):
    """
    Run Simulated Annealing on the BQM using dimod's sampler.

    Parameters
    ----------
    bqm         : dimod.BinaryQuadraticModel (SPIN)
    num_reads   : number of independent SA runs
    num_sweeps  : sweeps per run
    beta_start  : starting inverse temperature (high T, exploration)
    beta_end    : ending inverse temperature   (low T, exploitation)

    Returns
    -------
    sampleset : dimod.SampleSet  (sorted by energy, best first)
    """

    sampler = SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=[beta_start, beta_end],
        beta_schedule_type="geometric",
        seed=42,
    )

    # Sort so lowest energy (ground state) is first
    sampleset = sampleset.truncate(num_reads, sorted_by="energy")

    return sampleset


sampleset = run_sa(bqm)


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — RESULTS AND VALIDATION
# ══════════════════════════════════════════════════════════════════

def decode_results(sampleset, names, n):
    """
    Decode the SA sampleset into human-readable cascade results.

    Parameters
    ----------
    sampleset : dimod.SampleSet
    names     : institution names
    n         : number of institutions

    Returns
    -------
    best_sample : dict  {name: spin (+1 fail / -1 survive)}
    failures    : list of failed institution names
    energy      : ground state energy
    magnetization : (1/n) Σ s_i
    """

    best        = sampleset.first
    best_sample = best.sample
    energy      = best.energy

    # BINARY vartype: x=1 means fail, x=0 means survive
    failures = [names[i] for i in range(n)
                if best_sample[names[i]] == 1]

    # Magnetization: map x in {0,1} -> s in {-1,+1} via s = 2x-1
    spins         = np.array([2 * best_sample[nm] - 1 for nm in names])
    magnetization = spins.mean()

    return best_sample, failures, energy, magnetization


def validate_cascade(failures, names, A, theta):
    """
    Check that the failure set is cascade-consistent:
    for every edge j->i (A[i,j] > theta), if j fails then i must fail.

    Parameters
    ----------
    failures : list of failed institution names
    names    : all institution names
    A        : dependency matrix
    theta    : edge threshold

    Returns
    -------
    valid    : bool
    violations : list of (j_name, i_name) pairs that violate consistency
    """

    failed_idx  = {names.index(nm) for nm in failures}
    violations  = []

    for j in failed_idx:
        for i in range(len(names)):
            if i != j and A[i, j] > theta:
                if i not in failed_idx:
                    violations.append((names[j], names[i]))

    return len(violations) == 0, violations


def print_results(sampleset, names, n, A, theta, h, J, Q, Q_sym):

    best_sample, failures, energy, magnetization = decode_results(
        sampleset, names, n)

    valid, violations = validate_cascade(failures, names, A, theta)

    print("=" * 62)
    print("  MAXIMUM CASCADE — RESULTS")
    print("=" * 62)

    print("\n[ QUBO Matrix Q (raw, asymmetric) ]")
    np.set_printoptions(precision=3, suppress=True)
    print(Q)

    print("\n[ QUBO Matrix Q_sym (symmetrized) ]")
    print(Q_sym)

    print("\n[ Ising Parameters ]")
    print("  Local fields h_i:")
    for i, nm in enumerate(names):
        print(f"    h[{nm}] = {h[i]:8.4f}")
    print("\n  Couplings J_ij (upper triangle, non-zero only):")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-10:
                print(f"    J[{names[i]}, {names[j]}] = {J[i,j]:8.4f}")

    print("\n[ Ground State (SA solution) ]")
    print(f"  Energy         : {energy:.6f}")
    print(f"  Magnetization  : {magnetization:.4f}  "
          f"(+1 = all fail, -1 = all survive)")
    print("\n  Spin configuration:")
    for nm in names:
        s   = best_sample[nm]
        tag = "FAIL ✗" if s == 1 else "survive ✓"
        print(f"    {nm:10s}  s={s:+d}  {tag}")

    print(f"\n  Failed institutions ({len(failures)}/{n}): "
          f"{failures if failures else 'none'}")

    print("\n[ Cascade Consistency Check ]")
    if valid:
        print("  ✓ Failure set is cascade-consistent")
    else:
        print("  ✗ Violations found:")
        for j_nm, i_nm in violations:
            print(f"    {j_nm} fails but {i_nm} survives "
                  f"(A[{i_nm},{j_nm}] = {A[names.index(i_nm), names.index(j_nm)]:.4f} > θ)")

    print("\n[ Top 20 Samples by Energy ]")
    print(f"  {'Rank':>4}  {'Energy':>10}  {'#Fail':>6}  {'Config':>7}  Failed Institutions")
    print("  " + "-" * 58)
    seen = {}
    rank = 0
    for sample, energy_val, _ in sampleset.data(['sample', 'energy', 'num_occurrences']):
        spins  = [sample[nm] for nm in names]
        config = "".join("F" if s == 1 else "S" for s in spins)
        if config in seen:          # deduplicate identical configs
            continue
        seen[config] = True
        n_fail = sum(1 for s in spins if s == 1)
        fails  = [names[i] for i, s in enumerate(spins) if s == 1]
        survive= [names[i] for i, s in enumerate(spins) if s == 0]
        rank  += 1
        marker = "  ◄ ground state" if rank == 1 else ""
        print(f"  {rank:>4}  {energy_val:>10.4f}  {n_fail:>6}  {config}  "
              f"{', '.join(f[-1] for f in fails) or 'none':>12} fail | "
              f"{', '.join(s[-1] for s in survive) or 'none'} survive{marker}")
        if rank >= 20:
            break

    print("\n  Legend: F=Fail  S=Survive  |  Bank number shown in failed/survived cols")
    print("=" * 62)


print_results(sampleset, names, n, A, THETA, h, J, Q, Q_sym)
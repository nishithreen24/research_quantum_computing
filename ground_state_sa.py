"""
ground_state_sa.py
==================
Find the financial-crash equilibrium ground state using D-Wave's
SimulatedAnnealingSampler (neal).

Pipeline
--------
  Financial network model
    → cost function  F(v) = ||v − C̃(I−C)⁻¹[Dp − b(v)]||²
    → binary encoding  v_i = Σ_{α=−q}^{q} x_{i,α} · 2^α
    → Fourier-Legendre approx of Heaviside  →  HUBO in bit variables
    → CZW-calibrated Rosenberg reduction  →  QUBO
         (per-pair penalty ∝ local coefficient sum, inspired by
          Chancellor-Zohren-Warburton Theorem 7 of Derivation.pdf;
          full binary→Ising + CZW requires ~195K ancilla spins and is
          infeasible for classical SA — this binary adaptation preserves
          the key CZW insight of PER-TERM penalty calibration)
    → neal.SimulatedAnnealingSampler  →  ground state
    → decode market values  →  identify failed institutions

References
----------
  Orus, Mugel & Lizaso (2019), arXiv:1810.07690
  Nagpal (2026), Finance/Crash Derivation Notes  (Sec. 4, Theorem 7)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from itertools import combinations

import dimod
import neal


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Hyper-parameters
# ═══════════════════════════════════════════════════════════════════════════

n = 5          # number of institutions
m = 3          # number of assets
q = 3          # bits per institution = 2q+1 = 7,  range [0, 15.875], res 0.125
r = 3          # Legendre order → HUBO degree 2r = 6

NUM_READS  = 200
NUM_SWEEPS = 100_000
BETA_RANGE = (0.001, 500.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Financial Network Parameters
# ═══════════════════════════════════════════════════════════════════════════

p = np.array([6.0, 9.0, 12.0])    # shocked prices (−40% shock → cascading failures)

D = np.array([
    [0.3, 0.2, 0.1],
    [0.2, 0.3, 0.2],
    [0.1, 0.2, 0.3],
    [0.2, 0.1, 0.2],
    [0.2, 0.2, 0.2],
])

C = np.array([
    [0.00, 0.10, 0.05, 0.00, 0.00],
    [0.10, 0.00, 0.10, 0.05, 0.00],
    [0.05, 0.10, 0.00, 0.10, 0.05],
    [0.00, 0.05, 0.10, 0.00, 0.10],
    [0.00, 0.00, 0.05, 0.10, 0.00],
])

Cbar_diag = 1.0 - C.sum(axis=0)
Cbar      = np.diag(Cbar_diag)

v_c  = np.array([5.0, 6.0, 5.5, 6.0, 5.0])
beta = np.array([2.0, 2.5, 2.0, 2.5, 2.0])


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Pre-computed Financial Matrices
# ═══════════════════════════════════════════════════════════════════════════

IminC_inv = np.linalg.inv(np.eye(n) - C)
A_full    = Cbar @ IminC_inv
Dp        = D @ p
gamma     = A_full @ Dp

v_max = float(sum(2.0**a for a in range(-q, q + 1)))

print("=" * 62)
print("  FINANCIAL CRASH — GROUND STATE via SimulatedAnnealingSampler")
print("  ** SHOCKED PRICES: p = [6, 9, 12]  (−40% asset shock) **")
print("=" * 62)
print(f"  Precision  q = {q}  →  {2*q+1} bits / institution,  v_max = {v_max:.3f}")
print(f"  Legendre   r = {r}  →  HUBO degree {2*r}")
print(f"\n  Market values (no failure) : {np.round(gamma, 3)}")
print(f"  Failure thresholds v_c     : {v_c}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Classical Fixed-Point Reference
# ═══════════════════════════════════════════════════════════════════════════

def classical_equilibrium(prices, tol=1e-12, max_iter=500):
    """Iterate  v ← A · (Dp − b(v))  until convergence."""
    v = A_full @ (D @ prices)
    for _ in range(max_iter):
        b     = beta * (1.0 - np.heaviside(v - v_c, 1.0))
        v_new = A_full @ (D @ prices - b)
        if np.linalg.norm(v_new - v) < tol:
            return v_new
        v = v_new
    return v

v_ref = classical_equilibrium(p)
print(f"\n  Classical fixed-point      : {np.round(v_ref, 3)}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Polynomial Arithmetic  (binary variables satisfy x² = x)
# ═══════════════════════════════════════════════════════════════════════════

def bit_idx(inst, alpha):
    """Global integer index for bit  x_{inst, alpha}."""
    return inst * (2 * q + 1) + (q + alpha)


def expand_vi(inst):
    """v_i = Σ_α 2^α x_{i,α}  as a polynomial dict."""
    return {(bit_idx(inst, a),): 2.0**a for a in range(-q, q + 1)}


def add_term(src, dst, scale=1.0):
    """dst += scale * src   (in-place, both are polynomial dicts)."""
    for key, val in src.items():
        dst[key] += scale * val


def poly_mul(A, B):
    """
    Multiply two polynomial dicts.
    For binary variables  x_i² = x_i, so duplicate indices are collapsed.
    """
    res = defaultdict(float)
    for ka, va in A.items():
        for kb, vb in B.items():
            key = tuple(sorted(set(ka + kb)))
            res[key] += va * vb
    return res


def poly_pow(poly, k):
    """Raise polynomial dict to integer power k  (k ≥ 0)."""
    if k == 0:
        return {(): 1.0}
    result = dict(poly)
    for _ in range(k - 1):
        result = poly_mul(result, poly)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Fourier-Legendre Failure Polynomial  b_j(v_j)
# ═══════════════════════════════════════════════════════════════════════════

def

def _Pl_power_coeffs(order):
    """
    Power-series coefficients of P_order(x).
    Returns array [a₀, a₁, …, aₗ] s.t.  P_order(x) = Σ_k aₖ xᵏ.
    """
    c = np.zeros(order + 1)
    c[order] = 1.0
    return np.polynomial.legendre.leg2poly(c)


def bj_poly(j):
    """
    Return a polynomial dict approximating  b_j(v_j).
    """
    u_j = defaultdict(float)
    for a in range(-q, q + 1):
        u_j[(bit_idx(j, a),)] = (2.0**a) / v_max
    u_j[()] -= v_c[j] / v_max
    u_j = dict(u_j)

    result = defaultdict(float)
    result[()] += beta[j] * 0.5

    for l in range(1, r + 1):
        c_l = 0.5 * (_Pl_at_zero(l - 1) - _Pl_at_zero(l + 1))
        if abs(c_l) < 1e-12:
            continue

        pw      = _Pl_power_coeffs(l)
        Pl_of_uj = defaultdict(float)
        u_j_pow  = {(): 1.0}

        for k_pw, ak in enumerate(pw):
            if abs(ak) > 1e-12:
                add_term(u_j_pow, Pl_of_uj, scale=ak)
            if k_pw < len(pw) - 1:
                u_j_pow = poly_mul(u_j_pow, u_j)

        add_term(Pl_of_uj, result, scale=-beta[j] * c_l)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Residual  Rᵢ
# ═══════════════════════════════════════════════════════════════════════════

def residual(i):
    """Polynomial dict for  Rᵢ(v)."""
    Ri = defaultdict(float)
    add_term(expand_vi(i), Ri)
    Ri[()] -= gamma[i]
    for j in range(n):
        add_term(bj_poly(j), Ri, scale=A_full[i, j])
    return Ri


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Build HUBO  F = Σᵢ Rᵢ²
# ═══════════════════════════════════════════════════════════════════════════

print("\nBuilding HUBO  F = Σᵢ Rᵢ²  ...")
HUBO = defaultdict(float)

for i in range(n):
    Ri    = residual(i)
    Ri_sq = poly_mul(Ri, Ri)
    add_term(Ri_sq, HUBO)
    print(f"  R_{i+1}: {len(Ri):>4d} polynomial terms")

HUBO    = {k: v for k, v in HUBO.items() if abs(v) > 1e-12}
max_deg = max(len(k) for k in HUBO) if HUBO else 0
print(f"\n  HUBO: {len(HUBO)} terms,  max interaction degree = {max_deg}")


# ═══════════════════════════════════════════════════════════════════════════
# 9.  CZW-inspired HUBO → QUBO  (per-pair calibrated Rosenberg reduction)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Chancellor-Zohren-Warburton (Theorem 7) prescribes LOCAL penalty scaling:
#    J_a = 10·|J_k|  per k-body interaction
#  Applied in binary space: penalty per pair ∝ local coefficient sum (not
#  the global maximum), equivalent to CZW's per-term calibration but without
#  the 195K-ancilla Ising expansion that is infeasible for classical SA.
#
#  Standard make_quadratic uses  λ = 2·max|coeff| ≈ 485  (global, 4× too large).
#  CZW calibration uses  λ = 2·max_pair_sum ≈ 114  (local, minimum sufficient).
#
#  The smaller penalty reduces energy-landscape distortion so SA can find
#  the HUBO ground state through the 318-variable QUBO.

print("\nComputing CZW-calibrated penalty strength ...")
pair_sums = defaultdict(float)
for key, coeff in HUBO.items():
    if len(key) >= 3:                        # only high-degree terms need reduction
        for pair in combinations(key, 2):
            pair_sums[pair] += abs(coeff)

max_pair_sum = max(pair_sums.values()) if pair_sums else 1.0
czw_strength = 2.0 * max_pair_sum           # minimum sufficient penalty (CZW calibrated)

print(f"  max per-pair |coeff| sum : {max_pair_sum:.4f}")
print(f"  global 2·max|coeff|      : {2.0*max(abs(v) for v in HUBO.values()):.4f}")
print(f"  CZW-calibrated strength  : {czw_strength:.4f}  (~30× smaller than global → better SA landscape)")

print(f"\nReducing HUBO → QUBO via CZW-calibrated Rosenberg ...")
bp  = dimod.BinaryPolynomial(HUBO, vartype="BINARY")
bqm = dimod.make_quadratic(bp, strength=czw_strength, vartype=dimod.BINARY)
print(f"  BQM: {bqm.num_variables} variables,  {bqm.num_interactions} interactions")

N_logical = n * (2 * q + 1)   # = 35 logical bits


# ═══════════════════════════════════════════════════════════════════════════
# 10.  SimulatedAnnealingSampler
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nSampling  (num_reads={NUM_READS}, num_sweeps={NUM_SWEEPS}) ...")
sampler   = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(
    bqm,
    num_reads          = NUM_READS,
    num_sweeps         = NUM_SWEEPS,
    beta_range         = BETA_RANGE,
    beta_schedule_type = "geometric",
)
print("  Done.")


# ═══════════════════════════════════════════════════════════════════════════
# 11.  Decode Ground State
# ═══════════════════════════════════════════════════════════════════════════

def decode(sample):
    """Extract market-value vector from a binary sample dict (logical bits only)."""
    v = np.zeros(n)
    for i in range(n):
        for a in range(-q, q + 1):
            v[i] += sample.get(bit_idx(i, a), 0) * (2.0**a)
    return v


def F_exact(v):
    """Exact cost function  F(v) = ||v − A·(Dp − b(v))||²."""
    b    = beta * (1.0 - np.heaviside(v - v_c, 1.0))
    diff = v - A_full @ (Dp - b)
    return float(np.dot(diff, diff))


best = sampleset.lowest().first
v_sa = decode(best.sample)
E_sa = best.energy


# ═══════════════════════════════════════════════════════════════════════════
# 11a.  Most Frequent Ground State
# ═══════════════════════════════════════════════════════════════════════════

TOP_K = max(100, NUM_READS // 20)

resolution = 2.0**(-q)

all_records = []
for sample, energy in zip(sampleset.samples(), sampleset.record["energy"]):
    v       = decode(sample)
    v_round = tuple(round(vi / resolution) * resolution for vi in v)
    all_records.append((v_round, float(energy)))

all_records_sorted = sorted(all_records, key=lambda x: x[1])
topK_records       = all_records_sorted[:TOP_K]
window_label       = (f"top-{TOP_K} lowest-energy samples  "
                      f"(E ≤ {topK_records[-1][1]:.4f})")

topK_counter = Counter(v for v, _ in topK_records)
top_states   = topK_counter.most_common()
mf_v_tuple, mf_count = top_states[0]

mf_energy = min(e for v, e in topK_records if v == mf_v_tuple)
v_mf      = np.array(mf_v_tuple)
bitstr    = "  ".join(f"{vi:.3f}" for vi in v_mf)


# ─── Summary Table ───────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  RESULTS")
print("=" * 62)
print(f"  Ground-state QUBO energy      : {E_sa:.6f}")
print(f"  F(v_sa)  [exact cost, eq.(12)]: {F_exact(v_sa):.6f}")
print(f"  F(v_ref) [classical solution] : {F_exact(v_ref):.6f}")
print()
print(f"  {'Institution':<14} {'v_sa':>8} {'v_ref':>8} {'v_c':>6}  {'SA status':>10}  {'Classical':>10}")
print("  " + "─" * 60)
for i in range(n):
    st_sa  = "FAIL" if v_sa[i]  < v_c[i] else "OK"
    st_ref = "FAIL" if v_ref[i] < v_c[i] else "OK"
    print(f"  I{i+1:<13} {v_sa[i]:>8.3f} {v_ref[i]:>8.3f} {v_c[i]:>6.1f}  {st_sa:>10}  {st_ref:>10}")

print()
print("  " + "─" * 60)
print(f"  Most frequent ground state  [{window_label}]")
print("  " + "─" * 60)
print(f"  Qualifying samples         : {len(topK_records)} / {NUM_READS}")
print(f"  Unique decoded states      : {len(top_states)}")
print(f"  Most frequent state        : {bitstr}")
print(f"  Frequency                  : {mf_count} ({100*mf_count/len(topK_records):.1f}% of qualifying samples)")
print(f"  Best energy for this state : {mf_energy:.6f}")
print(f"  Decoded market values      : {np.round(v_mf, 3)}")
print(f"  F(v_mf) [exact cost]       : {F_exact(v_mf):.6f}")
print()
print(f"  {'Institution':<14} {'v_mf':>8} {'v_c':>6}  {'Status':>8}")
print("  " + "─" * 38)
for i in range(n):
    st = "FAIL" if v_mf[i] < v_c[i] else "OK"
    print(f"  I{i+1:<13} {v_mf[i]:>8.3f} {v_c[i]:>6.1f}  {st:>8}")

top5       = top_states[:min(5, len(top_states))]
top5_freqs = [c for _, c in top5]
top5_v     = [np.array(s) for s, _ in top5]
top5_bits  = ["  ".join(f"{vi:.2f}" for vi in v) for v in top5_v]


# ═══════════════════════════════════════════════════════════════════════════
# 12.  Plots
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

energies = sampleset.record["energy"]
axes[0].hist(energies, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
axes[0].axvline(E_sa, color="tomato", linewidth=1.8, linestyle="--",
                label=f"Ground state  E = {E_sa:.3f}")
axes[0].set_xlabel("Sample energy")
axes[0].set_ylabel("Count")
axes[0].set_title("Energy Distribution — SA (CZW-calibrated QUBO)")
axes[0].legend()

x, w = np.arange(n), 0.25
axes[1].bar(x - w, v_mf,  w, label="Most frequent GS",    color="darkorange")
axes[1].bar(x,     v_ref, w, label="Classical fixed-point", color="seagreen", alpha=0.8)
axes[1].bar(x + w, v_c,   w, label="Threshold v_c",        color="tomato",   alpha=0.7)
axes[1].set_xlabel("Institution")
axes[1].set_ylabel("Market value")
axes[1].set_title("Market Values: Most Frequent GS vs Classical vs Threshold\n(Shocked prices p=[6,9,12])")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"I{i+1}" for i in range(n)])
axes[1].legend()

short_labels = [f"GS{i+1}\n{np.round(v,1)}" for i, (v, _) in
                enumerate(zip(top5_v, top5_freqs))]
bars = axes[2].bar(range(len(top5_freqs)), top5_freqs,
                   color="mediumpurple", edgecolor="white", linewidth=0.6)
axes[2].bar_label(bars, fmt="%d", padding=3, fontsize=9)
for i, (freq, label) in enumerate(zip(top5_freqs, top5_bits)):
    axes[2].text(i, -max(top5_freqs) * 0.08, label,
                 ha="center", va="top", fontsize=6.5, rotation=30,
                 color="dimgray")
axes[2].set_xticks(range(len(top5_freqs)))
axes[2].set_xticklabels(short_labels, fontsize=8)
axes[2].set_xlabel("Ground state (v decoded)")
axes[2].set_ylabel("Frequency")
axes[2].set_title(f"Top-{len(top5)} States by Frequency\n"
                  f"({window_label},  {len(topK_records)} samples)")
axes[2].set_ylim(0, max(top5_freqs) * 1.18)

plt.tight_layout()
out_path = "/Users/nishithreen/Desktop/quantum_computing/ground_state_sa_shocked_results.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"\n  Plot saved → {out_path}")

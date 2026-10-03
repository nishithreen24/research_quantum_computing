"""
XBK – optimised implementation
================================
Root cause of slowness: OpenFermion's QubitOperator uses pure-Python dicts
and __mul__ / __add__ create new objects on every call.  For construct_C with
n=9 that means millions of object allocations inside tight loops.

Fix: represent every operator as a plain Python dict
    { frozenset_of_Z_indices : complex_coefficient }
and do all arithmetic directly on those dicts (mul_ops / add_ops below).
QubitOperator is only constructed once at the very end.

Additional speedups
-------------------
* construct_C: the product ∏_i (½ + v_i/2 · Z_{i+n·j}) has an analytic
  closed form – its terms are exactly the 2^n subsets S⊆{0..n-1} with
  coefficient (1/2)^n · ∏_{i∈S} v[i].  We compute this with numpy
  broadcasting, skipping all Python loops over i.
* XBK_transform: same dict algebra, pauli_map lookup replaces the linear scan.
* XBK: p-loop parallelised with ProcessPoolExecutor (processes, not threads).
"""

import math, itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import dimod
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits

from helper_functions import *


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Pauli dict helpers
# Keys are frozensets of qubit indices (all-Z Pauli strings after XBK mapping).
# The empty frozenset represents the identity.
# ─────────────────────────────────────────────────────────────────────────────

def _mul_ops(a: dict, b: dict) -> dict:
    """Multiply two sparse Z-only Pauli dicts.
    Z_i·Z_i = I so the support of the product is the symmetric difference."""
    out = defaultdict(complex)
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka ^ kb] += va * vb        # frozenset XOR = symmetric difference
    return {k: v for k, v in out.items() if v != 0}


def _add_ops(a: dict, b: dict) -> dict:
    out = defaultdict(complex, a)
    for k, v in b.items():
        out[k] += v
    return {k: v for k, v in out.items() if v != 0}


def _scale_op(a: dict, s) -> dict:
    if s == 0:
        return {}
    return {k: v * s for k, v in a.items()}


def _op_to_qubit(d: dict) -> QubitOperator:
    """Convert sparse dict to QubitOperator – called once at the very end."""
    op = QubitOperator()
    for support, coeff in d.items():
        if coeff == 0:
            continue
        term_str = ' '.join(f'Z{q}' for q in sorted(support))
        op += QubitOperator(term_str, coeff)
    op.compress()
    return op


# ─────────────────────────────────────────────────────────────────────────────
# construct_C  –  fully vectorised, no QubitOperator inside any loop
# ─────────────────────────────────────────────────────────────────────────────

def _product_j_dict(v: np.ndarray, base: int, n: int) -> dict:
    """
    Compute  ∏_{i=0}^{n-1} (½ + v[i]/2 · Z_{base+i})  as a sparse dict.

    Closed form: the product expands to a sum over all 2^n subsets S⊆{0..n-1}:
        coeff of ∏_{i∈S} Z_{base+i}  =  (½)^n · ∏_{i∈S} v[i]

    All subset coefficients are computed in one numpy vectorised step.
    """
    masks      = np.arange(1 << n, dtype=np.int32)                         # (2^n,)
    bit_matrix = ((masks[:, None] >> np.arange(n)) & 1).astype(np.int8)    # (2^n, n)
    # For each mask: product of v[i] for each set bit i
    prods = np.prod(np.where(bit_matrix, v[np.newaxis, :], 1), axis=1)     # (2^n,)
    scale = 0.5 ** n

    result = {}
    for mask_int, prod in zip(masks.tolist(), prods.tolist()):
        coeff   = scale * prod
        support = frozenset(base + i for i in range(n) if (mask_int >> i) & 1)
        result[support] = result.get(support, 0) + coeff
    return result


def construct_C(n: int, r: int, p: int) -> QubitOperator:
    """
    Fast construction of the XBK penalty operator C.

    For each of the 2^n permutations v ∈ {±1}^n:
      term_v = (∑_{j} sign_j · product_j(v))²
    C = ∑_v term_v²

    All arithmetic is done in sparse dicts; QubitOperator built once at end.
    """
    perms  = np.array(list(itertools.product([1, -1], repeat=n)), dtype=np.int8)
    C_dict: dict = {}

    for k, v in enumerate(perms):
        print(k + 1)

        # term_dict = ∑_j  sign_j · product_j(v)
        term_dict: dict = {}
        for j in range(r):
            sign = -1 if j < p else 1
            pj   = _product_j_dict(v, base=n * j, n=n)
            term_dict = _add_ops(term_dict, _scale_op(pj, sign))

        # C += term_dict²
        C_dict = _add_ops(C_dict, _mul_ops(term_dict, term_dict))

    return _op_to_qubit(C_dict)


# ─────────────────────────────────────────────────────────────────────────────
# XBK_transform  –  dict-based, no QubitOperator inside the loop
# ─────────────────────────────────────────────────────────────────────────────

def XBK_transform(op: QubitOperator, r: int, p: int) -> QubitOperator:
    n        = count_qubits(op)
    new_dict: dict = {}

    for key, coeff in op.terms.items():
        pauli_map = {idx: char for idx, char in key}
        term_dict: dict = {}

        for j in range(r):
            for k in range(r):
                sign = 1 if (j < p) == (k < p) else -1
                sub: dict = {frozenset(): 1}   # identity
                zero = False

                for i in range(n):
                    char = pauli_map.get(i, 'I')
                    qi_j = i + n * j
                    qi_k = i + n * k

                    if char == 'X':
                        if j == k:
                            zero = True; break
                        factor = {frozenset(): 0.5,
                                  frozenset([qi_j, qi_k]): -0.5}
                        sub = _mul_ops(sub, factor)

                    elif char == 'Y':
                        if j == k:
                            zero = True; break
                        factor = {frozenset([qi_k]): 0.5j,
                                  frozenset([qi_j]): -0.5j}
                        sub = _mul_ops(sub, factor)

                    elif char == 'Z':
                        if j == k:
                            factor = {frozenset([qi_j]): 1}
                        else:
                            factor = {frozenset([qi_j]): 0.5,
                                      frozenset([qi_k]): 0.5}
                        sub = _mul_ops(sub, factor)

                    else:  # Identity
                        if j != k:
                            factor = {frozenset(): 0.5,
                                      frozenset([qi_j, qi_k]): 0.5}
                            sub = _mul_ops(sub, factor)
                        # j == k → multiply by 1, no-op

                if not zero:
                    term_dict = _add_ops(term_dict, _scale_op(sub, sign))

        new_dict = _add_ops(new_dict, _scale_op(term_dict, coeff))

    return _op_to_qubit(new_dict)


# ─────────────────────────────────────────────────────────────────────────────
# _xbk_single_p  –  worker for one p value (module-level → picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _xbk_single_p(args):
    p, n, qubit_H, qubit_C, sampler, starting_lam, num_samples, strength = args

    C_dict = convert_dict(qubit_C.terms)
    C_func = dict_to_func(C_dict)

    lam          = starting_lam
    min_energy   = -1.0
    ground_state = []
    cycles       = 0

    while min_energy < 0 and cycles < 10:
        H_prime    = qubit_H - lam * qubit_C
        bqm        = dimod.higherorder.utils.make_quadratic(
                         convert_dict(H_prime.terms), strength, dimod.SPIN)
        qubo, constant = bqm.to_qubo()

        if not qubo:
            break

        response  = sampler.sample_qubo(qubo, num_reads=num_samples)
        solutions = pd.DataFrame(response.data())

        index         = int(solutions[['energy']].idxmin())
        min_energy    = round(solutions['energy'][index], 14) + constant
        full_solution = solutions['sample'][index]

        solution = []
        for var in [f's{i}' for i in range(n)]:
            try:
                solution.append(2 * full_solution[var] - 1)
            except KeyError:
                solution.append(0)
                print(f'KeyError: {var}')

        try:
            sumBsq = int(C_func(*solution))
        except TypeError:
            sumBsq = int(C_func)

        if sumBsq == 0:
            cycles += 1
            break

        eigenvalue = lam + min_energy / sumBsq
        if min_energy < 0:
            lam          = eigenvalue
            ground_state = [(val + 1) // 2 for val in solution]
        cycles += 1

    return p, round(lam, 14), ground_state


# ─────────────────────────────────────────────────────────────────────────────
# XBK  –  p-loop parallelised across all CPU cores
# ─────────────────────────────────────────────────────────────────────────────

def XBK(qubit_Hs, qubit_Cs, r, sampler,
        starting_lam=0, num_samples=1000, strength=1e3,
        verbose=False, max_workers=None):
    """
    Parameters
    ----------
    max_workers : int or None
        Number of CPU processes.  None → os.cpu_count().
    """
    n       = count_qubits(qubit_Hs[0])
    p_range = range(int(math.ceil(r / 2 + 1)))

    args_list = [
        (p, n, qubit_Hs[p], qubit_Cs[p], sampler, starting_lam, num_samples, strength)
        for p in p_range
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_xbk_single_p, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            p, energy, gs = fut.result()
            results[p] = (energy, gs)
            if verbose:
                print(f'P: {p}  E: {round(energy, 5)}')

    min_energies  = [results[p][0] for p in p_range]
    ground_states = [results[p][1] for p in p_range]
    best          = min_energies.index(min(min_energies))

    if verbose:
        print(f'Energy: {round(min_energies[best], 5)}')

    return min_energies[best], ground_states[best]


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(construct_C(9, 4, 4 // 2))
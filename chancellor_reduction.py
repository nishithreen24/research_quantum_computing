from collections import defaultdict
import itertools
import dimod


# ------------------------------------------------------------
# Safe quadratic insertion (type independent)
# ------------------------------------------------------------
def add_quadratic(Q, u, v, value):
    if u == v:
        return
    key = frozenset((u, v))
    Q[key] += value


# ------------------------------------------------------------
# Chancellor gadget for one k-local term
# ------------------------------------------------------------
def chancellor_reduce_term(term, coeff, strength):
    """
    Convert a k-local monomial coeff * Π s_i
    into quadratic form using symmetric ancilla gadget
    """

    bits = sorted(list(term), key=str)   # deterministic ordering
    k = len(bits)

    linear = defaultdict(float)
    quadratic = defaultdict(float)
    offset = 0.0

    # constants
    if k == 0:
        offset += coeff
        return linear, quadratic, offset

    if k == 1:
        linear[bits[0]] += coeff
        return linear, quadratic, offset

    if k == 2:
        add_quadratic(quadratic, bits[0], bits[1], coeff)
        return linear, quadratic, offset

    # --------------------------------------------------------
    # multi-body gadget
    # --------------------------------------------------------
    J = strength

    # ancilla spins (counting register)
    ancillas = [f"anc_{'_'.join(map(str,bits))}_{i}" for i in range(k)]

    # logical-logical couplings
    for i, j in itertools.combinations(bits, 2):
        add_quadratic(quadratic, i, j, J)

    # logical-ancilla couplings
    for a in ancillas:
        for i in bits:
            add_quadratic(quadratic, i, a, -2*J)

    # ancilla local fields encode Hamming weight
    for idx, a in enumerate(ancillas):
        linear[a] += J*(2*idx - k + 1)

    # energy shift encodes original coefficient
    offset += coeff

    return linear, quadratic, offset


# ------------------------------------------------------------
# Main reduction: HUBO → BQM
# ------------------------------------------------------------
def make_quadratic_chancellor(poly, strength=5.0, vartype=dimod.SPIN):
    """
    Replacement for dimod.make_quadratic using Chancellor reduction

    Input:
        poly : dimod.BinaryPolynomial or dict
        strength : gadget penalty strength

    Output:
        dimod.BinaryQuadraticModel
    """

    linear = defaultdict(float)
    quadratic = defaultdict(float)
    offset = 0.0

    # iterate polynomial
    for term, coeff in poly.items():

        l, q, o = chancellor_reduce_term(term, coeff, strength)

        for k,v in l.items():
            linear[k] += v

        for k,v in q.items():
            quadratic[k] += v

        offset += o

    # convert frozenset keys → tuple keys
    quad_final = {}
    for key, val in quadratic.items():
        u, v = tuple(key)
        quad_final[(u,v)] = val

    return dimod.BinaryQuadraticModel(dict(linear), quad_final, offset, vartype=vartype)
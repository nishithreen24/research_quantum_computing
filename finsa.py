import numpy as np
import matplotlib.pyplot as plt
"""
import neal

sampler = neal.SimulatedAnnealingSampler()

h = {0: -1, 1: -1}
J = {(0, 1): -1}
sampleset = sampler.sample_ising(h, J)

print(sampleset)
"""
"""
import neal
from dimod import BinaryQuadraticModel
bqm = BinaryQuadraticModel({0:1, 1:1, 2:1, 3:1}, {(0, 1):1, (1, 0):1, (1,2): 1, (2, 1):1, (2, 3):1, (3, 2):1, (0, 3):1, (3, 0):1}, vartype='SPIN')
sampler = neal.SimulatedAnnealingSampler()
print(sampler.sample(bqm, num_reads=16, num_sweeps=100000))
"""
"""
from collections import defaultdict
HUBO = defaultdict(float)
def add_monomial(HUBO, bits, coeff):
    key = tuple(sorted(bits))
    HUBO[key] += coeff

add_monomial(HUBO, [1, 3], 3)
add_monomial(HUBO, [2, 2], 2)
add_monomial(HUBO, [1], 1)
add_monomial(HUBO, [0, 2 , 1], 8)
print(HUBO)
"""
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
"""
n = 5
m = 3
q = 2
p = 0.9*np.array([10, 15, 20])
D = np.array(([0.3, 0.2, 0.1],
              [0.2, 0.3, 0.2],
              [0.1, 0.2, 0.3],
              [0.2, 0.1, 0.2],
              [0.2, 0.2, 0.2]))
C = np.array(([0.0, 0.1, 0.05, 0.0, 0.0],
              [0.1, 0.0, 0.1, 0.05, 0.0],
              [0.05, 0.1, 0.0, 0.1, 0.05],
              [0.0, 0.05, 0.1, 0.0, 0.1],
              [0.0, 0.0, 0.05, 0.1, 0.0]))

Cbar = np.zeros(n)

for i in range(n):
    sum = 0
    for j in range(n):
        sum += C[i][j]
    Cbar[i] = 1 - sum

Cbar = np.diag(Cbar)

v_c = np.array([5, 6, 5.5, 6, 5])
beta = np.array([2, 2.5, 2, 2.5, 2])
"""
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
n = 20
m = 10
q = 3

p, D, C, Cbar, v_c, beta = generate_financial_system(n, m, seed=42)
HUBO = defaultdict(float)

"""
def market_value(P):
    return Cbar @ np.linalg.inv((np.eye(n) - C)) @ D @ P


def floating(V):
    bitstring = []
    for i in range(n):
        if (V[i] > v_c[i]):
            bitstring.append("Afloat")
        else:
            bitstring.append("Failed")
    return bitstring


def failure(V):
    b = np.zeros(n)
    # print(V)
    # print(v_c)
    for i in range(n):
        # print(np.heaviside(V[i]-v_c[i], 1))
        b[i] = beta[i] * (1 - np.heaviside(V[i] - v_c[i], 1))
    return b


def market_value_wf(P):
    V = market_value(P)
    b = failure(V)
    return Cbar @ np.linalg.inv((np.eye(n) - C)) @ ((D @ P) - b)


def F(V):
    return np.linalg.norm(V - Cbar @ (np.linalg.inv(np.eye(n) - C)) @ (D @ p - failure(V)))


# print(F(v))
# print(F(v_nu))
# print(F(v_updated))
# print(F(v_final))

def binary_encode(V, Q):
    # rng = 0
    # res = np.pow(2.0, -Q)
    # for alpha in range(-Q, Q+1):
    # print(np.pow(2.0, alpha))
    #    rng += np.pow(2.0, alpha)
    # bits = np.zeros(2*Q + 1)
    # print("Range: ", rng)
    # print("Resolution: ", res)
    # if(V>rng):
    #    return np.ones(2*Q+1)
    # else:
    bitstring = np.zeros(2 * Q + 1)
    temp = V
    for alpha in range(Q, -Q - 1, -1):
        # print(temp, np.power(2.0, alpha))
        # print(Q-alpha)
        if (temp > np.power(2.0, alpha)):
            bitstring[Q - alpha] = 1
            temp = temp - np.power(2.0, alpha)
        else:
            bitstring[Q - alpha] = 0
    return bitstring


def max_v(Q):
    rng = 0
    for alpha in range(-Q, Q + 1):
        # print(np.pow(2.0, alpha))
        rng += np.pow(2.0, alpha)
    return rng


def fourier_legendre_expansion(x, r):
    sum = 0.5
    for l in range(1, r + 1):
        sum += (np.polynomial.legendre.legval(0, [0] * (l - 1) + [1]) - np.polynomial.legendre.legval(0,
                                                                                                      [0] * (l + 1) + [
                                                                                                          1])) * 0.5 * np.polynomial.legendre.legval(
            x, [0] * l + [1])
    return sum


# print(floating(v))

def polynomial_failure(V, Q):
    b = np.zeros(n)
    for i in range(n):
        b = beta[i] * (1 - fourier_legendre_expansion((V[i] - v_c[i]) / max_v(Q), 50))
    return b


def pauli_string(BITSTRING):
    paulis = np.zeros(len(BITSTRING))
    for i in range(len(BITSTRING)):
        paulis[i] = 2 * BITSTRING[i] - 1
    return paulis

"""
def bits_to_val(bitstring, Q):
    value = 0
    for counter in range(0, 2 * Q + 1):
        value += bitstring[counter] * (2 ** (counter - Q))
    return value


def bit_index(inst, alpha, Q):
    return inst * (2 * Q + 1) + Q + alpha
"""

def vi_squared(Q):
    return


def gamma_squared():
    return


def fail_squared(Q, r):
    return


def gamma_vi(Q):
    return


def gamma_fail(Q, r):
    return


def vi_vail(Q, r):
    return

"""
def spin_weight(alpha):
    return 2.0**alpha
"""
def expand_vi(inst, Q):
    terms = {}
    for alpha in range(-Q, Q + 1):
        i = bit_index(inst, alpha, Q)
        terms[(i,)] = 2 ** alpha
    return terms
"""
def expand_vi(inst, Q):
    """
    v_i = Σ w_α s_{i,α}
    returns polynomial dict representation
    """
    terms = {}
    for alpha in range(-Q, Q+1):
        idx = bit_index(inst, alpha, Q)
        terms[(idx,)] = spin_weight(alpha)
    return terms


def add_term(source, target, scale=1.0):
    for bits, coeffs in source.items():
        target[bits] += scale * coeffs
    return

"""
def multi_poly(A, B):
    result = defaultdict(float)
    for bitsA, coeffsA in A.items():
        for bitsB, coeffsB in B.items():
            newbits = tuple(sorted(bitsA + bitsB))
            result[newbits] += coeffsA * coeffsB
    return result
"""
def multi_poly(A, B):
    result = defaultdict(float)

    for bitsA, coeffA in A.items():
        for bitsB, coeffB in B.items():

            new = list(bitsA) + list(bitsB)

            # spin cancellation rule: s_i^2 = 1
            counts = {}
            for b in new:
                counts[b] = counts.get(b, 0) + 1

            final_bits = tuple(sorted(k for k,v in counts.items() if v % 2 == 1))

            result[final_bits] += coeffA * coeffB

    return result

def hubo_to_ising(H):
    """
    Convert polynomial dict to dimod SPIN BQM if quadratic
    """
    import dimod
    h = defaultdict(float)
    J = defaultdict(float)
    offset = 0

    for term, coeff in H.items():
        if len(term) == 0:
            offset += coeff
        elif len(term) == 1:
            h[term[0]] += coeff
        elif len(term) == 2:
            u,v = sorted(term)
            J[(u,v)] += coeff
        else:
            raise ValueError("Still higher order → needs reduction")

    return dimod.BinaryQuadraticModel(h, J, offset, vartype=dimod.SPIN)


def square_for_HUBO(Ri, HUBO):
    for bits1, coeffs1 in Ri.items():
        for bits2, coeffs2 in Ri.items():
            HUBO[(tuple(sorted(bits1 + bits2)))] += coeffs1 * coeffs2


def poly_dict(const):
    result = {}
    result[()] = const
    return result


def Ri_linear(i, Q, gamma):
    Ri = defaultdict(float)

    add_term(source=expand_vi(i, Q), target=Ri)

    Ri[()] += -gamma[i]
    return Ri


def poly_power(poly, k):
    if k == 0:
        return {(): 1.0}
    result = poly
    for _ in range(k - 1):
        result = multi_poly(result, poly)
    return result


def failure_poly(i, Q, r):
    result = defaultdict(float)
    vi = expand_vi(i, Q)
    for l in range(1, r + 1):
        coeff = 0.5 * (
                np.polynomial.legendre.legval(0, [0] * (l - 1) + [1])
                - np.polynomial.legendre.legval(0, [0] * (l + 1) + [1])
        )
        scale = -beta[i] * coeff
        vj_power = poly_power(vi, l)
        add_term(vj_power, result, scale=scale)
    return result


def full_Ri(i, Q, r):
    Ri = defaultdict(float)
    vi = expand_vi(i, Q)
    add_term(vi, Ri)
    A = np.linalg.inv(np.eye(n)-C)
    sumc = 0
    sumd = 0
    for j in range(n):
        sumc += A[i][j]
    for k in range(m):
        sumd += D[i][k]*p[k]

    Ri[()] += -(Cbar[i][i]*sumc*sumd)#-v_c[i]
    A = Cbar @ np.linalg.inv(np.eye(n) - C)
    for j in range(n):
        bj_poly = failure_poly(j, Q, r)
        add_term(Ri, bj_poly, scale=A[i, j])
    return Ri

def decode_values(bitstring, q):
    v = np.zeros(n)
    for i in range(n):
        for alpha in range(-q, q+1):
            v[i] += bitstring[i*(2*q+1) + q+alpha]* (2**alpha)
    return v

import dimod
from collections import defaultdict
import itertools

import dimod
from collections import defaultdict

def make_quadratic_czw(poly, strength=None, vartype=dimod.BINARY):

    if strength is None:
        strength = 2 * max(abs(v) for v in poly.values()) if poly else 1.0

    linear = defaultdict(float)
    quadratic = defaultdict(float)
    offset = 0.0

    # determine next available variable index
    max_var = -1
    for term in poly:
        if len(term) > 0:
            max_var = max(max_var, max(term))
    next_var = max_var + 1

    def new_var():
        nonlocal next_var
        v = next_var
        next_var += 1
        return v

    def add_pair(u, v, bias):
        if u == v:
            linear[u] += bias
        else:
            quadratic[tuple(sorted((u, v)))] += bias

    # ------------------------------
    # Main reduction loop
    # ------------------------------
    for term_set, coeff in poly.items():

        term = tuple(term_set)   # <-- canonical conversion

        if len(term) == 0:
            offset += coeff
            continue

        if len(term) == 1:
            linear[term[0]] += coeff
            continue

        if len(term) == 2:
            add_pair(term[0], term[1], coeff)
            continue

        # Higher-order term → mediator reduction
        vars_list = list(term)

        while len(vars_list) > 2:
            x1 = vars_list.pop()
            x2 = vars_list.pop()
            x3 = vars_list.pop()

            a = new_var()
            Δ = strength

            # (3a - x1 - x2 - x3)^2 gadget
            linear[a] += 9*Δ
            linear[x1] += Δ
            linear[x2] += Δ
            linear[x3] += Δ

            add_pair(a, x1, -6*Δ)
            add_pair(a, x2, -6*Δ)
            add_pair(a, x3, -6*Δ)

            add_pair(x1, x2, 2*Δ)
            add_pair(x1, x3, 2*Δ)
            add_pair(x2, x3, 2*Δ)

            vars_list.append(a)

        u, v = vars_list
        add_pair(u, v, coeff)

    return dimod.BinaryQuadraticModel(dict(linear), dict(quadratic), offset, vartype)


from chancellor_reduction import make_quadratic_chancellor

HUBO = defaultdict(float)

for i in range(n):
    Ri = full_Ri(i, Q=q, r=10)
    Ri_sq = multi_poly(Ri, Ri)
    add_term(Ri_sq, HUBO)

import dimod
import neal
from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler

#ground_states = []
#ground_state_energies = []
poly = dimod.BinaryPolynomial(HUBO, vartype="BINARY")
strength = 2 * max(abs(v) for v in HUBO.values()) if HUBO else 1.0
#bqm = dimod.make_quadratic(poly, strength=strength, vartype=dimod.BINARY)
#bqm = make_quadratic_czw(poly, strength=strength, vartype=dimod.BINARY)
bqm = make_quadratic_chancellor(poly, strength=strength)
sampler = SimulatedAnnealingSampler()
#for i in range(10000):
sampleset = sampler.sample(bqm, num_reads=1000, num_sweeps=10000)
#ground = sampleset.lowest()
#record = sampleset.record[0]
#lowest_sample = ground.lowest()
#print(record.sample)
#print(record.energy)
# print(HUBO)
# print(max(len(k) for k in HUBO))
    #print(sampleset.record)
    #if sampleset.record[0][1]==0.0:
#ground_states.append(sampleset.record[0][0])
    #ground_state_energies.append(sampleset.record[0][1])
ground_states = sampleset.record.sample
ground_states_energy = sampleset.record.energy


def count_array_frequencies(array_list):
    from collections import Counter

    # Convert arrays to tuples so they're hashable
    tuple_list = [tuple(arr) for arr in array_list]

    # Count frequencies
    frequencies = Counter(tuple_list)

    # Convert back to arrays if desired
    result = {tuple(arr): count for arr, count in frequencies.items()}

    return result

data = count_array_frequencies(ground_states)
keys = [str(k) for k in data.keys()]
values = list(data.values())

max_key = max(data, key=data.get)
max_count = data[max_key]
max_array = np.array(max_key)

#print(len(max_array))

print(f"Most frequent bitstring: {max_array}")
v = decode_values(max_array, q = q)
print(f"Frequency: {max_count}")
#print(len(data))
print(v)
print(v_c)
#print([np.array(k) for k in data.keys()])

plt.bar(keys, values)
plt.xlabel('Bitstrings')
plt.ylabel('Frequency')
plt.title('Frequencies')
plt.xticks(rotation=45)
#plt.tight_layout()
plt.show()
#print(ground_states)
# fail = defaultdict(float)
# add_term(source=expand_vi(0, 2), target=fail, scale=1.0)
# print(poly_power(fail, 2))

plt.bar(range(n), v, alpha=0.5, color='red')
plt.bar(range(n), v_c, alpha = 0.3, color='blue')
plt.legend(("Equilibrium Market Values", "Critical Market Values"))
plt.xlabel('Institutions')
plt.ylabel('Value')
plt.show()

# HUBO = defaultdict(float)
# print(HUBO)
# Ri = Ri_linear(0, Q=2, gamma=v_c)
# rint(Ri)
# Ri_sq = multi_poly(Ri, Ri)
# print(Ri_sq)
# add_term(Ri_sq, HUBO)
# print(HUBO)

# HUBO1 = defaultdict(float)
# HUBO2 = defaultdict(float)

# add_term(expand_vi(0, 2), HUBO1, 1)
# HUBO1[()] += -5
# print(HUBO1)

# add_term(expand_vi(0, 2), HUBO2, 3)
# HUBO2[()] += 2
# print(HUBO2)

# print(multi_poly(HUBO1, HUBO2))

# print(expand_vi(2, 2))
# Reduce higher-order HUBO terms to a quadratic BQM (Rosenberg reduction)

"""
v = Cbar @ np.linalg.inv((np.eye(n)-C)) @ D @ p


p_nu = 0.6*p

v_nu =  Cbar @ np.linalg.inv((np.eye(n)-C)) @ D @ p_nu
nu_bitstring = floating(v_nu)

#print(v_nu)
#print(nu_bitstring)


b = failure(v_nu)
#print(D @ p_nu - b)
v_updated = Cbar @ np.linalg.inv((np.eye(n)-C)) @ ((D @ p_nu) - b)
#print(v_updated)

#print(floating(v_updated))

b_it = failure(v_updated)
#print(b_it)
#print(D @ p_nu - b_it)

v_final = Cbar @ np.linalg.inv((np.eye(n)-C)) @ ((D @ p_nu) - b_it)
#print(v_final)
#print(floating(v_final))

#print(v[0])
#print(binary_encode(v[0], 3))

#print(pauli_string(binary_encode(v[0], 3)))

#print(v_nu[0])
#print(v_c[0])  
v_max = max_v(3)
#print((v_final[0]-v_c[0])/v_max)
#print(fourier_legendre_expansion((v_final[0]-v_c[0])/v_max, 1000))

b_poly = polynomial_failure(v, 3)
#print(b_poly)
#def G(x):
    #np.power(np.linalg.norm(5), 2)
"""
max_degree = max(len(term) for term in HUBO.keys())
print("Maximum HUBO degree =", max_degree)
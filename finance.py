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
from collections import defaultdict, Counter
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

n = 5
m = 3
q = 3
p = 0.5*np.array([10, 15, 20])
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
def expand_vi(inst, Q):
    terms = {}
    for alpha in range(-Q, Q + 1):
        i = bit_index(inst, alpha, Q)
        terms[(i,)] = 2 ** alpha
    return terms


def add_term(source, target, scale=1.0):
    for bits, coeffs in source.items():
        target[bits] += scale * coeffs
    return


def multi_poly(A, B):
    result = defaultdict(float)
    for bitsA, coeffsA in A.items():
        for bitsB, coeffsB in B.items():
            newbits = tuple(sorted(bitsA + bitsB))
            result[newbits] += coeffsA * coeffsB
    return result


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



HUBO = defaultdict(float)

for i in range(n):
    Ri = full_Ri(i, Q=q, r=10)
    Ri_sq = multi_poly(Ri, Ri)
    add_term(Ri_sq, HUBO)
print(HUBO)
import dimod
import neal
from dwave.samplers import PathIntegralAnnealingSampler
from qanneal import *
#ground_states = []
#ground_state_energies = []
#print(HUBO)
poly = dimod.BinaryPolynomial(HUBO, vartype="BINARY")
#print(poly)
strength = 2 * max(abs(v) for v in HUBO.values()) if HUBO else 1.0
bqm = dimod.make_quadratic(poly, strength=strength, vartype=dimod.BINARY)
#print(bqm)
#sampler = PathIntegralAnnealingSampler()
#for i in range(10000):
#sampleset = sampler.sample(bqm, num_reads=1000, num_sweeps=10000)
#qubo = QUBO(bqm)
#ising = qubo.to_ising()
#ising = bqm.to_ising()
#print(type(ising))
#h, J, offset = bqm.to_ising()
#h#amiltonian = Hamiltonian(h, J)

qubo = QUBO(bqm)
ising = qubo.to_ising()

betas = [0.2, 0.5, 1.0, 1.5]
gammas = [2.0, 1.5, 1.0, 0.5]
obs = SQAStateTraceObserver()
obs.stride = 1
schedule = SQASchedule.from_vectors(betas, gammas)
sample = SQAAnnealer(ising, schedule, trotter_slices=8, replicas=2)
result = sample.run(sweeps_per_beta=10, worldline_sweeps=2, observer=obs)
print(result.best_energy)
print(obs.replica_energy_trace)
from collections import Counter
# Histogram of all recorded SQA states (all replicas/slices across all records).
if obs.state_trace:
    states = np.array(obs.state_trace, dtype=np.int8)
    states = states.reshape(len(obs.state_trace), obs.replicas, obs.slices, obs.spins)
    flat_states = states.reshape(-1, obs.spins)
    counts = Counter(map(tuple, flat_states))
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_items = items[:10]
    labels = ["".join("1" if s == 1 else "0" for s in state) for state, _ in top_items]
    values = [count for _, count in top_items]

    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xlabel("State (bitstring)")
    plt.ylabel("Frequency")
    plt.title("SQA State Histogram (top 10 recorded states)")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()

    energy_by_state = {state: ising.energy(state) for state, _ in items}
    min_energy = min(energy_by_state.values())
    min_states = [s for s, e in energy_by_state.items() if np.isclose(e, min_energy)]
    best_min_state = max(min_states, key=lambda s: counts[s])
    spin_str = "".join("+" if s == 1 else "-" for s in best_min_state)
    bit_str = "".join("1" if s == 1 else "0" for s in best_min_state)
    print(f"Most frequent lowest-energy state (spins): {spin_str}")
    print(f"Most frequent lowest-energy state (bitstring): {bit_str}")
    print(f"Lowest energy: {min_energy} (frequency {counts[best_min_state]})")

    spins = np.array(best_min_state, dtype=np.int8)
    bits = (spins + 1) // 2
    v_decoded = decode_values(bits, q=q)
    print(f"Decoded v: {v_decoded}")

    plt.figure()
    x = np.arange(n)
    plt.bar(x - 0.2, v_decoded, width=0.4, label="v (decoded)")
    plt.bar(x + 0.2, v_c, width=0.4, label="v_c")
    plt.xlabel("Institution")
    plt.ylabel("Value")
    plt.title("Decoded v vs v_c (from most frequent lowest-energy state)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No states recorded in obs.state_trace; increase obs.stride or pass observer.")
#print(sampleset.record.energy)
"""
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

print(len(max_array))

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
"""
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

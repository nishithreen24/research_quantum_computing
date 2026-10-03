import numpy as np
from collections import defaultdict, Counter
import dimod
import neal
import matplotlib.pyplot as plt

def generate_system(n, m, density=0.25, seed=0):
    rng = np.random.default_rng(seed)

    # external assets
    p = rng.uniform(5, 20, size=m)

    # holdings matrix
    D = rng.uniform(0, 1, size=(n, m))
    D = D / (1.5 * D.sum(axis=1, keepdims=True))

    # cross holdings
    C = rng.uniform(0, 1, size=(n, n))
    mask = rng.random((n, n)) < density
    C *= mask
    np.fill_diagonal(C, 0)

    # ensure spectral radius < 1
    C = C / (1.2 * np.maximum(C.sum(axis=1, keepdims=True), 1))

    Cbar = np.diag(1 - C.sum(axis=1))

    # nominal values
    A = Cbar @ np.linalg.inv(np.eye(n) - C) @ D
    v_nominal = A @ p

    # default thresholds
    v_c = 0.7 * v_nominal

    # default penalties
    beta = 0.15 * v_nominal

    return p, D, C, Cbar, v_c, beta

def bit_index(inst, alpha, Q):
    return inst*(2*Q+1) + (alpha+Q)

def expand_vi(inst, Q):
    poly = {}
    for alpha in range(-Q, Q+1):
        poly[(bit_index(inst, alpha, Q),)] = 2.0**alpha
    return poly

def multiply_terms(t1, t2):
    counts = {}
    for x in t1+t2:
        counts[x] = counts.get(x,0)+1
    return tuple(sorted(k for k,v in counts.items() if v%2==1))

def poly_add(A,B,scale=1):
    R = defaultdict(float)
    for k,v in A.items(): R[k]+=v
    for k,v in B.items(): R[k]+=scale*v
    return R

def poly_mul(A,B):
    R = defaultdict(float)
    for t1,c1 in A.items():
        for t2,c2 in B.items():
            R[multiply_terms(t1,t2)] += c1*c2
    return R

def poly_power(P,k):
    if k==0: return {():1.0}
    R=P
    for _ in range(k-1):
        R=poly_mul(R,P)
    return R

def legendre_heaviside_poly(vi_poly, vc, Q, beta_i, order=8):

    result = defaultdict(float)

    scale = sum(2.0**a for a in range(-Q,Q+1))

    # (v_i - v_c)/scale
    shifted = poly_add(vi_poly,{():-vc})
    shifted = {k:v/scale for k,v in shifted.items()}

    # constant term
    result[()] += beta_i*0.5

    for l in range(1,order+1):
        coeff = 0.5*(np.polynomial.legendre.legval(0,[0]*(l-1)+[1])
                    -np.polynomial.legendre.legval(0,[0]*(l+1)+[1]))

        Pl = poly_power(shifted,l)
        for k,v in Pl.items():
            result[k] += -beta_i*coeff*v

    return result

def build_residual(i,Q,r,p,D,C,Cbar,v_c,beta):

    n=len(v_c)
    A=Cbar @ np.linalg.inv(np.eye(n)-C)

    Ri=expand_vi(i,Q)

    # constant term
    const = sum(A[i,j]*sum(D[j,k]*p[k] for k in range(len(p))) for j in range(n))
    Ri=poly_add(Ri,{():-const})

    # default terms
    for j in range(n):
        bj=legendre_heaviside_poly(expand_vi(j,Q),v_c[j],Q,beta[j],r)
        for k,v in bj.items():
            Ri[k]-=A[i,j]*v

    return Ri

def build_HUBO(n,Q,r,p,D,C,Cbar,v_c,beta):

    HUBO=defaultdict(float)

    for i in range(n):
        Ri=build_residual(i,Q,r,p,D,C,Cbar,v_c,beta)
        Ri2=poly_mul(Ri,Ri)
        for k,v in Ri2.items():
            HUBO[k]+=v

    return HUBO

def print_degree_info(H):
    deg=Counter(len(k) for k in H)
    print("\nDegree distribution:")
    for d in sorted(deg):
        print(d,deg[d])
    print("Max degree:",max(deg))

def reduce_to_bqm(H,strength=5):

    poly=dimod.BinaryPolynomial(H,vartype=dimod.SPIN)

    bqm=dimod.make_quadratic(poly,strength=strength,vartype=dimod.SPIN)
    return bqm

def solve_bqm(bqm,reads=500,sweeps=10000):
    sampler=neal.SimulatedAnnealingSampler()
    return sampler.sample(bqm,num_reads=reads,num_sweeps=sweeps)

def decode_values(sample,n,Q):
    v=np.zeros(n)
    for i in range(n):
        for a in range(-Q,Q+1):
            v[i]+=sample[bit_index(i,a,Q)]*(2.0**a)
    return v


def add_polynomial_square_to_bqm(poly, bqm, strength):

    from chancellor_reduction import chancellor_reduce_term

    items = list(poly.items())

    for i in range(len(items)):
        termA, coeffA = items[i]

        for j in range(i, len(items)):
            termB, coeffB = items[j]

            coeff = coeffA * coeffB
            if i != j:
                coeff *= 2

            # spin multiplication rule s^2=1
            counts = {}
            for x in termA + termB:
                counts[x] = counts.get(x,0)+1
            new_term = tuple(sorted(k for k,v in counts.items() if v%2==1))

            l,q,o = chancellor_reduce_term(new_term, coeff, strength)

            for k,v in l.items():
                bqm.add_variable(k, v)
            for key,v in q.items():
                u,vv = tuple(key)
                bqm.add_interaction(u,vv,v)
            bqm.offset += o



n,m=5,3
Q=2
r=2

p,D,C,Cbar,v_c,beta=generate_system(n,m)

#HUBO=build_HUBO(n,Q,r,p,D,C,Cbar,v_c,beta)
#print_degree_info(HUBO)
bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)

for i in range(n):
    Ri = build_residual(i,Q,r,p,D,C,Cbar,v_c,beta)
    add_polynomial_square_to_bqm(Ri, bqm, strength=5)
#bqm=reduce_to_bqm(HUBO,5)
from chancellor_reduction import make_quadratic_chancellor
#poly = dimod.BinaryPolynomial(HUBO, vartype="SPIN")
#strength = 2 * max(abs(v) for v in HUBO.values()) if HUBO else 1.0
#bqm = make_quadratic_chancellor(poly, strength=strength)
sampleset=solve_bqm(bqm)

best=sampleset.first.sample
v=decode_values(best,n,Q)

print("\nRecovered equilibrium:")
print(v)
print("\nCritical values:")
print(v_c)
print("Variables:", bqm.num_variables)
print("Interactions:", len(bqm.quadratic))
print("Energy scale:", max(abs(v) for v in list(bqm.linear.values())+list(bqm.quadratic.values())))

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

print(f"Most frequent bitstring: {max_array}")
v = decode_values(max_array, n, Q = Q)
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
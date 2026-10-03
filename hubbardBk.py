import itertools
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
from OpenFermion import FermionOperator, jordan_wigner, bravyi_kitaev

# ============================================================
# 1. LATTICE + FERMIONIC t–V HAMILTONIAN (SPINLESS)
# ============================================================

Lx, Ly = 3, 2
sites = [(x, y) for y in range(Ly) for x in range(Lx)]
site_index = {sites[i]: i for i in range(len(sites))}
n_qubits = len(sites)

neighbors = []
for (x, y), i in site_index.items():
    for dx, dy in [(1, 0), (0, 1)]:
        if (x + dx, y + dy) in site_index:
            neighbors.append((i, site_index[(x + dx, y + dy)]))

t, V = 1.0, 1.0
Hf = FermionOperator()

for i, j in neighbors:
    Hf += -t * (
        FermionOperator(((i, 1), (j, 0))) +
        FermionOperator(((j, 1), (i, 0)))
    )
    Hf += V * FermionOperator(((i, 1), (i, 0), (j, 1), (j, 0)))

# ============================================================
# 2. JORDAN–WIGNER AND BRAVYI–KITAEV
# ============================================================

HJW = jordan_wigner(Hf)
HBK = bravyi_kitaev(Hf)

# ============================================================
# 3. SAVE FULL HAMILTONIANS
# ============================================================

with open("2x3_spinless_tV_full_hamiltonians.txt", "w") as f:
    f.write("=== Fermionic Hamiltonian ===\n")
    f.write(str(Hf))
    f.write("\n\n=== Jordan–Wigner Hamiltonian ===\n")
    f.write(str(HJW))
    f.write("\n\n=== Bravyi–Kitaev Hamiltonian ===\n")
    f.write(str(HBK))

# ============================================================
# 4. PAULI HYPERGRAPH EXTRACTION
# ============================================================

def extract_hyperedges(qubit_op):
    """Each Pauli string support is a hyperedge."""
    return [
        frozenset(q for q, _ in term)
        for term in qubit_op.terms
        if len(term) > 0
    ]

HG_JW = extract_hyperedges(HJW)
HG_BK = extract_hyperedges(HBK)

# ============================================================
# 5. HYPEREDGE RANK DISTRIBUTION
# ============================================================

def rank_distribution(hyperedges):
    return Counter(len(e) for e in hyperedges)

# ============================================================
# 6. TREEWIDTH PROXY (2-SECTION + MIN-FILL)
# ============================================================

def two_section_graph(hyperedges):
    G = nx.Graph()
    for e in hyperedges:
        for i, j in itertools.combinations(e, 2):
            G.add_edge(i, j)
    return G

def treewidth_proxy(hyperedges):
    G = two_section_graph(hyperedges)
    width, _ = nx.approximation.treewidth_min_fill_in(G)
    return width

# ============================================================
# 7. PENALTY LOWER-BOUND ESTIMATE
# ============================================================

def penalty_lower_bounds(hyperedges):
    penalties = []
    for i, e in enumerate(hyperedges):
        overlap = sum(
            len(e & f)
            for j, f in enumerate(hyperedges)
            if i != j
        )
        penalties.append(len(e) + overlap)
    return {
        "max": max(penalties),
        "mean": sum(penalties) / len(penalties),
    }

# ============================================================
# 8. COMPUTE METRICS
# ============================================================

metrics = {
    "JW": {
        "rank": rank_distribution(HG_JW),
        "treewidth": treewidth_proxy(HG_JW),
        "penalty": penalty_lower_bounds(HG_JW),
    },
    "BK": {
        "rank": rank_distribution(HG_BK),
        "treewidth": treewidth_proxy(HG_BK),
        "penalty": penalty_lower_bounds(HG_BK),
    },
}

print("=== METRICS ===")
for k, v in metrics.items():
    print(k, v)

# ============================================================
# 9. PLOTTING (SAFE FOR NON-INTERACTIVE BACKENDS)
# ============================================================

def plot_rank_hist(rank_jw, rank_bk):
    xs = sorted(set(rank_jw) | set(rank_bk))
    jw_vals = [rank_jw.get(x, 0) for x in xs]
    bk_vals = [rank_bk.get(x, 0) for x in xs]

    plt.figure()
    plt.bar(xs, jw_vals, alpha=0.7, label="JW")
    plt.bar(xs, bk_vals, alpha=0.7, label="BK")
    plt.xlabel("Hyperedge rank |e|")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rank_distribution_JW_vs_BK.png")
    plt.close()

def plot_scalar(title, jw_val, bk_val, ylabel, fname):
    plt.figure()
    plt.bar(["JW", "BK"], [jw_val, bk_val])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

plot_rank_hist(metrics["JW"]["rank"], metrics["BK"]["rank"])

plot_scalar(
    "Treewidth proxy",
    metrics["JW"]["treewidth"],
    metrics["BK"]["treewidth"],
    "treewidth",
    "treewidth_proxy.png",
)

plot_scalar(
    "Penalty lower bound (max)",
    metrics["JW"]["penalty"]["max"],
    metrics["BK"]["penalty"]["max"],
    "λ",
    "penalty_max.png",
)

plot_scalar(
    "Penalty lower bound (mean)",
    metrics["JW"]["penalty"]["mean"],
    metrics["BK"]["penalty"]["mean"],
    "λ",
    "penalty_mean.png",
)


def plot_incidence_graph(hyperedges, title, fname):
    """
    Bipartite graph:
    - Left: qubits
    - Right: Pauli terms (hyperedges)
    """
    B = nx.Graph()

    qubits = set().union(*hyperedges)

    # Add nodes
    for q in qubits:
        B.add_node(f"q{q}", bipartite=0)

    for i, e in enumerate(hyperedges):
        term_node = f"t{i}"
        B.add_node(term_node, bipartite=1)
        for q in e:
            B.add_edge(term_node, f"q{q}")

    # Layout: bipartite
    left = [n for n in B.nodes if n.startswith("q")]
    pos = nx.bipartite_layout(B, left)

    plt.figure(figsize=(10, 6))
    nx.draw(
        B, pos,
        with_labels=True,
        node_size=500,
        font_size=8
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()



plot_incidence_graph(HG_JW,
                     "JW Pauli Hypergraph (Incidence Graph)",
                     "JW_hypergraph_incidence.png")

plot_incidence_graph(HG_BK,
                     "BK Pauli Hypergraph (Incidence Graph)",
                     "BK_hypergraph_incidence.png")



import matplotlib.patches as patches

def plot_physical_hypergraph(hyperedges, site_positions, title, fname):
    """
    Draw qubits at physical lattice positions.
    Hyperedges drawn as convex hulls / connecting lines.
    """
    plt.figure(figsize=(6, 4))

    # Plot qubits
    for q, (x, y) in site_positions.items():
        plt.scatter(x, y, s=300)
        plt.text(x, y + 0.05, str(q), ha='center')

    # Plot hyperedges
    for e in hyperedges:
        if len(e) < 2:
            continue
        pts = [site_positions[q] for q in e]
        xs, ys = zip(*pts)
        plt.plot(xs + (xs[0],), ys + (ys[0],), alpha=0.3)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()




site_positions = {
    0: (0, 1), 1: (1, 1), 2: (2, 1),
    3: (0, 0), 4: (1, 0), 5: (2, 0),
}



plot_physical_hypergraph(
    HG_JW, site_positions,
    "JW Hypergraph on Physical Lattice",
    "JW_hypergraph_physical.png"
)

plot_physical_hypergraph(
    HG_BK, site_positions,
    "BK Hypergraph on Physical Lattice",
    "BK_hypergraph_physical.png"
)

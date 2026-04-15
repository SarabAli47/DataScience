import numpy as np
import matplotlib.pyplot as plt

def q_number(m, q):
    # Compute the q-number [m]_q = (1-q^m) / (1-q).
    # Return m in the limit as q approaches 1 to avoid instability via L'Hopital's rule.
    if abs(q - 1) < 1e-12:  # Handle the case when q is close to 1 to avoid numerical instability
        return m
    return (1 - q**m) / (1 - q)


def compute_phi(A):
    # Compute structural weights phi_i = in-degree(i) + 1 for each node i.
    # Used in the qpdeformed matrix construction (Section 4.3 of paper).
    return np.sum(A, axis=1) + 1



def compute_base_W(A):
    # Compute the classical column-stohastic web matrix W.
    # Each column j is normalized by the out-degree of node j.
    # Dangling nodes (out-degree 0) are handled by assigning equal probability to all nodes (1/N).
    # This serves as the q=1 reference point.
    N = A.shape[0]
    W0 = np.zeros((N, N))

    col_sums = np.sum(A, axis=0)

    for j in range(N):
        if col_sums[j] == 0:
            W0[:, j] = 1 / N
        else:
            W0[:, j] = A[:, j] / col_sums[j]

    return W0


def q_deformed_W(A, q):
    # Compute the q-deformed column-stochastic web matrix W(q) using the q-number weights.
    # Each outgoing link from node j to node i is weighted by q_number (ie [phi_i]_q / phi_i).
    # The columns are then normalized to sum to 1. Dangling nodes are handled as in the classical case.
    N = A.shape[0]
    W = np.zeros((N, N))

    phi = compute_phi(A)
    col_sums = np.sum(A, axis=0)

    for j in range(N):
        # dangling node
        if col_sums[j] == 0:
            W[:, j] = 1 / N
            continue

        mask = A[:, j] == 1

        weights = np.zeros(N)
        weights[mask] = q_number(phi[mask], q) / phi[mask]

        denom = np.sum(weights)
        if denom > 0:
            W[:, j] = weights / denom

    return W



def google_matrix(W, alpha=0.85):
    # Construct the Google matrix G from a web matrix W and damping factor alpha.
    # G = alpha * W + (1 - alpha) * E, where E is the teleportation matrix (all entries 1/N).
    # G is guranteed to be positive and column-stochastic, ensuring convergence of the power iteration by the Perron-Frobenius theorem.
    N = W.shape[0]
    E = np.ones((N, N)) / N
    return alpha * W + (1 - alpha) * E


def pagerank(G, tol=1e-8, max_iter=1000):
    # Compute the PageRank vector via power iteration.
    # Start with a uniform distribution and iteratively apply G until convergence (L1 norm of difference < tol) or max iterations reached.
    N = G.shape[0]
    r = np.ones(N) / N

    for _ in range(max_iter):
        r_new = G @ r
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return r



def q_pagerank(A, q, alpha=0.85):
    # Compute the q-deformed PageRank vector r(q) for a given graph and q value.
    # Full scale computation: construct W(q), then G(q), then compute PageRank via power iteration.
    Wq = q_deformed_W(A, q)
    Gq = google_matrix(Wq, alpha)
    return pagerank(Gq)

def verify_classical_recovery(A, tol=1e-8):
    # Verify that q_deformed_W(A, q=1) recovers the classical web matrix.
    # This serves as a sanity check that our q-deformation is consistent with the classical case when q=1.

    W_classical = compute_base_W(A)
    W_q1 = q_deformed_W(A, q=1.0)
    if np.allclose(W_classical, W_q1, atol=tol):
        print("Classical recovery at q=1 verified successfully.")
    else:
        print("Classical recovery at q=1 : x - check q_number formula")
        print(" Max difference:", np.max(np.abs(W_classical - W_q1)))

def dr_dq(A, q, alpha=0.85, eps=1e-5):
    # Compute the numerical derivative dr/dq of the PageRank vector with respect to q using finite differences.
    # This can be used to analyze the sensitivity of the PageRank values to changes in q, especially around critical points where the behavior may change significantly.
    r_plus = q_pagerank(A, q + eps, alpha)
    r_minus = q_pagerank(A, q - eps, alpha)
    return (r_plus - r_minus) / (2 * eps)

def find_crossings(qs, results):
    # Detect rank crossings - values of q where two nodes swap their relative ordering (ie phase tranistion points).
    # Returns a list of tuples (q_value, node_i, node_j) for each crossing.
    crossings = []
    n = results.shape[1]

    for i in range(n):
        for j in range(i+1, n):
            diff = results[:, i] - results [:, j]
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            for idx in sign_changes:
                crossings.append((qs[idx], i+1, j+1 ))  # Store q value and node indices (1-based)
    return crossings


# Adjacency matrix for a directed graph with n nodes; n=5.
# Convention: A[i, j] = 1 if there is a directed edge from node j to node i (ie column j points to row i).
# Self-loops are excluded per Bryan & Leise (2006) convention (a page cannot vote for itself).
A = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 0]
])


# Verify that classical recovery holds at q=1.
verify_classical_recovery(A)

W0 = compute_base_W(A)
W_half= q_deformed_W(A, q=0.5)  # Example q value for demonstration
print("\nClassical web matrix W (q=1):\n", np.round(W0, 4))
print("\nq-Deformed web matrix (q=0.5):\n", np.round(W_half, 4))

# Discrete q Sweep
# Compute the q-deformed PageRank for a range of q values and print results.
print("\nPageRank values at selected q values:")

qs_test = np.arange(0, 1, 0.05)  # q values from 0 to 1 in steps of 0.05
for q in qs_test:
    r = q_pagerank(A, q)
    print(f"q={q:.2f}: {np.round(r, 5 )}")

# Continuous q Sweep: Rank and Sensitivity Analysis
qs = np.linspace(0.0, 10.0, 300)
# 300 q values from 0 to 10 for smooth curve
results = np.array([q_pagerank(A, q) for q in qs])  # Compute PageRank for each q
sensitivities = np.array([dr_dq(A, q) for q in qs])  # Compute sensitivity dr/dq for each q

# Crossing detection
crossings = find_crossings(qs, results)
print("\nDetected rank crossings (phase-transitions-like behavior):")
for q_val, i, j in crossings:
    print(f"Nodes {i} and {j} swap ranking near q={q_val:.4f}")
else:
    print("No rank crossings detected for this graph.")



# Plotting the results
fix, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,8), sharex=True)

# Top: rank values r(q)
for i in range(A.shape[0]):
    ax1.plot(qs, results[:, i], label=f'Node {i+1}')
for q_val, i, j in crossings:
    ax1.axvline(x=q_val, color='gray', linestyle='--', alpha=0.5, label=f'Crossing{i}&{j}' if (i==crossings[0][1] and j==crossings[0][2]) else "")  # Label only the first crossing for legend
ax1.set_ylabel('PageRank r(q)')
ax1.set_title('q-Deformed PageRank vs q')
ax1.legend()
ax1.grid(True)

# Bottom: sensitivity dr/dq
for i in range(A.shape[0]):
    ax2.plot(qs, sensitivities[:, i], label=f'Node {i+1}')
ax2.set_xlabel('q')
ax2.set_ylabel('Sensitivity dr/dq')
ax2.set_title('Sensitivity of q-Deformed PageRank to q')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
import numpy as np

# calculating q-number 
def q_number(m, q):
    if q ==1:
        return m
    return (1 - q**m) / (1 - q)


# compute node's in-degree
def in_degree(A):
    return np.sum(A, axis=1)


# compute phi
def compute_phi(A):
    kin = in_degree(A)
    return kin + 1


# build q-deformed web matrix W(q)
def q_deformed_W(A, q):
    N = A.shape[0]
    W = np.zeros((N, N))

    phi = compute_phi(A)

    for j in range(N): # outgoing from j column
        column = A[:, j]

        if np.sum(column) == 0:
            # dangling node
            W[:, j] = 1/N
            continue

        weights = np.zeros(N)

        for i in range(N):
            if A[i, j] == 1:
                weights[i] = q_number(phi[i], q) / phi[i]


        denom = np.sum(weights)
        if denom > 0:
            W[:, j] = weights / denom
# print(W)

    return W

# build Google matrix G(q)
def google_matrix(W, alpha=0.85):
    N = W.shape[0]
    return alpha * W + (1 - alpha) * np.ones((N, N)) / N


# power iteration to compute PageRank
def pagerank(G, tol=1e-8, max_iter=1000):
    N = G.shape[0]
    r = np.ones(N) / N

    for _ in range(max_iter):
        r_new = G @ r
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return r

# full pipeline to compute q-deformed PageRank
def q_pagerank(A, q, alpha=0.85):
    Wq = q_deformed_W(A, q)
    Gq = google_matrix(Wq, alpha)
    r = pagerank(Gq)
    return r

# funxtion testing with example matrix
A = np.array([
    [1, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
])

qs = [0.5]

for q in qs:
    r = q_pagerank(A, q)
    print(f"q={q}: PageRank = {r}")

# Visually analyze rank changes with q
import matplotlib.pyplot as plt

qs = np.linspace(0.0,10.0, 5)
results = []
for q in qs:
    r = q_pagerank(A, q)
    results.append(r)

results = np.array(results)

for i in range(A.shape[0]):
    plt.plot(
        qs,
        results[:, i],
        label=f'Node {i+1}'
    )

plt.xlabel('q')
plt.ylabel('Rank')
plt.title('q-Deformed PageRank')
plt.legend(title='Nodes')
plt.grid(True)
plt.show()
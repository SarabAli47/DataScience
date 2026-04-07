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


# compute base (pre-deformation) column-stochastic matrix
# A.shape[0] gives the number of rows in the adjacency matrix A, which corresponds to the number of nodes in the graph.
def compute_base_W(A):
    N = A.shape[0]
# W0 is initialized as a zero matrix of size N x N, where N is the number of nodes.
    W0 = np.zeros((N, N))
# The loop iterates over each column j of the adjacency matrix A, calculating the sum of that column (col_sum).
    for j in range(N):
        column = A[:, j]
        col_sum = np.sum(column)
# If col_sum is zero, it means that node j is a dangling node, and we assign a uniform distribution (1/N) to all entries in column j of W0.
        if col_sum == 0:
            W0[:, j] = 1 / N
# Otherwise, we normalize the column by dividing each entry by col_sum to create a column-stochastic matrix.
        else:
            W0[:, j] = column / col_sum
    return W0



# build q-deformed web matrix W(q)
def q_deformed_W(A, q):
    N = A.shape[0]
    W = np.zeros((N, N))

    # compute base matrix (printing moved to caller so it happens once)
    W0 = compute_base_W(A)

    phi = compute_phi(A)

    for j in range(N):  # outgoing from j column
        column = A[:, j]

        if np.sum(column) == 0:
            # dangling node
            W[:, j] = 1 / N
            continue

# phi[i] is the in-degree of node i plus one, and q_number(phi[i], q) computes the q-deformed value of that in-degree.
        weights = np.zeros(N)
        for i in range(N):
            if A[i, j] == 1:
                weights[i] = q_number(phi[i], q) / phi[i]
# Renormalize the weights to ensure that the column sums to 1, making W a column-stochastic matrix. The sum of the weights is calculated, and if it's greater than zero, each weight is divided by this sum to normalize the column.
        denom = np.sum(weights)
        if denom > 0:
            W[:, j] = weights / denom

    return W

# build Google matrix G(q)
# alpha is the teleportation parameter (default 0.85)
# np.ones((N, N)) / N creates a matrix where each entry is 1/N, representing the uniform distribution for teleportation.
def google_matrix(W, alpha=0.85):
    N = W.shape[0]
    return alpha * W + (1 - alpha) * np.ones((N, N)) / N


# power iteration to compute PageRank
def pagerank(G, tol=1e-8, max_iter=1000):
    N = G.shape[0]
    r = np.ones(N) / N
# Computes iterations until convergence or max iterations reached, using L1 norm to check for convergence
    for _ in range(max_iter):
        r_new = G @ r
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return r

# main function to compute q-deformed PageRank
def q_pagerank(A, q, alpha=0.85):
    Wq = q_deformed_W(A, q)
    Gq = google_matrix(Wq, alpha)
    r = pagerank(Gq)
    return r

# funxtion testing with example matrix
# A here serves as a matrix of connections. This is not the stochastic column matrix
A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1]
])

# print the base column-stochastic matrix once
W0 = compute_base_W(A)
print('\nColumn-stochastic matrix (pre-deformation):\n', W0)

qs = [0.5, 1, 2, 5, 10, 20, 50, 100]

for q in qs:
    r = q_pagerank(A, q)
    print(f"q={q}: PageRank = {r}")

# Visually analyze rank changes with q
import matplotlib.pyplot as plt
# qs is a a set of equally divided points given by the third intiger between first and second float.
# qs = np.linspace(start, end, range)
qs = np.linspace(0.0,40.0, 200)
# results is an array of all the q points for the plot between float one and float 2
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
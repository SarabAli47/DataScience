import numpy as np
import matplotlib.pyplot as plt

# q-number function. Here we calculate q-number for a given m and q. This is used in the q-deformed matrix construction.
def q_number(m, q):
    if q == 1:
        return m
    return (1 - q**m) / (1 - q)


# compute phi function. This calculates the phi value for each node, which is the sum of its incoming links plus one. This is used in the q-deformed matrix construction to determine the weights for each node.
def compute_phi(A):
    return np.sum(A, axis=1) + 1


# compute base column-stochastic matrix W0. This function takes the adjacency matrix A and computes the base column-stochastic matrix W0, which is used as the starting point for the q-deformed matrix construction. It handles dangling nodes by assigning them equal probability to all nodes.
def compute_base_W(A):
    N = A.shape[0]
    W0 = np.zeros((N, N))

    col_sums = np.sum(A, axis=0)

    for j in range(N):
        if col_sums[j] == 0:
            W0[:, j] = 1 / N
        else:
            W0[:, j] = A[:, j] / col_sums[j]

    return W0


# compute q-deformed column-stochastic matrix Wq. This function constructs the q-deformed column-stochastic matrix Wq based on the adjacency matrix A and the deformation parameter q. It uses the phi values and q-numbers to determine the weights for each node, ensuring that the resulting matrix is column-stochastic. It also handles dangling nodes by assigning them equal probability to all nodes.
def q_deformed_W(A, q):
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


# compute Google matrix G from W. This function takes the q-deformed column-stochastic matrix W and computes the Google matrix G by applying the damping factor alpha. The Google matrix is a convex combination of W and a uniform matrix, which ensures that it is column-stochastic and suitable for PageRank computation.
def google_matrix(W, alpha=0.85):
    N = W.shape[0]
    E = np.ones((N, N)) / N
    return alpha * W + (1 - alpha) * E


# power iteration to compute PageRank. This function implements the power iteration method to compute the PageRank vector from the Google matrix G. It iteratively multiplies G by the rank vector until convergence, which is determined by the L1 norm of the difference between successive rank vectors being less than a specified tolerance. The function returns the final PageRank vector after convergence or after reaching the maximum number of iterations.
def pagerank(G, tol=1e-8, max_iter=1000):
    N = G.shape[0]
    r = np.ones(N) / N

    for _ in range(max_iter):
        r_new = G @ r
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return r


# main function to compute q-deformed PageRank. This function takes the adjacency matrix A, the deformation parameter q, and the damping factor alpha as inputs.
# It first computes the q-deformed column-stochastic matrix Wq using the q_deformed_W function, then constructs the Google matrix Gq using the google_matrix function, and finally computes the PageRank vector using the pagerank function. The resulting PageRank vector is returned as the output.
def q_pagerank(A, q, alpha=0.85):
    Wq = q_deformed_W(A, q)
    Gq = google_matrix(Wq, alpha)
    return pagerank(Gq)


# A sample adjacency matrix representing a directed graph with 5 nodes
A = np.array([
    [0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1]
])

# compute and display the base column-stochastic matrix W0 for the given adjacency matrix A. This serves as a reference point for understanding how the q-deformation modifies the transition probabilities in the PageRank computation.
W0 = compute_base_W(A)
print("\nBase column-stochastic matrix:\n", W0)


# q values to test.
qs_test = [0.0,0.5,1.0,2.0, 5.0, 10.0, 50.0, 100.0]

for q in qs_test:
    r = q_pagerank(A, q)
    print(f"q={q}: PageRank = {r}")


# compute q-deformed PageRank for a range of q values and store the results. This allows us to analyze how the PageRank values change as we vary the deformation parameter q, providing insights into the sensitivity of the PageRank algorithm to this parameter.
qs = np.linspace(0.0, 100.0, 200)
results = np.array([q_pagerank(A, q) for q in qs])


# plot the q-deformed PageRank results for each node as a function of q.
for i in range(A.shape[0]):
    plt.plot(qs, results[:, i], label=f'Node {i+1}')

plt.xlabel('q')
plt.ylabel('Rank')
plt.title('q-Deformed PageRank')
plt.legend()
plt.grid(True)
plt.show()
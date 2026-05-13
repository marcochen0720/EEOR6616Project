"""Utilities: data generation and OT helpers."""
import numpy as np
from scipy.spatial.distance import cdist


DEFAULT_MU1 = np.array([0.0, 0.0])
DEFAULT_MU2 = np.array([4.0, 3.0])
DEFAULT_SIGMA1 = np.array([[1.0, 0.6], [0.6, 1.2]])
DEFAULT_SIGMA2 = np.array([[1.5, -0.7], [-0.7, 1.0]])


def sample_point_clouds(n, mu1=DEFAULT_MU1, mu2=DEFAULT_MU2,
                        sigma1=DEFAULT_SIGMA1, sigma2=DEFAULT_SIGMA2,
                        seed=0):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mu1, sigma1, size=n)
    Y = rng.multivariate_normal(mu2, sigma2, size=n)
    return X, Y


def cost_matrix(X, Y):
    """Squared Euclidean cost C_ij = ||x_i - y_j||^2."""
    return cdist(X, Y, metric="sqeuclidean")


def uniform_marginals(m, n):
    return np.full(m, 1.0 / m), np.full(n, 1.0 / n)


def transport_cost(C, P):
    return float(np.sum(C * P))


def marginal_violation(P, a, b):
    return float(np.sum(np.abs(P.sum(axis=1) - a)) + np.sum(np.abs(P.sum(axis=0) - b)))


def assemble_marginal_constraints(m, n):
    """Build sparse Aeq (size (m+n, m*n)) for vec(P) ordered row-major."""
    from scipy.sparse import lil_matrix, csr_matrix

    A = lil_matrix((m + n, m * n))
    for i in range(m):
        A[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        A[m + j, j::n] = 1.0
    return csr_matrix(A)

"""Empirical Lipschitz-constant estimator for the Q -> K map."""

import numpy as np

EPS = 1e-12


def lipschitz_max_vectorized(Qs, Ks, eps=EPS):
    """Empirical Lipschitz constant L = max_{i<j} ||K_i - K_j||_F / ||Q_i - Q_j||_F.

    Qs: sequence of Q matrices (N, n, n). Ks: sequence of K matrices (N, m, n).
    """
    Qs = np.array(Qs)
    Ks = np.array(Ks)
    N = Qs.shape[0]
    Q_flat = Qs.reshape(N, -1)
    K_flat = Ks.reshape(N, -1)

    def pairwise_frobenius(X):
        norms = (X * X).sum(axis=1, keepdims=True)
        D2 = norms + norms.T - 2.0 * (X @ X.T)
        np.maximum(D2, 0.0, out=D2)
        return np.sqrt(D2, dtype=X.dtype)

    DQ = pairwise_frobenius(Q_flat)
    DK = pairwise_frobenius(K_flat)

    iu = np.triu_indices(N, k=1)
    ratios = DK[iu] / (DQ[iu] + eps)
    return float(ratios.max())

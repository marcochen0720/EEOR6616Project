"""Part 4: Gaussian OT and Bures--Wasserstein closed form."""
import time
import numpy as np
from scipy.linalg import sqrtm

from .utils import (
    sample_point_clouds, cost_matrix, uniform_marginals, transport_cost,
    DEFAULT_MU1, DEFAULT_MU2, DEFAULT_SIGMA1, DEFAULT_SIGMA2,
)
from .part1_lp import solve_lp_scipy


def bures_wasserstein_squared(mu1, sigma1, mu2, sigma2):
    """W_2^2 between N(mu1, sigma1) and N(mu2, sigma2)."""
    diff = mu1 - mu2
    s1_half = sqrtm(sigma1)
    inner = sqrtm(s1_half @ sigma2 @ s1_half)
    inner = np.real(inner)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * inner))


def benchmark_part4(seeds=range(10), ns=(50, 100, 200, 500),
                    mu1=DEFAULT_MU1, mu2=DEFAULT_MU2,
                    sigma1=DEFAULT_SIGMA1, sigma2=DEFAULT_SIGMA2,
                    save_dir=None):
    import os
    import json

    print("=== Part 4: Gaussian OT ===")
    w2_closed = bures_wasserstein_squared(mu1, sigma1, mu2, sigma2)
    print(f"  Bures-Wasserstein W_2^2 (closed form) = {w2_closed:.5f}")

    means, stds = [], []
    for n in ns:
        costs = []
        for seed in seeds:
            X, Y = sample_point_clouds(n, mu1, mu2, sigma1, sigma2, seed=seed)
            C = cost_matrix(X, Y)
            a, b = uniform_marginals(n, n)
            P, _, _ = solve_lp_scipy(C, a, b, method="highs-ds")
            costs.append(transport_cost(C, P))
        m, s = float(np.mean(costs)), float(np.std(costs))
        means.append(m); stds.append(s)
        print(f"  n={n:4d}:  mean={m:.4f}  std={s:.4f}  (over {len(costs)} seeds)")

    out = {
        "w2_closed": w2_closed,
        "ns": list(ns),
        "mean_costs": means,
        "std_costs": stds,
    }
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "part4_results.json"), "w") as f:
            json.dump(out, f, indent=2)

    # Return one (X, Y, P) for n = 200 visualization
    X, Y = sample_point_clouds(200, mu1, mu2, sigma1, sigma2, seed=0)
    C = cost_matrix(X, Y)
    a, b = uniform_marginals(200, 200)
    P, _, _ = solve_lp_scipy(C, a, b, method="highs-ds")
    return out, (X, Y, P), (mu1, sigma1, mu2, sigma2)


if __name__ == "__main__":
    benchmark_part4(save_dir="results")

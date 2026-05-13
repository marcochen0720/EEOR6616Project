"""Part 3: Entropically regularized OT (Sinkhorn algorithm).

We provide a custom log-domain Sinkhorn (stable for small lambda),
the POT library reference implementation, and a generic conic-solver
baseline through CVXPY (using cp.entr / the exponential cone).  Note
that the available conic backends play different algorithmic roles:
Clarabel is an interior-point conic solver, SCS is a first-order
operator-splitting conic solver -- neither is an IPM in the same sense
as HiGHS-IPM in Part 1.
"""
import time
import numpy as np
import cvxpy as cp
import ot

from .utils import (
    sample_point_clouds, cost_matrix, uniform_marginals,
    transport_cost, marginal_violation,
)


def sinkhorn_log_custom(C, a, b, lam, max_iter=2000, tol=1e-9,
                       record_every=1, return_history=False):
    """Numerically stable Sinkhorn in the log domain.

    Updates dual potentials f, g such that
        log P_ij = (f_i + g_j - C_ij) / lambda.
    Equivalent to multiplicative scaling but avoids underflow when lambda
    is small.
    """
    m, n = C.shape
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)
    f = np.zeros(m)
    g = np.zeros(n)

    history = []
    t0 = time.perf_counter()
    for it in range(max_iter):
        M = (g[None, :] - C) / lam
        f = lam * (log_a - _logsumexp(M, axis=1))
        M = (f[:, None] - C) / lam
        g = lam * (log_b - _logsumexp(M, axis=0))

        if (it + 1) % record_every == 0:
            P = np.exp((f[:, None] + g[None, :] - C) / lam)
            mv = marginal_violation(P, a, b)
            if return_history:
                history.append((it + 1, mv))
            if mv < tol:
                break
    t1 = time.perf_counter()

    P = np.exp((f[:, None] + g[None, :] - C) / lam)
    info = {"iters": it + 1}
    if return_history:
        info["history"] = history
    info["time"] = t1 - t0
    return P, t1 - t0, info


def _logsumexp(M, axis):
    M_max = np.max(M, axis=axis, keepdims=True)
    out = M_max + np.log(np.exp(M - M_max).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def solve_sinkhorn_pot(C, a, b, lam, numItermax=10000, stopThr=1e-9):
    t0 = time.perf_counter()
    P = ot.sinkhorn(a, b, C, reg=lam, numItermax=numItermax,
                    stopThr=stopThr, method="sinkhorn_log")
    t1 = time.perf_counter()
    return P, t1 - t0, {}


def solve_entropic_cvxpy(C, a, b, lam, solver="SCS"):
    """Solve the entropy-regularized OT via a generic conic solver.

    The default is SCS (first-order operator splitting on the exponential
    cone).  Pass solver='CLARABEL' for the interior-point conic backend
    instead.  Both reformulate cp.entr through exponential-cone constraints.
    """
    m, n = C.shape
    P = cp.Variable((m, n), nonneg=True)
    obj = cp.Minimize(cp.sum(cp.multiply(C, P)) - lam * cp.sum(cp.entr(P)))
    constraints = [cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b]
    prob = cp.Problem(obj, constraints)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver)
    except Exception as e:
        return None, time.perf_counter() - t0, {"status": "FAILED",
                                                  "error": str(e)}
    t1 = time.perf_counter()
    return P.value, t1 - t0, {"status": prob.status, "obj": prob.value}


def benchmark_part3(seed=0, n=200, save_dir=None):
    import os
    import json

    print(f"=== Part 3 (n = {n}) ===")
    X, Y = sample_point_clouds(n, seed=seed)
    C = cost_matrix(X, Y)
    a, b = uniform_marginals(n, n)

    print("-- (a) Custom log-domain Sinkhorn vs POT --")
    for lam in (1.0, 0.1, 0.01):
        P_pot, t_pot, _ = solve_sinkhorn_pot(C, a, b, lam=lam)
        P_my, t_my, info_my = sinkhorn_log_custom(C, a, b, lam=lam,
                                                   max_iter=20000, tol=1e-9)
        print(f"  lam={lam:6.3g}:  POT t={t_pot:.4f}s cost={transport_cost(C, P_pot):.5f} "
              f"viol={marginal_violation(P_pot, a, b):.2e}  | "
              f"custom t={t_my:.4f}s iters={info_my['iters']} "
              f"cost={transport_cost(C, P_my):.5f} "
              f"viol={marginal_violation(P_my, a, b):.2e}")

    print("-- (b) Compare with a generic conic solver (SCS) --")
    # Entropy cones at n=200 stress conic solvers; we use SCS and a smaller side
    n_ipm = 50
    X_s, Y_s = sample_point_clouds(n_ipm, seed=seed)
    C_s = cost_matrix(X_s, Y_s)
    a_s, b_s = uniform_marginals(n_ipm, n_ipm)
    lam_ipm = 0.1
    P_ipm, t_ipm, info_ipm = solve_entropic_cvxpy(C_s, a_s, b_s, lam_ipm,
                                                   solver="SCS")
    P_sk, t_sk, _ = solve_sinkhorn_pot(C_s, a_s, b_s, lam=lam_ipm,
                                        numItermax=20000, stopThr=1e-10)
    if P_ipm is not None:
        print(f"  n={n_ipm}, lam={lam_ipm}: SCS t={t_ipm:.3f}s "
              f"cost={transport_cost(C_s, P_ipm):.5f}  "
              f"Sinkhorn t={t_sk:.3f}s cost={transport_cost(C_s, P_sk):.5f}  "
              f"diff={np.linalg.norm(P_ipm - P_sk):.2e}")
    else:
        print(f"  Conic solver failed: {info_ipm}")

    print("-- (c) Convergence (custom log-Sinkhorn) --")
    history_by_lambda = {}
    for lam in (1.0, 0.1, 0.01):
        _, _, info = sinkhorn_log_custom(C, a, b, lam=lam, max_iter=2000,
                                         tol=1e-12, record_every=1,
                                         return_history=True)
        history_by_lambda[lam] = info["history"]

    print("-- (d) Regularization path (Sinkhorn / POT) --")
    lambdas = np.logspace(1, -2, 15)
    costs_path = []
    for lam_i in lambdas:
        P_i, _, _ = solve_sinkhorn_pot(C, a, b, lam=lam_i,
                                       numItermax=20000, stopThr=1e-10)
        costs_path.append(transport_cost(C, P_i))
        print(f"  lam={lam_i:8.4g}  cost={costs_path[-1]:.5f}")

    out = {
        "ipm": {"time": t_ipm,
                "cost": transport_cost(C_s, P_ipm) if P_ipm is not None else None,
                "lam": lam_ipm, "n": n_ipm},
        "sinkhorn_at_ipm_lam": {"time": t_sk,
                                  "cost": transport_cost(C_s, P_sk),
                                  "n": n_ipm},
        "lambdas_path": lambdas.tolist(),
        "costs_path": costs_path,
        "convergence": {f"{lam:g}": history_by_lambda[lam]
                         for lam in history_by_lambda},
    }
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "part3_results.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out, X, Y, C


if __name__ == "__main__":
    benchmark_part3(seed=0, n=200, save_dir="results")

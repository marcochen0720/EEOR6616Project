"""Part 2: Quadratic-regularized OT.

Two algorithms (both via CVXPY for fair high-level comparison):
  - ADMM via OSQP
  - IPM via Clarabel (interior-point conic solver, replaces ECOS).

Also: a from-scratch ADMM implementation following the splitting in the
project handout (used to study the effect of the penalty rho).
"""
import time
import numpy as np
import cvxpy as cp

from .utils import (
    sample_point_clouds, cost_matrix, uniform_marginals,
    transport_cost, marginal_violation,
)


def solve_quad_cvxpy(C, a, b, lam, solver="OSQP", **solver_kwargs):
    """Solve the quadratically regularized OT QP via CVXPY."""
    m, n = C.shape
    P = cp.Variable((m, n), nonneg=True)
    obj = cp.Minimize(cp.sum(cp.multiply(C, P)) + 0.5 * lam * cp.sum_squares(P))
    constraints = [cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b]
    prob = cp.Problem(obj, constraints)
    t0 = time.perf_counter()
    prob.solve(solver=solver, **solver_kwargs)
    t1 = time.perf_counter()
    return P.value, t1 - t0, {"status": prob.status, "obj": prob.value}


def project_transport_polytope(M, a, b, max_iter=200, tol=1e-10):
    """Project M onto U(a, b) under Euclidean distance via Dykstra-style
    alternating projection of the row/column marginal hyperplanes plus
    nonnegativity.  Used inside the ADMM Q-update.
    """
    Q = M.copy()
    p = np.zeros_like(Q)  # correction for nonneg
    q = np.zeros_like(Q)  # correction for marginal projection (combined)
    for _ in range(max_iter):
        Q_prev = Q
        # Step 1: project onto C_marg = {P : P 1 = a, P^T 1 = b}
        # Linear subspace; closed form using IPFP-like single step is not exact,
        # but the affine projection has a closed form.
        Y = Q + q
        Q_aff = _project_marginal_affine(Y, a, b)
        q = Y - Q_aff
        # Step 2: project onto nonneg
        Y = Q_aff + p
        Q = np.maximum(0.0, Y)
        p = Y - Q
        if np.linalg.norm(Q - Q_prev) < tol * (np.linalg.norm(Q_prev) + 1e-12):
            break
    return Q


def _project_marginal_affine(M, a, b):
    """Closed-form projection of M onto {P : P 1 = a, P^T 1 = b}.

    For a, b satisfying sum(a) = sum(b), solution is:
        P = M + (1/n)(a - r)1^T + 1(1/m)(b - c)^T - (1/(mn)) (sum(a)-sum(M))11^T
    where r = M 1, c = M^T 1.
    """
    m, n = M.shape
    r = M.sum(axis=1)
    c = M.sum(axis=0)
    total = M.sum()
    correction_r = (a - r) / n
    correction_c = (b - c) / m
    correction_total = (a.sum() - total) / (m * n)
    return M + correction_r[:, None] + correction_c[None, :] - correction_total


def solve_quad_admm_custom(C, a, b, lam, rho=1.0, max_iter=5000,
                           tol=1e-6, record_every=10):
    """Custom ADMM for min <C,P> + (lam/2)||P||_F^2 s.t. P in U(a,b), P>=0.

    Splitting: nonneg + cost in P-block, affine marginal constraints in Q-block.
        P_ij <- max(0, (rho (Q_ij - U_ij) - C_ij) / (lam + rho))    # nonneg + cost
        Q    <- Pi_{affine} (P + U)                                 # marginals
        U    <- U + P - Q
    The affine projection is closed-form (single SVD-free step).
    Equivalent to the splitting described in the handout, with nonneg moved
    into the P-block so the Q-update has a closed form (no Dykstra inner loop).
    """
    m, n = C.shape
    P = np.zeros_like(C)
    Q = np.zeros_like(C)
    U = np.zeros_like(C)

    history = []
    t0 = time.perf_counter()
    for it in range(max_iter):
        P = np.maximum(0.0, (rho * (Q - U) - C) / (lam + rho))
        Q = _project_marginal_affine(P + U, a, b)
        U = U + P - Q

        if (it + 1) % record_every == 0:
            primal_res = np.linalg.norm(P - Q)
            dual_res = rho * np.linalg.norm(Q - history[-1][4]) if history else np.inf
            mv = marginal_violation(P, a, b)
            history.append((it + 1, transport_cost(C, P), primal_res, mv, Q.copy()))
            if primal_res < tol and mv < tol:
                break

    t1 = time.perf_counter()
    # strip Q snapshots from history before returning
    history_clean = [(h[0], h[1], h[2], h[3]) for h in history]
    return P, t1 - t0, {"iters": it + 1, "history": history_clean}


def benchmark_part2(seed=0, n=200, save_dir=None):
    import os
    import json

    print(f"=== Part 2 (n = {n}) ===")
    X, Y = sample_point_clouds(n, seed=seed)
    C = cost_matrix(X, Y)
    a, b = uniform_marginals(n, n)

    lam = 0.1
    print(f"-- (a) ADMM (OSQP) and IPM (Clarabel) at lambda={lam} --")
    P_admm, t_admm, info_admm = solve_quad_cvxpy(C, a, b, lam, solver="OSQP")
    P_ipm, t_ipm, info_ipm = solve_quad_cvxpy(C, a, b, lam, solver="CLARABEL")
    print(f"  ADMM (OSQP)   : t={t_admm:.4f}s  cost={transport_cost(C, P_admm):.6f} "
          f"viol={marginal_violation(P_admm, a, b):.2e}")
    print(f"  IPM (Clarabel): t={t_ipm:.4f}s  cost={transport_cost(C, P_ipm):.6f} "
          f"viol={marginal_violation(P_ipm, a, b):.2e}")

    print(f"-- (b) Regularization path --")
    lambdas = np.logspace(1, -3, 12)
    costs = []
    for lam_i in lambdas:
        P_i, _, _ = solve_quad_cvxpy(C, a, b, lam_i, solver="CLARABEL")
        costs.append(transport_cost(C, P_i))
        print(f"  lam={lam_i:8.4g}  cost={costs[-1]:.6f}")

    print(f"-- (c) Effect of rho (custom ADMM, lambda=0.1) --")
    # rho needs to scale roughly with n to balance the dual updates
    rhos = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    rho_results = []
    for rho in rhos:
        P_r, t_r, info = solve_quad_admm_custom(C, a, b, lam=lam, rho=rho,
                                                max_iter=20000, tol=1e-6,
                                                record_every=50)
        rho_results.append({
            "rho": rho, "iters": info["iters"], "time": t_r,
            "cost": transport_cost(C, P_r),
            "viol": marginal_violation(P_r, a, b),
        })
        print(f"  rho={rho:8.2f}  iters={info['iters']:6d}  "
              f"t={t_r:.3f}s  cost={transport_cost(C, P_r):.6f}  "
              f"viol={marginal_violation(P_r, a, b):.2e}")

    out = {
        "lam_compare": lam,
        "admm": {"time": t_admm, "cost": transport_cost(C, P_admm),
                 "viol": marginal_violation(P_admm, a, b)},
        "ipm": {"time": t_ipm, "cost": transport_cost(C, P_ipm),
                "viol": marginal_violation(P_ipm, a, b)},
        "lambdas_path": lambdas.tolist(),
        "costs_path": costs,
        "rho_study": rho_results,
        "P_quad_lam_0p1": None,  # filled by caller for plotting
    }
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "part2_results.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out, P_ipm, X, Y, C


if __name__ == "__main__":
    benchmark_part2(seed=0, n=200, save_dir="results")

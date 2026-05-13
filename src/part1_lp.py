"""Part 1: LP for OT via Simplex (HiGHS-DS), IPM (HiGHS-IPM), and PDLP."""
import time
import numpy as np
from scipy.optimize import linprog

from .utils import (
    sample_point_clouds, cost_matrix, uniform_marginals,
    assemble_marginal_constraints, transport_cost, marginal_violation,
)


def solve_lp_scipy(C, a, b, method="highs-ds"):
    """Solve OT LP using scipy.optimize.linprog with given method.

    Returns (P, time, info_dict).
    """
    m, n = C.shape
    Aeq = assemble_marginal_constraints(m, n)
    beq = np.concatenate([a, b])
    c = C.ravel()
    t0 = time.perf_counter()
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0, None), method=method)
    t1 = time.perf_counter()
    if not res.success:
        raise RuntimeError(f"linprog failed ({method}): {res.message}")
    P = res.x.reshape(m, n)
    return P, t1 - t0, {"status": res.status, "message": res.message}


def solve_lp_pdlp(C, a, b, eps=1e-8, time_limit_sec=600):
    """Solve OT LP using Google's PDLP via ortools.pdlp.python."""
    from ortools.pdlp.python import pdlp
    from ortools.pdlp import solvers_pb2

    m, n = C.shape
    Aeq = assemble_marginal_constraints(m, n)
    beq = np.concatenate([a, b])
    c = C.ravel()

    qp = pdlp.QuadraticProgram()
    qp.resize_and_initialize(len(c), len(beq))
    qp.objective_vector = c
    qp.objective_offset = 0.0
    qp.variable_lower_bounds = np.zeros_like(c)
    qp.variable_upper_bounds = np.full_like(c, np.inf)
    qp.constraint_lower_bounds = beq
    qp.constraint_upper_bounds = beq
    qp.constraint_matrix = Aeq

    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.termination_criteria.simple_optimality_criteria.eps_optimal_relative = eps
    params.termination_criteria.simple_optimality_criteria.eps_optimal_absolute = eps
    params.termination_criteria.time_sec_limit = time_limit_sec
    params.verbosity_level = 0
    params.num_threads = 1

    t0 = time.perf_counter()
    result = pdlp.primal_dual_hybrid_gradient(qp, params)
    t1 = time.perf_counter()
    sol = result.primal_solution
    P = np.asarray(sol).reshape(m, n)
    return P, t1 - t0, {"status": str(result.solve_log.termination_reason)}


def solve_lp_pdhg_numpy(C, a, b, max_iter=50000, tol=1e-5,
                        primal_weight=1.0, restart_every=2000,
                        record_every=200):
    """Custom NumPy PDHG implementation of OT LP (extra credit).

    Saddle point form:
        min_{P>=0} max_{f,g}  <C,P> + f^T (a - P 1) + g^T (b - P^T 1).

    The constraint operator K(P) = (P 1, P^T 1) has ||K|| = sqrt(m+n),
    so we set tau = 0.99/(w*L), sigma = 0.99*w/L for primal weight w.
    We rescale cost C to unit infinity-norm to make w=1 reasonable, and
    perform restarts from the running average iterate.
    """
    m, n = C.shape
    L = np.sqrt(m + n)
    scale = max(C.max(), 1e-12)
    Cn = C / scale  # rescaled cost; recover true cost at the end via *scale

    tau = 0.99 / (primal_weight * L)
    sigma = 0.99 * primal_weight / L

    P = np.zeros_like(Cn)
    f = np.zeros(m)
    g = np.zeros(n)

    P_avg = np.zeros_like(Cn)
    f_avg = np.zeros(m)
    g_avg = np.zeros(n)
    weight_sum = 0.0

    history = []
    t0 = time.perf_counter()
    for it in range(max_iter):
        P_new = np.maximum(0.0, P - tau * (Cn - f[:, None] - g[None, :]))
        P_bar = 2.0 * P_new - P
        f = f + sigma * (a - P_bar.sum(axis=1))
        g = g + sigma * (b - P_bar.sum(axis=0))
        P = P_new

        weight_sum += 1.0
        P_avg += (P - P_avg) / weight_sum
        f_avg += (f - f_avg) / weight_sum
        g_avg += (g - g_avg) / weight_sum

        if (it + 1) % record_every == 0:
            mv = marginal_violation(P_avg, a, b)
            history.append((it + 1, scale * transport_cost(Cn, P_avg), mv))
            if mv < tol:
                break

        if restart_every and (it + 1) % restart_every == 0:
            P, f, g = P_avg.copy(), f_avg.copy(), g_avg.copy()
            P_avg = np.zeros_like(Cn)
            f_avg = np.zeros(m)
            g_avg = np.zeros(n)
            weight_sum = 0.0

    t1 = time.perf_counter()
    P_out = P_avg if weight_sum > 0 else P
    return P_out, t1 - t0, {
        "iters": it + 1, "history": history,
        "primal_weight": primal_weight, "scale": scale,
    }


def verify_correctness():
    """Sanity-check on small instances n in {3, 4}."""
    print("=== Part 1 sanity check (small instances) ===")
    for n in (3, 4):
        X, Y = sample_point_clouds(n, seed=1)
        C = cost_matrix(X, Y)
        a, b = uniform_marginals(n, n)
        P_s, t_s, _ = solve_lp_scipy(C, a, b, method="highs-ds")
        P_i, t_i, _ = solve_lp_scipy(C, a, b, method="highs-ipm")
        c_s = transport_cost(C, P_s)
        c_i = transport_cost(C, P_i)
        print(f"  n={n}: simplex={c_s:.6f}, ipm={c_i:.6f}, diff={abs(c_s-c_i):.2e}")
    print()


def benchmark_part1(ns=(50, 100, 200, 500), seed=0, save_dir=None,
                   include_custom_pdhg=True):
    """Run all three solvers across sizes; save timing/accuracy results."""
    import json

    rows = []
    P_ref_by_n = {}

    for n in ns:
        X, Y = sample_point_clouds(n, seed=seed)
        C = cost_matrix(X, Y)
        a, b = uniform_marginals(n, n)

        print(f"--- n = {n} ---")
        # Simplex
        P_simplex, t_simplex, _ = solve_lp_scipy(C, a, b, method="highs-ds")
        rows.append(dict(n=n, method="simplex",
                         time=t_simplex,
                         cost=transport_cost(C, P_simplex),
                         marg_viol=marginal_violation(P_simplex, a, b)))
        print(f"  simplex: t={t_simplex:.4f}s  cost={transport_cost(C, P_simplex):.6f}")
        P_ref_by_n[n] = P_simplex

        # IPM
        P_ipm, t_ipm, _ = solve_lp_scipy(C, a, b, method="highs-ipm")
        rows.append(dict(n=n, method="ipm",
                         time=t_ipm,
                         cost=transport_cost(C, P_ipm),
                         marg_viol=marginal_violation(P_ipm, a, b)))
        print(f"  ipm    : t={t_ipm:.4f}s  cost={transport_cost(C, P_ipm):.6f}")

        # PDLP
        P_pdlp, t_pdlp, info_p = solve_lp_pdlp(C, a, b, eps=1e-8)
        rows.append(dict(n=n, method="pdlp",
                         time=t_pdlp,
                         cost=transport_cost(C, P_pdlp),
                         marg_viol=marginal_violation(P_pdlp, a, b)))
        print(f"  pdlp   : t={t_pdlp:.4f}s  cost={transport_cost(C, P_pdlp):.6f}  "
              f"viol={marginal_violation(P_pdlp, a, b):.2e}")

        # Optional: custom PDHG (limited iters for big n)
        if include_custom_pdhg:
            mi = 20000 if n <= 200 else 30000
            P_pdhg, t_pdhg, info = solve_lp_pdhg_numpy(C, a, b, max_iter=mi, tol=1e-5)
            rows.append(dict(n=n, method="pdhg_numpy",
                             time=t_pdhg,
                             cost=transport_cost(C, P_pdhg),
                             marg_viol=marginal_violation(P_pdhg, a, b)))
            print(f"  pdhg_np: t={t_pdhg:.4f}s  iters={info['iters']}  "
                  f"cost={transport_cost(C, P_pdhg):.6f}  "
                  f"viol={marginal_violation(P_pdhg, a, b):.2e}")

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "part1_results.json"), "w") as f:
            json.dump(rows, f, indent=2)

    return rows, P_ref_by_n


if __name__ == "__main__":
    verify_correctness()
    rows, _ = benchmark_part1(ns=(50, 100, 200, 500), seed=0,
                              save_dir="results", include_custom_pdhg=True)

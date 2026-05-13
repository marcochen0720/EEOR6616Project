"""Fair-comparison benchmark for Part 3 Table 2.

For each n, use one fixed random instance (seed 0), warm up each backend once,
then time R repeats and report the median.  The fixed instance keeps the table
internally consistent: each solver's cost and wall time refer to the same point
cloud.  Wall time includes the Python/CVXPY/POT call overhead around the solver.
"""
import warnings, time, json, statistics, os
warnings.filterwarnings("ignore")

import cvxpy as cp

from src.utils import sample_point_clouds, cost_matrix, uniform_marginals, transport_cost
from src.part3_sinkhorn import sinkhorn_log_custom


N_LIST = [30, 50, 100, 200]
LAM = 0.1
SINKHORN_TOL = 1e-6
SINKHORN_MAXIT = 200_000
SEED = 0
REPEATS = 3


def timed_solve(fn, repeats=REPEATS):
    """Warm up once (discarded), then median of `repeats` timed calls."""
    fn()
    ts = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        ts.append(time.perf_counter() - t0)
    return result, statistics.median(ts), ts


def solve_sinkhorn(C, a, b):
    P, _, info = sinkhorn_log_custom(C, a, b, lam=LAM,
                                     max_iter=SINKHORN_MAXIT,
                                     tol=SINKHORN_TOL)
    return P, info


def make_cvxpy(C, a, b):
    n_, m_ = C.shape
    P = cp.Variable((n_, m_), nonneg=True)
    obj = cp.Minimize(cp.sum(cp.multiply(C, P)) - LAM * cp.sum(cp.entr(P)))
    cons = [cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b]
    return cp.Problem(obj, cons), P


def solve_cvxpy(C, a, b, solver):
    prob, P = make_cvxpy(C, a, b)
    prob.solve(solver=solver)
    return prob.status, prob.value, (P.value if P.value is not None else None)


def bench_one_n(n, seed=SEED):
    X, Y = sample_point_clouds(n, seed=seed)
    C = cost_matrix(X, Y); a, b = uniform_marginals(n, n)

    (P_sk, info_sk), t_sk, ts_sk = timed_solve(lambda: solve_sinkhorn(C, a, b))
    cost_sk = transport_cost(C, P_sk)

    try:
        (st_cl, _, P_cl), t_cl, ts_cl = timed_solve(
            lambda: solve_cvxpy(C, a, b, "CLARABEL")
        )
        if st_cl == "optimal" and P_cl is not None:
            cost_cl = transport_cost(C, P_cl)
        else:
            t_cl, ts_cl, cost_cl, st_cl = None, [], None, "FAILED"
    except Exception:
        t_cl, ts_cl, cost_cl, st_cl = None, [], None, "FAILED"

    try:
        (st_scs, _, P_scs), t_scs, ts_scs = timed_solve(
            lambda: solve_cvxpy(C, a, b, "SCS")
        )
        if st_scs == "optimal" and P_scs is not None:
            cost_scs = transport_cost(C, P_scs)
        else:
            t_scs, ts_scs, cost_scs, st_scs = None, [], None, "FAILED"
    except Exception:
        t_scs, ts_scs, cost_scs, st_scs = None, [], None, "FAILED"

    return {
        "n": n, "seed": seed,
        "sinkhorn": {"time_median": t_sk, "time_trials": ts_sk,
                     "iters": info_sk["iters"], "cost": cost_sk,
                     "status": "optimal"},
        "clarabel": {"time_median": t_cl, "time_trials": ts_cl,
                     "cost": cost_cl, "status": st_cl},
        "scs":      {"time_median": t_scs, "time_trials": ts_scs,
                     "cost": cost_scs, "status": st_scs},
    }


def main():
    rows = []
    for n in N_LIST:
        row = bench_one_n(n)
        rows.append(row)
        print(f"n={n:3d}  (seed={SEED}, median of {REPEATS} timed repeats)")
        for k in ("sinkhorn", "clarabel", "scs"):
            g = row[k]
            t = f"{g['time_median']:.4f}" if g["time_median"] is not None else "  --  "
            c = f"{g['cost']:.5f}" if g["cost"] is not None else "  --  "
            iters = f"  iters={g['iters']}" if k == "sinkhorn" else ""
            print(f"   {k:9s}: t_med={t}s{iters}  cost={c}  status={g['status']}")

    os.makedirs("results", exist_ok=True)
    with open("results/part3_table2_fair.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Wrote results/part3_table2_fair.json")


if __name__ == "__main__":
    main()

"""End-to-end driver for the OT programming project.

Runs all parts, saves figures and JSON summaries to ./results/.
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="ot")

from src.utils import sample_point_clouds, cost_matrix, uniform_marginals
from src.part1_lp import (
    verify_correctness, benchmark_part1, solve_lp_pdhg_numpy,
)
from src.part2_quad import benchmark_part2, solve_quad_cvxpy
from src.part3_sinkhorn import benchmark_part3, solve_sinkhorn_pot
from src.part4_gauss import benchmark_part4
from src.visualize import (
    plot_part1_timings, plot_couplings, plot_regularization_path,
    plot_sinkhorn_convergence, plot_admm_rho_effect,
    plot_gauss_convergence, plot_gauss_arrows, plot_pdhg_history,
)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def main():
    # ---- Part 1 ----
    print("\n############### PART 1: LP ###############")
    verify_correctness()
    rows1, P_ref = benchmark_part1(ns=(50, 100, 200, 500), seed=0,
                                    save_dir=RESULTS, include_custom_pdhg=True)
    plot_part1_timings(rows1, os.path.join(RESULTS, "part1_timing.png"))

    # PDHG history at n = 200 for diagnostic plot
    X, Y = sample_point_clouds(200, seed=0)
    C = cost_matrix(X, Y)
    a, b = uniform_marginals(200, 200)
    _, _, info = solve_lp_pdhg_numpy(C, a, b, max_iter=30000, tol=1e-7,
                                      record_every=200)
    lp_cost = next(r["cost"] for r in rows1 if r["n"] == 200 and r["method"] == "simplex")
    plot_pdhg_history(info["history"], os.path.join(RESULTS, "part1_pdhg_history.png"),
                      lp_cost=lp_cost)

    # ---- Part 2 ----
    print("\n############### PART 2: Quadratic ###############")
    out2, P_quad, X2, Y2, C2 = benchmark_part2(seed=0, n=200, save_dir=RESULTS)
    plot_regularization_path(out2["lambdas_path"], out2["costs_path"],
                              lp_cost=lp_cost,
                              save_path=os.path.join(RESULTS, "part2_reg_path.png"),
                              title=r"Quadratic regularization path")
    rho_results = out2["rho_study"]
    plot_admm_rho_effect(
        [r["rho"] for r in rho_results],
        [r["iters"] for r in rho_results],
        [r["time"] for r in rho_results],
        viols=[r["viol"] for r in rho_results],
        lp_cost=lp_cost,
        save_path=os.path.join(RESULTS, "part2_admm_rho.png"),
    )

    # ---- Part 3 ----
    print("\n############### PART 3: Entropic ###############")
    out3, X3, Y3, C3 = benchmark_part3(seed=0, n=200, save_dir=RESULTS)
    plot_regularization_path(out3["lambdas_path"], out3["costs_path"],
                              lp_cost=lp_cost,
                              save_path=os.path.join(RESULTS, "part3_reg_path.png"),
                              title=r"Sinkhorn regularization path")
    history_by_lambda = {float(k): v for k, v in out3["convergence"].items()}
    plot_sinkhorn_convergence(history_by_lambda,
                               os.path.join(RESULTS, "part3_convergence.png"))

    # ---- Part 4 ----
    print("\n############### PART 4: Gaussian OT ###############")
    out4, (X4, Y4, P4), (mu1, sigma1, mu2, sigma2) = benchmark_part4(
        seeds=range(10), ns=(50, 100, 200, 500), save_dir=RESULTS)
    plot_gauss_convergence(out4["ns"], out4["mean_costs"], out4["std_costs"],
                            out4["w2_closed"],
                            os.path.join(RESULTS, "part4_convergence.png"))
    plot_gauss_arrows(X4, Y4, P4, mu1, sigma1, mu2, sigma2,
                       os.path.join(RESULTS, "part4_arrows.png"))

    # ---- Visualization (Part 5) ----
    print("\n############### Visualization ###############")
    # Use the same n=200 point cloud for all four
    X, Y = sample_point_clouds(200, seed=0)
    C = cost_matrix(X, Y)
    a, b = uniform_marginals(200, 200)
    P_lp = P_ref[200]
    P_quad_vis, _, _ = solve_quad_cvxpy(C, a, b, lam=0.1, solver="CLARABEL")
    P_sk_low, _, _ = solve_sinkhorn_pot(C, a, b, lam=0.01, numItermax=20000,
                                          stopThr=1e-10)
    P_sk_hi, _, _ = solve_sinkhorn_pot(C, a, b, lam=1.0, numItermax=10000,
                                        stopThr=1e-10)
    plot_couplings(
        (X, Y),
        [P_lp, P_quad_vis, P_sk_low, P_sk_hi],
        ["LP (exact)",
         r"Quadratic, $\lambda=0.1$",
         r"Sinkhorn, $\lambda=0.01$",
         r"Sinkhorn, $\lambda=1$"],
        os.path.join(RESULTS, "couplings.png"),
        threshold_ratio=1e-3,
    )

    print("\nAll done. See ./results/")


if __name__ == "__main__":
    main()

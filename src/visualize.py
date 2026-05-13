"""Visualization utilities for the OT project."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_part1_timings(rows, save_path):
    methods = ["simplex", "ipm", "pdlp", "pdhg_numpy"]
    colors = {"simplex": "tab:blue", "ipm": "tab:orange",
              "pdlp": "tab:green", "pdhg_numpy": "tab:red"}
    labels = {"simplex": "Simplex (HiGHS-DS)", "ipm": "IPM (HiGHS-IPM)",
              "pdlp": "PDLP (OR-Tools)", "pdhg_numpy": "PDHG (custom NumPy)"}

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for m in methods:
        ns = sorted({r["n"] for r in rows if r["method"] == m})
        ts = [next(r["time"] for r in rows if r["method"] == m and r["n"] == n) for n in ns]
        if ns:
            ax.loglog(ns, ts, "o-", color=colors[m], label=labels[m])
    ax.set_xlabel("n (sources = targets)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Part 1: LP solve time vs. problem size")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_couplings(point_clouds, couplings, titles, save_path,
                   threshold_ratio=1e-3, max_lines=None):
    """Two-row panel: (top) sparsity heatmap of P (sorted), (bottom) connections.

    point_clouds: (X, Y) pair (shared across panels).
    couplings: list of P matrices.
    titles: list of strings, one per panel.
    """
    from matplotlib.colors import LogNorm

    X, Y = point_clouds
    n_panels = len(couplings)
    fig, axes = plt.subplots(2, n_panels,
                             figsize=(3.6 * n_panels, 6.6),
                             squeeze=False)

    # Sort source by x and target by x to expose diagonal structure
    src_order = np.argsort(X[:, 0])
    tgt_order = np.argsort(Y[:, 0])

    # Top row: heatmap of |P| on log scale
    for k, (P, title) in enumerate(zip(couplings, titles)):
        ax = axes[0, k]
        Pn = P[np.ix_(src_order, tgt_order)] / P.max()
        ax.imshow(np.log10(Pn + 1e-12), cmap="viridis", aspect="auto",
                  vmin=-5, vmax=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("target idx (sorted)", fontsize=8)
        if k == 0:
            ax.set_ylabel(r"source idx (sorted)" + "\n" +
                          r"$\log_{10}(P/P_{\max})$", fontsize=8)
        ax.tick_params(labelsize=7)

    # Bottom row: lines (weighted)
    for k, (P, title) in enumerate(zip(couplings, titles)):
        ax = axes[1, k]
        thr = threshold_ratio * P.max()
        idx = np.argwhere(P > thr)
        if max_lines and len(idx) > max_lines:
            order = np.argsort(-P[idx[:, 0], idx[:, 1]])[:max_lines]
            idx = idx[order]
        weights = P[idx[:, 0], idx[:, 1]]
        wmax = weights.max() if len(weights) > 0 else 1.0
        for (i, j), w in zip(idx, weights):
            ax.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]],
                    "-", color="black",
                    alpha=min(1.0, 0.05 + 0.95 * (w / wmax) ** 0.5),
                    linewidth=0.3 + 1.4 * (w / wmax) ** 0.5)
        ax.scatter(X[:, 0], X[:, 1], s=14, c="tab:blue", edgecolor="white",
                   linewidth=0.4, label="source", zorder=3)
        ax.scatter(Y[:, 0], Y[:, 1], s=14, c="tab:red", edgecolor="white",
                   linewidth=0.4, label="target", zorder=3)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlabel(f"#edges (>{threshold_ratio:.0e}$P_{{\\max}}$): {len(idx)}",
                      fontsize=8)
    axes[1, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_regularization_path(lambdas, costs, lp_cost, save_path,
                              title=r"Regularization path",
                              ylabel=r"$\langle C, P^*_\lambda\rangle - W_{LP}$"):
    """Plot cost gap to LP optimum on log--log axes."""
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    gap = np.array(costs) - lp_cost
    gap = np.maximum(gap, 1e-12)  # for log plot
    ax.loglog(lambdas, gap, "o-")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_sinkhorn_convergence(history_by_lambda, save_path):
    """history_by_lambda: dict {lambda: [(iter, marg_viol), ...]}."""
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for lam, hist in history_by_lambda.items():
        iters, viols = zip(*hist)
        ax.semilogy(iters, viols, "o-", markersize=3,
                    label=fr"$\lambda={lam:g}$")
    ax.set_xlabel("Sinkhorn iteration")
    ax.set_ylabel(r"$\|P\mathbf{1}-a\|_1+\|P^\top\mathbf{1}-b\|_1$")
    ax.set_title("Part 3: Marginal violation vs. iteration")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_admm_rho_effect(rhos, iters, times, viols=None, lp_cost=None,
                         costs=None, save_path=None):
    n_panels = 2 if viols is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.4))
    axes[0].loglog(rhos, iters, "o-")
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel("ADMM iterations\n(cap = $\\max$, lower = converged earlier)")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1].loglog(rhos, times, "o-")
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel("Wall time (s)")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
    if viols is not None:
        axes[2].loglog(rhos, np.maximum(viols, 1e-16), "o-")
        axes[2].set_xlabel(r"$\rho$")
        axes[2].set_ylabel("final marginal violation")
        axes[2].grid(True, which="both", linestyle=":", alpha=0.5)
    fig.suptitle(r"Effect of ADMM penalty $\rho$ (quadratic OT, $n=200$)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_gauss_convergence(ns, mean_costs, std_costs, w2_closed, save_path):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    mean_costs = np.asarray(mean_costs)
    std_costs = np.asarray(std_costs)
    ax.errorbar(ns, mean_costs, yerr=std_costs, fmt="o-", capsize=3,
                label=r"discrete $\langle C,P^*\rangle$")
    ax.axhline(w2_closed, color="k", linestyle="--",
               label=r"$W_2^2$ (Bures--Wasserstein)")
    ax.set_xlabel("n (samples per Gaussian)")
    ax.set_ylabel(r"transport cost")
    ax.set_xscale("log")
    ax.set_title(r"Part 4: discrete OT $\to$ Bures--Wasserstein $W_2^2$")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_gauss_arrows(X, Y, P, mu1, sigma1, mu2, sigma2, save_path,
                       threshold_ratio=1e-3):
    from matplotlib.patches import Ellipse

    def cov_ellipse(ax, mean, cov, color, n_std=2.0):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0 * n_std * np.sqrt(vals)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor="none", lw=1.5, ls="--")
        ax.add_patch(ell)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    cov_ellipse(ax, mu1, sigma1, "tab:blue")
    cov_ellipse(ax, mu2, sigma2, "tab:red")
    thr = threshold_ratio * P.max()
    for i, j in np.argwhere(P > thr):
        w = P[i, j] / P.max()
        ax.annotate("", xy=Y[j], xytext=X[i],
                    arrowprops=dict(arrowstyle="->", color="black",
                                    alpha=0.1 + 0.9 * w, lw=0.3 + 1.0 * w))
    ax.scatter(X[:, 0], X[:, 1], s=18, c="tab:blue", edgecolor="white",
               linewidth=0.5, label="source", zorder=3)
    ax.scatter(Y[:, 0], Y[:, 1], s=18, c="tab:red", edgecolor="white",
               linewidth=0.5, label="target", zorder=3)
    ax.set_aspect("equal")
    ax.set_title("Part 4: Gaussian OT arrows (n = 200)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_pdhg_history(history, save_path, lp_cost):
    iters = [h[0] for h in history]
    costs = [h[1] for h in history]
    viols = [h[2] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].semilogx(iters, costs, "o-", markersize=3)
    axes[0].axhline(lp_cost, color="k", linestyle="--", label=r"$W_{LP}$")
    axes[0].set_xlabel("PDHG iteration")
    axes[0].set_ylabel(r"$\langle C, \bar P\rangle$")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].legend()
    axes[1].loglog(iters, viols, "o-", markersize=3)
    axes[1].set_xlabel("PDHG iteration")
    axes[1].set_ylabel("marginal violation (L1)")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
    fig.suptitle("Custom PDHG (NumPy) on OT LP, n = 200")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

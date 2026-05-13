# IEOR 6616 — Computational Optimal Transport

Jiayi Chen UNI:JC6601

Programming project: solving discrete optimal transport with three solver
paradigms for the LP, ADMM/IPM for the quadratically regularized OT, and
Sinkhorn for the entropy-regularized OT, plus a comparison with the
Bures–Wasserstein closed form on Gaussians.


## Layout

```
project/
├── src/
│   ├── utils.py           data generation, cost matrix, marginal helpers
│   ├── part1_lp.py        Simplex (HiGHS-DS), IPM (HiGHS-IPM), PDLP, custom PDHG
│   ├── part2_quad.py      ADMM (OSQP), IPM (Clarabel), custom-NumPy ADMM
│   ├── part3_sinkhorn.py  POT Sinkhorn, log-domain custom Sinkhorn, conic solver
│   ├── part4_gauss.py     Bures–Wasserstein closed form + sample-based OT
│   └── visualize.py       all plotting routines
├── run_all.py             end-to-end driver: produces ./results/*
├── results/               figures (.png) and JSON summaries
├── report/
│   ├── report.tex         3-page NeurIPS-style writeup
│   ├── report.pdf         compiled report (3 pages text + 3 pages figures)
│   ├── refs.bib
│   └── neurips_2024.sty
└── requirements.txt
```

## Setup

```bash
python -m pip install -r requirements.txt
```

This pulls in NumPy, SciPy, matplotlib, CVXPY (with OSQP/Clarabel/SCS),
POT, and OR-Tools (PDLP).

## Reproduction

```bash
# Regenerate every figure and JSON in ./results/
python run_all.py

# Recompile the LaTeX report
cd report && pdflatex report && bibtex report && pdflatex report && pdflatex report
```

`run_all.py` runs all four parts end-to-end (≈ 3 minutes on a laptop CPU
at the default sizes).  Each `partN_*.py` module also has a `__main__`
block so the parts can be run in isolation.

### Solver / algorithm map

| File                | Algorithm                              | Library                              |
|---------------------|----------------------------------------|--------------------------------------|
| `part1_lp.solve_lp_scipy`     | Simplex                       | `scipy.optimize.linprog` (HiGHS-DS)  |
| `part1_lp.solve_lp_scipy`     | Interior point                | `scipy.optimize.linprog` (HiGHS-IPM) |
| `part1_lp.solve_lp_pdlp`      | PDHG with restarts (PDLP)     | `ortools.pdlp.python`                |
| `part1_lp.solve_lp_pdhg_numpy`| Vanilla PDHG (extra credit)   | NumPy from scratch                   |
| `part2_quad.solve_quad_cvxpy` | ADMM                          | CVXPY → OSQP                         |
| `part2_quad.solve_quad_cvxpy` | Interior point                | CVXPY → Clarabel                     |
| `part2_quad.solve_quad_admm_custom` | ADMM (extra credit)     | NumPy from scratch                   |
| `part3_sinkhorn.solve_sinkhorn_pot` | Sinkhorn (log-domain)   | `ot.sinkhorn`                        |
| `part3_sinkhorn.sinkhorn_log_custom`| Sinkhorn (extra credit) | NumPy from scratch (log-stable)      |
| `part3_sinkhorn.solve_entropic_cvxpy`| Conic (first-order, exp-cone) | CVXPY → SCS (or Clarabel for IPM conic) |
| `part4_gauss.bures_wasserstein_squared` | Closed form         | `scipy.linalg.sqrtm`                 |

## LLM usage

A coding LLM was used as an interactive pair-programmer for boilerplate
(plot styling, sparse-matrix assembly, NeurIPS LaTeX scaffolding) and
to surface a bug in our first ADMM splitting (an underconverged inner
Dykstra projection).  All algorithmic decisions, parameter sweeps, the
analytic discussion in the report, and the final code reviews were done
by the author.

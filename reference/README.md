# Reference implementation — PSOMbest reproduction

Python port of `PSOMbest_Main_Master.m` (Twumasi et al., 2024), a DFFG-style benchmark driver that compares five optimisers on 22 benchmark cases.

## Files

| File | Purpose |
|---|---|
| `psombest_main_master_python_corrected.py` | Main driver: scenario configuration, experiment loop, statistics, Excel/NPZ export, convergence plots |
| `benchmarks/optimizers_fast.py` | Vectorised standard PSO, PSOMbest, and B1-dynamic-gbest |
| `benchmarks/optimizers_variants.py` | Experimental B1 grouping variants (`b1_coord`, `b1_proj`, `b1_hybrid`) plus a unified runner with extra metrics |

## Algorithms

- **PSO** — classical global-best PSO, uniform initialisation, mirror-boundary velocity flip.
- **PSO-AWDV** — adaptive inertia and time-varying acceleration coefficients with a delay term (Xu et al., 2021).
- **PSOEA** — rank-selected crossover to initialise the swarm, then classical PSO.
- **GA** — real-coded genetic algorithm: rank selection, arithmetic crossover, cooled Gaussian mutation, elitism.
- **PSOMbest** — k-means clustering of the swarm each iteration; a third velocity term pulls each particle toward its cluster leader.

## Benchmark cases

| Label | Function | Dimension | Optimum |
|---|---|---|---|
| F1 | Rotated hyper-ellipsoid | 30 | 0 |
| F2 | Zakharov | 30 | 0 |
| F3 | Dixon–Price | 30 | 0 |
| F4 | Sum of different powers | 30 | 0 |
| F5 | Bohachevsky 1 | 2 | 0 |
| F6 | Matyas | 2 | 0 |
| F7 | Drop-Wave | 2 | −1 |
| F8 | Rastrigin | 2 | 0 |
| F9 | Ackley | 100 | 0 |
| F10 | Lévy | 100 | 0 |
| F11 | Griewank | 100 | 0 |
| F12 | Styblinski–Tang | 100 | −3916.16599 |
| F13 | Noisy Sphere | 30 | 0 |
| F14 | Noisy Rastrigin | 30 | 0 |
| F15 | Noisy Ackley | 30 | 0 |
| F16 | Noisy Griewank | 30 | 0 |
| F17 (2D/30D/100D) | Rosenbrock | 2 / 30 / 100 | 0 |
| F18 (2D/30D/100D) | Powell sum | 2 / 30 / 100 | 0 |

F18 uses the same formula as F4 in the source code and is therefore excluded from the comparison tables (but retained in the driver for completeness). Noise standard deviation is `σ = 0.01`.

## Usage

```bash
python psombest_main_master_python_corrected.py --mode debug
python psombest_main_master_python_corrected.py --mode sample
python psombest_main_master_python_corrected.py --mode full --workers 8
```

| Mode | Cases | Particles | Iterations | Runs |
|---|---|---|---|---|
| `debug` | F8 | 30 | 100 | 3 |
| `sample` | 9 cases | 60 | 300 | 10 |
| `full` | all 22 | 150 | 1000 | 50 |

## Outputs

Written to `PSOMbest_Python_Reproduction_Results/`:

- `PSOMbest_Python_Reproduction_Statistics.xlsx` — per-case sheets plus `Journal_Summary`, `Paper_Match`, `Published_Optimum`, `Reproduced_Optimum`, `Computation_Time`
- `PSOMbest_Python_Reproduction_RawResults.npz` — raw scores, timings, convergence curves, best positions
- `Figures/*.png` — per-case mean convergence curves
- `Composite_Plots/*.png` — six-panel composite figures
- `PSOMbest_Python_Reproduction_Log.txt` — tee'd console log

## Corrections relative to the MATLAB source

This file is the **corrected** port (2026-11-04). Four parity bugs were fixed:

1. `local_lhsdesign` — MATLAB's `randperm` is 1-based; added `+1` so the design lies in (0, 1) instead of sometimes falling below the lower bound.
2. `run_pso_awd` — inertia/`c1`/`c2` were computed but the old values were used. Values are now computed at the end of an iteration and used from the next one, matching MATLAB.
3. `local_kmeans` — the previous-label vector now starts as all cluster 1 (index 0), matching MATLAB's `ones`.
4. `run_real_coded_ga` — elite count now uses MATLAB half-away-from-zero rounding instead of Python's half-to-even.

Random streams still differ from MATLAB (MT19937 in both, different seeding/consumption), so compare statistics rather than individual runs.

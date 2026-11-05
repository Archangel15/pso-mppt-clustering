# 07 — Leader isolation

**Question:** Is PSOMbest's behaviour caused by its **one-number group leader**?

## Design

PSOMbest settings are fixed (`k = 5`, `c1, c2, c3 = 0.5, 1.5, 1.0`, `Vmax = 1×range`, velocity discarded, LHS start, k-means every iteration, particle-by-particle update, `w` decays by 0.99). Only the leader rule changes.

6 rules × 10 settings × {original, shifted} × 30 runs = **3,600 runs**.

| Rule | Description |
|---|---|
| `scalar_first` | The supplied code: first cluster member's **first coordinate**, copied to every coordinate |
| `scalar_mean` | Cluster mean over all members and all coordinates (one number), copied to every coordinate |
| `scalar_best1` | Best member's (lowest current cost) **first coordinate**, copied to every coordinate |
| `stored_best` | A real vector: the cluster member with the lowest current cost |
| `pbest_vec` | A real vector: the best personal best among cluster members |
| `none` | No group pull (`c3 = 0`): PSOMbest settings with plain PSO behaviour |

Cases: `F2`, `F3`, `F9`, `F10`, `F11`, `F12`, `F14`, `F16`, `F17_30D`, `F17_100D`, each in original and shifted form.

## Files

| File | Purpose |
|---|---|
| `leader_test.py` | Six leader rules and the runner |
| `shifted.py` | Shifted-optimum objective |
| `ports.py` | Objectives and helpers (shared snapshot) |
| `run_all.py` | Worker runner |
| `merge.py` | Merges shards into `merged_leader.csv` |
| `selftest.py` | Quick reproducibility check |

## Run

```bash
python selftest.py               # ~10 s
python run_all.py 0 8 30         # repeat for worker IDs 0..7
python merge.py
```

## Outputs

- `results_w*.csv` (columns: `case`, `version`, `config`, `run`, `best`, `n_evals`, `time_s`)
- `merged_leader.csv`

## Result summary

Median absolute error over the 10 settings:

| Leader rule | Original | Shifted |
|---|---|---|
| `scalar_mean` | 4.6e-10 | 385.05 |
| `scalar_best1` | 7.2e-05 | 417.43 |
| `scalar_first` (supplied) | 0.4506 | 538.96 |
| `stored_best` (vector) | 15.64 | 16.78 |
| `pbest_vec` (vector) | 16.10 | 17.48 |
| `none` | 79.66 | 104.61 |

The scalar leader is near-perfect on the original benchmarks and collapses on shifted ones. The vector rules, especially `stored_best`, are stable across both. This isolates the equal-coordinate artefact: the scalar leader's advantage is tied to benchmark structure, not to a general optimisation benefit.

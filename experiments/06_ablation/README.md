# 06 — Ablation: which PSOMbest features help B1-proj-fade?

**Question:** Of the features borrowed from the supplied PSOMbest code, which ones actually help the strongest B1 variant?

## Design

6 configurations × 10 settings × {original, shifted} × 30 runs = **3,600 runs**.

The base algorithm is B1-proj-fade: groups built once along the first principal direction, 5 equal-count cuts, leader = best personal best in the group, `c3` fades 1 → 0, particle-by-particle update, immediate global-best update, mirror boundary, `w` starts at 1 and decays by 0.99.

| Config | Change relative to base |
|---|---|
| `base` | `c1, c2 = 1.5, 2.0`; `Vmax = 0.1×range`; velocity kept; uniform start |
| `c12` | `c1, c2 = 0.5, 1.5` (the PSOMbest source settings) |
| `vmax` | `Vmax = 1.0×range` (the literal source setting) |
| `novel` | velocity discarded after every move (no inertia) |
| `lhs` | Latin-hypercube start (same routine as the PSOMbest port) |
| `all4` | all four switches on |

Cases: `F2`, `F3`, `F9`, `F10`, `F11`, `F12`, `F14`, `F16`, `F17_30D`, `F17_100D`.

## Files

| File | Purpose |
|---|---|
| `ablation.py` | Six configuration definitions and the B1-proj-fade runner |
| `shifted.py` | Shifted-optimum objective used by the `shift` version |
| `ports.py` | Objectives, helpers and base implementations |
| `run_all.py` | Worker runner |
| `merge.py` | Merges shards into `merged_ablation.csv` |
| `selftest.py` | Quick reproducibility check (base config, F2, seed 1) |

## Run

```bash
python selftest.py               # ~20 s
python run_all.py 0 8 30         # repeat for worker IDs 0..7
python merge.py
```

Expected total: about 1 h on 8 workers (roughly 8 s per run).

## Outputs

- `results_w*.csv` (columns: `case`, `version`, `config`, `run`, `best`, `n_evals`, `time_s`)
- `merged_ablation.csv`

## Result summary

Median absolute error over the 10 settings; per-case average rank in parentheses (1 = best):

| Config | Original | Shifted |
|---|---|---|
| `base` | 2.19 (2.00) | 10.38 (2.10) |
| `lhs` | 2.78 (2.40) | 12.54 (2.00) |
| `novel` | 1.90 (2.50) | 2.43 (3.20) |
| `vmax` | 17.79 (3.40) | 22.06 (3.20) |
| `c12` | 6.83 (4.80) | 12.95 (4.60) |
| `all4` | 45.91 (5.90) | 82.82 (5.90) |

The defaults are already near-optimal. Both "source-faithful" switches (`c12`, `vmax`) hurt. LHS is roughly neutral (slightly better on shifted cases). Stacking all four is catastrophic.

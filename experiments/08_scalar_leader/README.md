# 08 — Scalar leader applied to B1

**Question:** If B1's three grouping rules get the PSOMbest-style **one-number** group leader, do they beat PSOMbest?

## Design

Everything is identical to the Master-loop B1 runs in experiment 03: groups built once and frozen, `c1, c2 = 1.5, 2.0`, `Vmax = 0.1×range`, velocity kept, uniform start, mirror boundary, `w` starts at 1 and decays by 0.99, `c3` fades 1 → 0, leader table rebuilt from personal bests every iteration. Only the target of the group pull changes.

3 grouping rules × 3 scalar rules = 9 configurations.

| Scalar rule | Description |
|---|---|
| `none` | Normal B1 vector leader (must reproduce experiment 03 exactly; the self-test checks this) |
| `lead0` | First coordinate of the normal leader vector, copied to all coordinates |
| `leadmean` | Mean of all coordinates of the normal leader vector, copied to all coordinates |
| `first` | The supplied PSOMbest rule: first group member (lowest particle number), its first coordinate, using current positions; empty group → first coordinate of the global best |

> The self-test uses `scalar=None` as the "off" setting, which is the same code path as `none`; `run_all.py` uses the three scalar rules (`lead0`, `leadmean`, `first`).

Scale:

- 19 original cases (F18 excluded; duplicate of F4) × 9 configs × 50 runs = 8,550
- 10 shifted cases × 9 configs × 30 runs = 2,700
- **Total: 11,250 runs**, roughly 3 h on 8 workers

## Files

| File | Purpose |
|---|---|
| `scalar_test.py` | The three grouping rules with a scalar-leader switch |
| `shifted.py` | Shifted-optimum objective |
| `ports.py` | Objectives, helpers and baseline implementations |
| `run_all.py` | Worker runner (original + shifted in one job list) |
| `merge.py` | Merges shards into `merged_scalar.csv` |
| `analyse.py` | Writes `results.md`: mean/median error, win/loss, rank-sum comparisons against PSOMbest and vector-leader baselines |
| `selftest.py` | Verifies that `scalar=None` reproduces the earlier code (must print `True` three times) |
| `baseline_orig.csv`, `baseline_shift.csv` | Baseline runs used by `analyse.py` |
| `paper_values.json` | Published MAE values for the paper comparison |

## Run

```bash
python selftest.py                              # ~20 s; must print True three times
python run_all.py 0 8 50 30                     # repeat for worker IDs 0..7
python merge.py                                 # writes merged_scalar.csv
python analyse.py                               # writes results.md
```

Use `python run_all.py 0 8 50 0` to skip the shifted part. Jobs are ordered run-by-run, so the sweep can be stopped and analysed at any time; restarting resumes.

## Outputs

- `results_w*.csv` (columns: `case`, `version`, `kind`, `scalar`, `run`, `best`, `n_evals`, `time_s`)
- `merged_scalar.csv`
- `results.md`

## Result summary

Median absolute error (original | shifted):

| Config | Original | Shifted |
|---|---|---|
| b1 + leadmean | 0.0000 | 27.18 |
| b1 + first | 0.0001 | 36.27 |
| b1 + lead0 | 0.0014 | 34.07 |
| b1_coord + leadmean | 0.0000 | 27.34 |
| b1_coord + first | 0.0002 | 35.61 |
| b1_coord + lead0 | 0.0012 | 35.88 |
| b1_proj + leadmean | 0.0000 | 26.76 |
| b1_proj + first | 0.0001 | 37.15 |
| b1_proj + lead0 | 0.0023 | 37.45 |

The scalar leader gives near-zero error on the original functions and degrades by ~27–37 on the shifted ones. The equal-coordinate structure of the original benchmarks is doing the work. `leadmean` is the least harmful scalar rule. See `results.md` for the full statistical comparison.

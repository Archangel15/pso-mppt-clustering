# 02 — Shifted-optimum robustness

**Question:** Do the algorithms still work when each benchmark's optimum is moved so its best value differs in every coordinate?

## Design

10 cases × 6 algorithms × 30 runs.

Each benchmark is shifted: `f_shift(x) = f(x − s)`. The offset `s_j` is drawn once per case from ±25% of the search range (clipped so the optimum stays inside the bounds). The optimum value is unchanged, but the optimum *point* no longer has equal coordinates across dimensions.

Algorithms: `pso`, `master_repro`, `psombest_master_vm`, `b1_fade`, `b1_coord_fade`, `b1_proj_fade`.

## Why this exists

The original benchmarks have optima at points like `(0, 0, …, 0)`, `(1, 1, …, 1)` or `(-2.903534, …)`. A leader rule that copies one number to every coordinate can exploit that structure. Shifting the optimum removes the shortcut without changing the difficulty of the landscape, isolating whether a method's advantage is general or an artefact of coordinate equality.

## Files

| File | Purpose |
|---|---|
| `shifted.py` | Builds per-case offsets and installs the shifted objective |
| `ports.py` | Algorithm profiles and helpers (shared snapshot) |
| `run_all.py` | Worker runner (10 cases × 6 algorithms × runs) |
| `merge.py` | Merges `results_*.csv` into `merged_shift.csv` |

## Run

```bash
python run_all.py 0 8 30     # repeat for worker IDs 0..7
python merge.py
```

## Outputs

- `results_w*.csv`
- `merged_shift.csv`

## Use it with

Experiments 07 and 08 use the same shifted cases to test leader rules. Use the original vs shifted contrast there; this experiment establishes the baseline robustness of the full algorithms.

# 03 — B1 inside the Master PSO loop

**Question:** How do B1-fade, B1-coord-fade and B1-proj-fade perform inside the same particle-by-particle loop as the paper's algorithms, rather than the vectorised `b1_pkg` runner?

## Design

22 cases × 3 algorithms × 50 runs, `N = 150`, `MaxIt = 1000`, `σ = 0.01`, seeds 1–50.

The loop is the Master-file loop: particle-by-particle updates, immediate global-best updates, mirror boundaries, `w` starts at 1 and decays by 0.99, `c3` fades 1 → 0. Groups are built once and frozen, exactly as in the B1 design. This makes the results directly comparable to the Master-code PSOMbest runs in experiment 01.

## Files

| File | Purpose |
|---|---|
| `ports.py` | Master-loop implementations, including `run_b1_masterloop` |
| `run_all.py` | Worker runner; writes algorithm names with a `_ml` suffix |
| `merge.py` | Merges shards into `merged_ml.csv` |

## Run

```bash
python run_all.py 0 8 50     # repeat for worker IDs 0..7
python merge.py
```

## Outputs

- `results_w*.csv`
- `merged_ml.csv` (columns: `case`, `algo`, `run`, `best`, `n_evals`, `time_s`)

## Notes

- The `_ml` suffix (`b1_fade_ml`, `b1_coord_fade_ml`, `b1_proj_fade_ml`) distinguishes these runs from the vectorised runs in experiment 01.
- Median absolute errors are near zero for all three variants on most cases; the interesting comparison is experiment 04 (regrouping) and experiment 01 (head-to-head).

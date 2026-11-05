# 00 — Version comparison

**Question:** How do the supplied PSOMbest implementations differ, and does the earlier in-house version reproduce them?

## Design

22 benchmark cases × 7 versions × `RUNS` runs.

| Version | Description |
|---|---|
| `fixed` | Port of `psokmeans_fixed.m`; re-evaluates every cluster member's current position before choosing a leader |
| `master_repro` | Master code as supplied: scalar leader, no velocity memory |
| `master_corrected` | Master code with a stored-best vector leader and velocity memory |
| `master_table3` | `master_corrected` with `c1 = 1.5`, `c2 = 2.0` |
| `pso` | Master classical PSO port |
| `mineA` | Earlier in-house vectorised version, `c1 = 1.5`, `c2 = 2.0`, `Vmax = 0.1×range` |
| `mineB` | Earlier in-house version with the source settings, `c1 = 0.5`, `c2 = 1.5`, `Vmax = 1×range` |

## Files

| File | Purpose |
|---|---|
| `ports.py` | Objective functions, LHS/k-means helpers, version profiles, PSO/PSO-AWDV/PSOEA/GA/B1 ports |
| `run_all.py` | Restart-safe worker runner |
| `analyse.py` | Merges `results_*.csv`, prints mean/SD/best/worst and average rank |
| `benchmarks/` | Early in-house optimiser implementations (`mineA`/`mineB` depend on these) |

## Run

```bash
python run_all.py 0 2 10     # worker 0 of 2, 10 runs
python run_all.py 1 2 10
python analyse.py
```

For the paper setting, use `RUNS=50`. A 10-run sweep takes roughly 1–1.5 h on 2 cores.

Optional equivalence check (the original `psokmeans.m` profile versus the Master reproduction):

```bash
python run_all.py 0 1 5 orig_check
```

This writes `results_orig_check_0.csv`. Compare its rows with `master_repro` rows at the same seeds; they should match exactly because the profiles share the same update path.

## Outputs

- `results_<worker>.csv` — one row per `(case, algorithm, run)`
- `all_results_merged.csv` — merged results
- Console tables: mean/SD/min/max per case, average rank by mean error, mean time, mean evaluations

## Notes

- `orig` is available only in the `orig_check` mode; the regular run uses `fixed` as the closest supplied reference.
- This experiment predates the B1/B2 work and is kept for provenance. The authoritative paper comparison is experiment 01.

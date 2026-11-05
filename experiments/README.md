# Experiments

Each subfolder is a self-contained, restart-safe experiment bundle with its own `ports.py` snapshot, runner, merge/analysis script, and checked-in result files.

| # | Folder | Question | Scale |
|---|---|---|---|
| 00 | [00_version_comparison](00_version_comparison/) | How do the supplied PSOMbest versions differ? | 22 cases × 7 versions × runs |
| 01 | [01_toe_to_toe_50runs](01_toe_to_toe_50runs/) | Head-to-head against the paper with 10 algorithms | 22 × 10 × 50 |
| 02 | [02_shifted_optimum](02_shifted_optimum/) | Do results survive shifting each optimum? | 10 × 6 × 30 |
| 03 | [03_b1_master_loop](03_b1_master_loop/) | B1 variants inside the Master PSO loop | 22 × 3 × 50 |
| 04 | [04_b1_regrouping](04_b1_regrouping/) | Does rebuilding groups at 300–700 help? | 19 × 3 × 50 |
| 05 | [05_timing_and_resources](05_timing_and_resources/) | Wall/CPU time, memory, grouping cost | 6 × 6 × 3 + microbenchmarks |
| 06 | [06_ablation](06_ablation/) | Which PSOMbest features help B1-proj-fade? | 6 × 10 × 2 × 30 |
| 07 | [07_leader_isolation](07_leader_isolation/) | Is PSOMbest's lead caused by its scalar leader? | 6 × 10 × 2 × 30 |
| 08 | [08_scalar_leader](08_scalar_leader/) | Scalar leader applied to all B1 variants | 9 × 19 × 50 + 9 × 10 × 30 |

## Common workflow

```bash
# One process per worker; repeat with worker IDs 0..N-1
python run_all.py <WORKER_ID> <N_WORKERS> <RUNS>

# Combine the shards
python merge.py          # or analyze.py / analyse.py
```

Runners scan existing `results_*.csv` files and skip completed jobs, so they are safe to stop and resume. Job ownership is fixed by position in the full list, so restarting never duplicates or reassigns work.

## Which results are checked in?

Each folder contains the completed result CSVs (per-worker shards and merged tables) and any generated analysis. They are evidence for the tables in [docs/03_results.md](../docs/03_results.md), not required inputs for running the code.

## Quick checks

- `06_ablation/selftest.py` — base config, F2, seed 1
- `07_leader_isolation/selftest.py` — reduced leader-rule check
- `08_scalar_leader/selftest.py` — verifies `scalar=None` reproduces experiment 03 exactly

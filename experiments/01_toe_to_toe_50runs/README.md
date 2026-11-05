# 01 — Toe-to-toe against the paper (50 runs)

**Question:** How do the paper's five algorithms, the corrected PSOMbest, and B1's three variants compare under the paper protocol?

## Design

22 cases × 10 algorithms × 50 runs, `N = 150`, `MaxIt = 1000`, `σ = 0.01`.

| Algorithm | Implementation |
|---|---|
| `pso` | Master-file classical PSO |
| `pso_awdv` | PSO with adaptive weighted delay velocity |
| `psoea` | Evolutionary-initialised PSO |
| `ga` | Real-coded GA |
| `master_repro` | PSOMbest, supplied Master code |
| `psombest_master_vm` | PSOMbest + velocity memory |
| `master_corrected` | PSOMbest with stored-best vector leader + velocity memory |
| `b1_fade` | Frozen k-means groups, fading cluster pull |
| `b1_coord_fade` | Per-coordinate quantile groups, fading pull |
| `b1_proj_fade` | First-principal-axis groups, fading pull |

Every cell has 50 runs. F18 is excluded from the analysis tables because its formula duplicates F4.

## Files

| File | Purpose |
|---|---|
| `ports.py` | Profiles for the PSOMbest family and ports of PSO/PSO-AWDV/PSOEA/GA/B1 |
| `b1_pkg/` | Vectorised B1 optimisers used by the B1 variants |
| `run_all.py` | Restart-safe, run-by-run job sharding |
| `analyze.py` | Merges results and writes paper-layout tables |
| `paper_values.json` | Published `best`/`mae`/`sd`/`worst` values, transcribed from the paper |
| `tables.md` | Generated analysis: reproduction counts, average ranks, B1 head-to-heads, timing |

## Run

```bash
pip install numpy scipy pandas tabulate
python run_all.py 0 8 50     # repeat for worker IDs 0..7
python analyze.py            # writes tables.md and merged.csv
```

Each worker appends to its own `results_w<ID>.csv`. Jobs are ordered run-by-run, so every algorithm has the same number of completed runs at any stopping point; an interrupted sweep can be analysed and resumed.

## Outputs

- `results_w*.csv` — per-worker shards
- `merged.csv` — deduplicated union
- `tables.md` — paper-reported vs re-run MAE/SD/best/worst, reproduction counts, average ranks, B1 win/loss/rank-sum table, mean time per run

## Headline result

B1-fade, B1-coord-fade and B1-proj-fade each beat the reproduced PSOMbest on MAE in **11 of 19 cases** (8 significant at rank-sum *p* < 0.05), but lose to plain PSO on most cases. See the repository [results snapshot](../../docs/03_results.md) for the full discussion.

## Notes

- Algorithm labels in `tables.md` are human-readable; raw `merged.csv` uses the short keys above.
- The `b1_pkg` runs are vectorised and synchronous, while the PSOMbest family is particle-by-particle, so the timing column is **not** implementation-matched. Use experiment 05 for timing.

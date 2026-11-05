# 04 — B1 with regrouping at iterations 300–700

**Question:** Does periodically rebuilding the frozen groups help, or does the frozen structure already capture what matters?

## Design

19 cases (F18 excluded as a duplicate of F4) × 3 algorithms × 50 runs.

Groups are rebuilt at iterations **300, 350, 400, …, 700** (9 regroupings) using each variant's own grouping method, each time from the current positions. The groups are frozen before iteration 300 and after iteration 700. Seeds, cases and settings match experiment 03, so the two sweeps can be compared run by run.

## Files

| File | Purpose |
|---|---|
| `ports.py` | `run_b1_masterloop(..., regroup_iters=...)` and the Master-loop implementations |
| `run_all.py` | Worker runner; writes names like `b1_fade_rg300_700_50` |
| `merge.py` | Merges shards into `merged_regroup.csv` |

## Run

```bash
python run_all.py 0 8 50     # repeat for worker IDs 0..7
python merge.py
```

## Outputs

- `results_w*.csv`
- `merged_regroup.csv`

## Result summary (per-case median error, 19 cases)

| Variant | Frozen (experiment 03) | Regroup | Regroup wins / losses / ties |
|---|---|---|---|
| B1-fade | 0.0016 | 0.0099 | 1 / 10 / 8 |
| B1-coord-fade | 0.0019 | 0.0032 | 11 / 0 / 8 |
| B1-proj-fade | 0.0004 | 0.0100 | 4 / 7 / 8 |

Regrouping helps B1-coord-fade (whose per-coordinate groups can become stale), but hurts B1-fade and is mixed for B1-proj-fade. The frozen design is the safer default.

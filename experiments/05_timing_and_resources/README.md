# 05 — Timing and resource cost

**Question:** What does each algorithm actually cost in wall time, CPU time, memory, and grouping work?

## Design

- **Full runs:** 6 representative cases (`F1` 30-D, `F5` 2-D, `F9` 100-D, `F12` 100-D, `F14` 30-D noisy, `F17_100D` 100-D) × 6 algorithms × 3 repeats, `N = 150`, `MaxIt = 1000`.
- **Memory:** `tracemalloc` peak over a 100-iteration 100-D run per algorithm.
- **Grouping microbenchmark:** the grouping step alone, 1000 iterations, at dimensions 2, 30 and 100.
- **k-means counters:** number of calls and Lloyd iterations per run (via a counted drop-in replacement).

Algorithms: PSO, PSOMbest (Master code), PSOMbest + velocity memory, B1-fade, B1-coord-fade, B1-proj-fade.

Algorithms run one at a time in a single process, with the order rotated every repeat so slow drifts (throttling, background load) hit every algorithm equally.

## Before running (Windows)

1. Plug in the laptop.
2. Settings → System → Power → Power mode = **Best performance**.
3. Close browsers, VS Code and other Python processes.
4. Run from a normal PowerShell/cmd window, **not** a VS Code terminal.
5. Task Manager → Details: `python.exe` must not show "Efficiency mode". The script tries to disable power throttling and prints `power throttling switched off: True`.
6. Do not start multiple workers for this test.

## Files

| File | Purpose |
|---|---|
| `timing_test.py` | The full timing/memory/microbenchmark suite |
| `ports.py` | Algorithm implementations (shared snapshot) |
| `summary.txt` | Generated summary tables |
| `timing_runs.csv`, `memory.csv`, `grouping_micro.csv` | Raw measurements |

## Run

```bash
python timing_test.py            # 3 repeats, 1000 iterations (~20–30 min)
python timing_test.py 1 100      # quick smoke test (~1–2 min)
```

## Outputs

- `timing_runs.csv` — one row per `(case, algorithm, repeat)`
- `memory.csv` — peak memory per algorithm
- `grouping_micro.csv` — grouping-only seconds per dimension
- `summary.txt` — formatted summary, including ratios to PSO

## Headline result

| Algorithm | Ratio to PSO (wall) | k-means calls/run |
|---|---|---|
| PSO | 1.00 | 0 |
| PSOMbest (Master code) | 1.36 | 1000 |
| PSOMbest + velocity memory | 1.27 | 1000 |
| B1-fade | 1.21 | 0 |
| B1-coord-fade | 1.25 | 0 |
| B1-proj-fade | 1.14 | 0 |

The paper reports 1.78× for PSOMbest vs PSO on an i7-7500U in MATLAB. The Python port is faster relative to PSO because the k-means loop is comparatively cheap in NumPy.

## Notes

- Absolute seconds are hardware-dependent. Compare ratios and medians across machines.
- Peak memory is small in absolute terms (< 2 MB traced); the differences are allocation patterns, not practical constraints.

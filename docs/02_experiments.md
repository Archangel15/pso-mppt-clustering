# Experiment catalogue

Every experiment is a self-contained folder under `experiments/`. Each has its own `ports.py`, so it can be copied out and run independently. The shared pattern is:

1. Start `N_WORKERS` processes, one per worker ID.
2. Each worker appends to its own `results_w<ID>.csv` and skips work already present (restart-safe).
3. Run the merge/analysis script to combine the shards.

On Windows, use separate PowerShell or cmd windows rather than VS Code terminals; the terminal can throttle background processes.

```powershell
# Worker w of 8
python run_all.py $w 8 <RUNS>
```

```bash
# Worker w of 8
python run_all.py "$w" 8 <RUNS>
```

---

## 00 — Version comparison

**Folder:** `experiments/00_version_comparison`

**Question:** How do the supplied PSOMbest versions differ, and does the earlier in-house implementation reproduce them?

**Design:** 22 cases × 7 algorithm versions × `RUNS` runs.

| Version | Description |
|---|---|
| `fixed` | Port of `psokmeans_fixed.m`; re-evaluates cluster members to choose a leader |
| `master_repro` | Master code, scalar leader, no velocity memory |
| `master_corrected` | Master code, stored-best vector leader, velocity memory |
| `master_table3` | `master_corrected` with `c1 = 1.5`, `c2 = 2.0` |
| `pso` | Master classical PSO port |
| `mineA` | Earlier in-house version, standard PSO settings (`c1 = 1.5`, `c2 = 2.0`, `Vmax = 0.1×range`) |
| `mineB` | Earlier in-house version with the source settings (`c1 = 0.5`, `c2 = 1.5`, `Vmax = 1×range`) |

**Run:**

```bash
python run_all.py 0 2 10     # worker 0 of 2, 10 runs
python run_all.py 1 2 10
python analyse.py
```

Optional equivalence check:

```bash
python run_all.py 0 1 5 orig_check
```

**Outputs:** `results_<id>.csv`, `all_results_merged.csv`, console tables of mean/SD/best/worst and average rank.

**Notes:** A 10-run sweep takes roughly 1–1.5 h on 2 cores; the paper setting is `RUNS=50`.

---

## 01 — Toe-to-toe against the paper (50 runs)

**Folder:** `experiments/01_toe_to_toe_50runs`

**Question:** How do the paper's five algorithms, the corrected PSOMbest, and B1's three variants compare under the paper protocol?

**Design:** 22 cases × 10 algorithms × 50 runs, `N = 150`, `MaxIt = 1000`, `σ = 0.01`.

| Algorithm | Implementation |
|---|---|
| PSO, PSO-AWDV, PSOEA, GA | Master-file ports |
| PSOMbest (Master code) | `master_repro` |
| PSOMbest + velocity memory | `psombest_master_vm` |
| PSOMbest (corrected leader + velocity memory) | `master_corrected` |
| B1-fade, B1-coord-fade, B1-proj-fade | Vectorised `b1_pkg` runner |

**Run:**

```bash
pip install numpy scipy pandas tabulate
python run_all.py 0 8 50     # through worker 7
python analyze.py            # writes tables.md and merged.csv
```

**Outputs:** `results_w*.csv`, `merged.csv`, `tables.md` (paper-reported vs re-run, MAE/SD/best/worst, reproduction counts, average ranks, B1 head-to-heads, timing).

**Notes:** Jobs are ordered run-by-run, so every algorithm has the same number of completed runs at any stopping point. Restarting resumes automatically.

---

## 02 — Shifted-optimum robustness

**Folder:** `experiments/02_shifted_optimum`

**Question:** Do the algorithms still work when each benchmark's optimum is moved so its best value differs in every coordinate?

**Design:** 10 cases × 6 algorithms × 30 runs. Each benchmark is shifted by `f(x − s)`; `s_j` is drawn once per case from ±25% of the range and clipped to keep the optimum inside the bounds. The optimum *value* is unchanged, but the optimum *point* no longer has equal coordinates.

**Algorithms:** PSO, PSOMbest (Master code), PSOMbest + velocity memory, B1-fade, B1-coord-fade, B1-proj-fade.

**Run:**

```bash
python run_all.py 0 8 30     # through worker 7
python merge.py
```

**Outputs:** `results_w*.csv`, `merged_shift.csv`.

**Why it matters:** This is the control for the scalar-leader hypothesis. A leader that copies one number to every coordinate can only help when the optimum has (near-)equal coordinates.

---

## 03 — B1 inside the Master PSO loop

**Folder:** `experiments/03_b1_master_loop`

**Question:** How do B1-fade, B1-coord-fade and B1-proj-fade perform when run inside the same particle-by-particle loop as the paper's algorithms, rather than the vectorised `b1_pkg` runner?

**Design:** 22 cases × 3 algorithms × 50 runs, `N = 150`, `MaxIt = 1000`, `σ = 0.01`, seeds 1–50. This makes B1 results directly comparable to the Master-code PSOMbest runs (grouping frozen; `c3` fades 1 → 0).

**Run:**

```bash
python run_all.py 0 8 50     # through worker 7
python merge.py
```

**Outputs:** `results_w*.csv`, `merged_ml.csv`.

**Notes:** Algorithm names are written with a `_ml` suffix (`b1_fade_ml`, etc.) to distinguish them from the vectorised runs.

---

## 04 — B1 with regrouping at iterations 300–700

**Folder:** `experiments/04_b1_regrouping`

**Question:** Does periodically rebuilding the frozen groups help, or does the frozen structure already capture what matters?

**Design:** 19 cases (F18 excluded as a duplicate of F4) × 3 algorithms × 50 runs. Groups are rebuilt at iterations 300, 350, 400, …, 700 using each variant's own grouping method, and frozen otherwise. Seeds, cases and settings match experiment 03, so results can be compared run by run.

**Run:**

```bash
python run_all.py 0 8 50     # through worker 7
python merge.py
```

**Outputs:** `results_w*.csv`, `merged_regroup.csv`.

---

## 05 — Timing and resource cost

**Folder:** `experiments/05_timing_and_resources`

**Question:** What does each algorithm actually cost in wall time, CPU time, memory, and grouping work?

**Design:** 6 representative cases (`F1`, `F5`, `F9`, `F12`, `F14`, `F17_100D`) × 6 algorithms × 3 repeats, plus:

- peak memory (`tracemalloc`) for a 100-iteration 100-D run;
- a grouping-only microbenchmark at dimensions 2, 30 and 100;
- k-means call and Lloyd-iteration counters.

**Run:**

```bash
python timing_test.py            # default: 3 repeats, 1000 iterations
python timing_test.py 1 100      # quick smoke test
```

**Before running (Windows):** plug in the laptop, set power mode to "Best performance", close other applications, run outside VS Code, and do not start multiple workers. The script tries to disable Windows power throttling and prints whether it succeeded.

**Outputs:** `timing_runs.csv`, `memory.csv`, `grouping_micro.csv`, `summary.txt`.

---

## 06 — Ablation: which PSOMbest features help B1-proj-fade?

**Folder:** `experiments/06_ablation`

**Question:** Of the features borrowed from the supplied PSOMbest code, which ones actually help the best B1 variant?

**Design:** 6 configurations × 10 settings × {original, shifted} × 30 runs = 3,600 runs.

| Config | Change relative to base |
|---|---|
| `base` | `c1, c2 = 1.5, 2.0`; `Vmax = 0.1×range`; velocity kept; uniform start |
| `c12` | `c1, c2 = 0.5, 1.5` |
| `vmax` | `Vmax = 1.0×range` |
| `novel` | velocity discarded after every move (no inertia) |
| `lhs` | Latin-hypercube start |
| `all4` | all four switches on |

**Run:**

```bash
python selftest.py               # ~20 s sanity check
python run_all.py 0 8 30         # through worker 7
python merge.py
```

**Outputs:** `results_w*.csv`, `merged_ablation.csv`.

**Notes:** All else is identical to `ports.run_b1_masterloop(kind='b1_proj_fade')`: one-off PCA grouping, 5 equal-count cuts, best-personal-best leader, fading `c3`, mirror boundaries, `w` starts at 1 and decays by 0.99.

---

## 07 — Leader isolation

**Folder:** `experiments/07_leader_isolation`

**Question:** Is PSOMbest's behaviour caused by its one-number group leader?

**Design:** PSOMbest settings are fixed; only the leader rule changes. 6 rules × 10 settings × {original, shifted} × 30 runs = 3,600 runs.

| Rule | Description |
|---|---|
| `scalar_first` | Supplied code: first cluster member's first coordinate, copied to every coordinate |
| `scalar_mean` | Cluster mean over all members and coordinates, copied to every coordinate |
| `scalar_best1` | Best member's first coordinate, copied to every coordinate |
| `stored_best` | Full vector of the cluster member with the lowest current cost |
| `pbest_vec` | Full vector of the member with the best personal best |
| `none` | No group pull (`c3 = 0`) |

**Run:**

```bash
python selftest.py               # ~10 s
python run_all.py 0 8 30         # through worker 7
python merge.py
```

**Outputs:** `results_w*.csv`, `merged_leader.csv`.

---

## 08 — Scalar leader applied to B1

**Folder:** `experiments/08_scalar_leader`

**Question:** If B1's three grouping rules get the PSOMbest-style one-number leader, do they beat PSOMbest?

**Design:** 3 grouping rules × 3 scalar rules = 9 configurations.

| Scalar rule | Description |
|---|---|
| `lead0` | First coordinate of the normal leader vector, copied to all coordinates |
| `leadmean` | Mean of the normal leader vector, copied to all coordinates |
| `first` | Supplied PSOMbest rule: first group member's first coordinate |

Scale: 19 original cases × 9 configs × 50 runs = 8,550, plus 10 shifted cases × 9 configs × 30 runs = 2,700 → 11,250 runs total (roughly 3 h on 8 workers).

**Run:**

```bash
python selftest.py                              # ~20 s; must print True three times
python run_all.py 0 8 50 30                     # through worker 7
python merge.py                                 # merged_scalar.csv
python analyse.py                               # results.md
```

Use `python run_all.py 0 8 50 0` to skip the shifted part.

**Outputs:** `results_w*.csv`, `merged_scalar.csv`, `results.md`.

**Dependencies:** numpy, pandas, scipy (rank-sum tests), plus the baseline CSVs `baseline_orig.csv` and `baseline_shift.csv` for `analyse.py`.

---

## Results files included in this repository

The result CSVs from the completed runs are checked in next to each experiment:

| Folder | Result files |
|---|---|
| 01 | `results_w0..7.csv`, `merged.csv`, `tables.md` |
| 02 | `results_w0..7.csv`, `merged_shift.csv` |
| 03 | `results_w0..7.csv`, `merged_ml.csv` |
| 04 | `results_w0..7.csv`, `merged_regroup.csv` |
| 05 | `timing_runs.csv`, `memory.csv`, `grouping_micro.csv`, `summary.txt` |
| 06 | `results_w0..7.csv`, `merged_ablation.csv` |
| 07 | `results_w0..7.csv`, `merged_leader.csv` |
| 08 | `results_w0..7.csv`, `merged_scalar.csv`, `baseline_orig.csv`, `baseline_shift.csv`, `paper_values.json` |

They are provided as evidence and for auditing; re-running a bundle overwrites/appends to the local files.

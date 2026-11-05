# Reproducibility notes, deviations, and limitations

This project values *honest* reproduction: deviations from the reference implementation are flagged in code and documented here rather than silently patched.

---

## 1. MATLAB → Python mapping

| MATLAB | Python |
|---|---|
| `rng(seed,'twister')` | `numpy.random.RandomState(seed)` (MT19937 family; streams are statistically equivalent, **not** bit-identical) |
| `lhsdesign` (toolbox-free) | `local_lhsdesign()` — maximin LHS |
| `kmeans` (toolbox-free) | `local_kmeans()` — k-means++ with Lloyd iterations |
| `writecell` / `xlsread` | `pandas` / `openpyxl` workbook |
| `figure` / `saveas` / `semilogy` | `matplotlib` (`Agg`) |
| `.mat -v7.3` | `.npz` compressed archive |
| `diary` | tee'd log file |

GPU: the source mention an RTX 4050, but no CUDA stack is required. Everything here runs on CPU.

---

## 2. Corrections made to the Master-file port

The file `reference/psombest_main_master_python_corrected.py` was compared line by line with `PSOMbest_Main_Master.m`. Four bugs were found and fixed:

1. **LHS index offset.** MATLAB's `randperm` is 1-based; the first port used 0-based permutation indices, producing design points slightly below 0 (roughly −0.006 to 0.992). Added `+1` so the design lies in (0, 1).
2. **PSO-AWDV inertia and coefficients.** The hook computed the next inertia/`c1`/`c2` but returned the old ones, so inertia stayed at 1.0 and the delay term was always zero. Values are now computed at the end of an iteration and used from the next one, as in MATLAB. Iteration 1 uses `w = 1`, `c1 = 1.5`, `c2 = 2.0`; the evolutionary state is taken at the start of the iteration.
3. **k-means initial labels.** The previous-label vector started as all −1; MATLAB starts with `ones` (all cluster 1). Now starts as index 0.
4. **GA elite rounding.** MATLAB rounds half away from zero; Python's `round` uses half-to-even. The elite count now uses `floor(x + 0.5)`.

Everything else is unchanged. Random streams still differ, so compare statistics rather than individual runs.

---

## 3. Flagged deviation: velocity limit

The source `psokmeans.m` uses:

```
VelMax = 1 * (VarMax - VarMin)
```

The vectorised benchmark code (`reference/benchmarks/optimizers_fast.py`) uses `VEL_MAX_SCALE = 0.1` instead. This is deliberate:

- The literal full-range limit causes severe overshoot and divergence in this implementation across all tested functions.
- Diagnostic example: mean error on 30-D Zakharov drops from **52.5** at scale 1.0 to **9.4e-8** at scale 0.1.
- `0.1` is the standard PSO convention (Eberhart & Shi recommend 0.1–0.2 × range).

The constant and the rationale are in the module docstring. The Master-loop B1 runs in experiments 03/04 and the Master-code PSOMbest runs use the source's full-range limit, so the two families are not mixed silently.

---

## 4. Random seeds and restart safety

- All experiment runners use seeds `1..RUNS` and write a row per `(case, algorithm, run)`.
- A runner scans existing `results_*.csv` files and skips completed jobs.
- Job ownership is fixed by position in the full job list, so restarting or changing the number of workers never reassigns a job.
- Experiment 01 orders jobs run-by-run, so every algorithm has the same number of completed runs at any stopping point. This is what makes partial results comparable.
- Seeds are isolated per run, so multiprocessing scheduling cannot change results.

---

## 5. Environment

Tested with Python 3.14.6 and:

| Package | Version |
|---|---|
| numpy | 2.5.1 |
| pandas | 3.0.3 |
| scipy | 1.18.1 |
| matplotlib | (3.x) |
| openpyxl | (3.x) |

`requirements.txt` specifies minimum versions. Python 3.10+ is required by the `int | None` annotations used in the B2 modules.

Windows-specific notes:

- The timing experiment tries to disable Windows power throttling via `SetProcessInformation`; it prints whether this succeeded.
- Run timing tests on AC power, with "Best performance" mode, and without background applications.
- Use separate PowerShell/cmd windows for multiple workers rather than VS Code terminals, which can throttle background processes.
- `psutil` is optional but recommended for the memory measurement.

---

## 6. Known limitations

### 6.1 Missing B1 baseline module

`variants/b1_baseline/run_b1_experiments.py` imports `b1_static_confined_pso.py`, which was **not present in the supplied snapshot**. The harness (`eval_harness.py`), experiment runner and report are included, and the B2 modules are self-contained. To re-run the baseline harness, add `b1_static_confined_pso.py` (and `scenarios.py` from `pv_testbed/`) to that folder.

The checked-in `report_b1.md` is the historical report from the original run; its referenced plot PNGs are not in the snapshot.

### 6.2 PSOMbest is not fully reproduced

Only 5/19 cases fall within 10× of the published MAE. This is a genuine finding, not hidden. The main contributors are the scalar leader rule, the literal velocity limit, and the random-stream difference. See [03_results.md](03_results.md).

### 6.3 Scalar-leader benchmarks

The original benchmark set has optima with equal (often zero) coordinates. A scalar leader that copies one number to all coordinates is therefore unusually well suited to it. The shifted-optimum experiments (02, 07, 08) exist specifically to expose this. Do not generalise scalar-leader results without the shifted control.

### 6.4 PV testbed simplifications

- One string only; multi-string / DMPPT (separate instances per panel) is not implemented.
- Temperature is a free parameter but no temperature-driven `Isc` drift is modelled (`k_i = 0`). The source appendix lists `k_i = 1.380658e-23 A`, which is numerically Boltzmann's constant and is almost certainly a transcription error; the model uses the standard Boltzmann constant in the diode term and sets `k_i = 0`.
- The idealised bypass-diode treatment clamps a bypassed module to −0.6 V instead of following the avalanche branch.
- `prominence_W` for ground-truth peak counting is a tunable design parameter, not a fixed constant. The adversarial scenario deliberately contains a low-prominence local peak to make this sensitivity testable.
- No physical minimum-peak-width bound has been derived from a specific reference system; the scenarios were tuned empirically.

### 6.5 B2 semantics

B2-confined *does* reduce evaluations by freezing captive particles. B2-dynamic *does not* and is not intended to: it drops a stale leader term but keeps evaluating. Any comparison that treats the two as equivalent compute optimisations is wrong.

### 6.6 Results are hardware-dependent

Wall-clock and memory numbers depend on the machine, Python version, BLAS build and power state. Use ratios and medians, not absolute seconds, when comparing across machines.

---

## 7. Reproducing a subset quickly

```bash
# 1. PV model sanity
cd pv_testbed && python verify.py

# 2. Reference driver smoke test
cd ../reference && python psombest_main_master_python_corrected.py --mode debug

# 3. One experiment, tiny scale (edit RUNS/MAXIT or use the available args)
cd ../experiments/02_shifted_optimum
python run_all.py 0 1 2

# 4. Merge and inspect
python merge.py
```

For quick algorithm checks, the `selftest.py` files in experiments 06/07/08 compare a reduced run against an expected value and catch accidental changes to the random stream or update order.

---

## 8. Adding a new experiment

1. Copy an existing experiment folder that matches the interface you need.
2. Keep its `ports.py` snapshot; do not import across experiment folders (self-containment is intentional).
3. Add the new algorithm as a function with the `(name, lb, ub, d, ...) -> {"best", "n_evals"}` contract.
4. Use the same job-sharding and resume pattern (`j % nw == wid`, scan existing CSVs).
5. Add a `README.md` with question, design, run command, outputs, and caveats.
6. Check in the merged CSV and the analysis output, not just the code.

---

## 9. Source provenance

| Component | Source |
|---|---|
| Master driver, five algorithms | `PSOMbest_Main_Master.m` (Twumasi et al., 2024) |
| Original / fixed PSOMbest versions | `psokmeans.m`, `psokmeans_fixed.m` |
| B1 grouping variants | This project |
| B2 coverage mechanism | This project |
| PV model | Mao et al. (2018), Energy 143:181–190, with the flagged `k_i` correction |
| PSO-AWDV | Xu et al. (2021) settings, ported from the Master file |

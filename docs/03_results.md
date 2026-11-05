# Results snapshot

All numbers are taken from the result files in `experiments/` and `variants/`. They are a summary, not a substitute for the full tables.

> **Statistical caveat.** The MATLAB reference uses a different random stream from NumPy (same MT19937 family, different consumption), so run-level values are not bit-identical. Comparisons here are distributional: medians, means, win/loss counts, and rank-sum tests where available.

---

## 1. Reproduction of the paper (experiment 01)

50 runs per cell, 22 cases, 10 algorithms, `N = 150`, `MaxIt = 1000`, `σ = 0.01`. F18 is excluded from the comparison because its formula duplicates F4.

### Reproduction count (re-run MAE within 10× of published MAE, 19 cases)

| Algorithm | Cases reproduced |
|---|---|
| PSO | 17 / 19 |
| PSO-AWDV | 16 / 19 |
| PSOEA | 17 / 19 |
| GA | 9 / 19 |
| PSOMbest (Master code) | 5 / 19 |

PSO, PSO-AWDV and PSOEA reproduce reasonably. GA and especially PSOMbest do not. The PSOMbest gap is concentrated in high-dimensional and noisy cases (F1, F2, F3, F10, F11, F13, F14, F17), which is consistent with the known deviations:

- literal `Vmax = 1 × range` in the source;
- the scalar leader rule;
- no velocity memory in the Master code;
- a different random stream from MATLAB.

### Average rank on re-run MAE (lower is better)

| Algorithm | Average rank |
|---|---|
| PSO | 3.82 |
| PSOEA | 3.92 |
| GA | 4.03 |
| B1-proj-fade | 4.68 |
| B1-fade | 5.16 |
| B1-coord-fade | 5.37 |
| PSOMbest + velocity memory | 6.21 |
| PSOMbest (Master code) | 6.84 |
| PSO-AWDV | 7.42 |
| PSOMbest (corrected leader + velocity memory) | 7.55 |

### B1 vs PSOMbest (per-case MAE)

| B1 variant | Wins | Losses | Ties | Significantly better (rank-sum *p* < 0.05) |
|---|---|---|---|---|
| B1-fade vs PSOMbest | 11 | 6 | 2 | 8 |
| B1-coord-fade vs PSOMbest | 11 | 6 | 2 | 8 |
| B1-proj-fade vs PSOMbest | 11 | 6 | 2 | 7 |
| B1-fade vs PSO | 2 | 13 | 4 | 1 |
| B1-coord-fade vs PSO | 2 | 13 | 4 | 1 |
| B1-proj-fade vs PSO | 3 | 12 | 4 | 1 |

**Interpretation.** B1 is a clear, statistically supported improvement over the *reproduced* PSOMbest on most cases, and it is much cheaper. It is not a replacement for PSO: plain PSO wins most cases on MAE. B1's value proposition is "better than PSOMbest at a fraction of the grouping cost", not "best optimiser overall".

---

## 2. Timing and resource cost (experiment 05)

Median over 3 repeats, 6 cases (`F1`, `F5`, `F9`, `F12`, `F14`, `F17_100D`), 1000 iterations, `N = 150`, one process.

| Algorithm | Total wall (s) | Ratio to PSO | CPU ratio | k-means calls/run | Lloyd iterations/run |
|---|---|---|---|---|---|
| PSO | 24.6 | 1.00 | 1.00 | 0 | 0 |
| PSOMbest (Master code) | 33.6 | 1.36 | 1.37 | 1000 | 7681 |
| PSOMbest + velocity memory | 31.3 | 1.27 | 1.27 | 1000 | 6094 |
| B1-fade | 29.9 | 1.21 | 1.21 | 0 | 0 |
| B1-coord-fade | 30.8 | 1.25 | 1.25 | 0 | 0 |
| B1-proj-fade | 28.1 | 1.14 | 1.14 | 0 | 0 |

Grouping step alone, 1000 iterations:

| Step | 2-D | 30-D | 100-D |
|---|---|---|---|
| PSOMbest: k-means every iteration | 1.006 s | 1.196 s | 1.119 s |
| B1-fade: label + leader table | 0.042 s | 0.067 s | 0.480 s |
| B1-coord-fade: label + leader table | 0.081 s | 0.160 s | 0.302 s |
| B1-proj-fade: label + leader table | 0.031 s | 0.031 s | 0.039 s |

Peak memory (tracemalloc, F9 100-D, 100 iterations): PSO 0.36 MB, PSOMbest 0.92 MB, B1-fade 1.51 MB, B1-coord-fade 0.72 MB, B1-proj-fade 1.21 MB.

**Interpretation.** The paper reports PSOMbest at 1.78× PSO on an i7-7500U in MATLAB. The Python port gives 1.36× because it is particle-by-particle but the k-means loop is comparatively cheaper in NumPy. B1-proj-fade is the cheapest variant; its grouping is ~30× cheaper than per-iteration k-means at 100-D.

**Caveat.** Experiment 01's `b1_pkg` timings (0.4–0.64 s/run) are not directly comparable to the Master-loop timings above: `b1_pkg` is vectorised and evaluates the whole population per iteration, whereas the Master loop evaluates particle by particle. Use experiment 05 for implementation-matched timing.

---

## 3. B2 coverage freezing (PV testbed)

Full suite: 7 scenarios × 30 seeds, `N = 150`, `MaxIt = 200`, success = voltage error < 0.15 V.

| Algorithm | Mean evaluations | Mean voltage error | Success rate |
|---|---|---|---|
| B1_confined | 30,150 | ~0 V | 100% |
| **B1_confined_B2** | **150** | 0.007–0.064 V | 100% |
| B1_dynamic | 30,150 | ~0 V | 100% |
| B1_dynamic_B2 | 30,150 | 0.007–0.048 V | 100% |
| P&O | 501 | 0.02–4.31 V | 13–100% |
| Standard PSO | 30,150 | ~0 V | 100% |

Per-scenario success for the B2 confined variant is 100% in every scenario, including the adversarial narrow-peak case. Its voltage error is slightly higher than plain B1 because most particles are frozen after the initial coverage sweep; the best personal best is still found.

**Interpretation.** In the confined variant, B2 is a genuine compute optimisation: 150 evaluations instead of 30,150 is a **99.5% reduction**. In the dynamic variant, B2 is deliberately not a compute optimisation: it changes which local signal is trusted, and its evaluation count is identical to B1-dynamic. This asymmetry is a design choice, documented in the module docstrings.

### Budget sweep (selected rows)

| Budget | B1_confined success | B1_confined_B2 success | Evals (B1 / B2) |
|---|---|---|---|
| N=150, MaxIt=200 | 100% | 100% | 30,150 / 150 |
| N=30, MaxIt=50 | 100% | 100% | 1,530 / 1,489 |
| N=15, MaxIt=30 | 100% | 100% | 465 / 465 |
| N=8, MaxIt=20 | 80% | 80% | 168 / 168 |
| N=5, MaxIt=15 | 13% | 13% | 80 / 80 |
| N=3, MaxIt=15 | 0% | 0% | 0 / 0 |

At small budgets the two variants converge to the same behaviour, because the initial scatter does not exhaust enough bins to allow freezing. The advantage appears when the budget is large enough for coverage to complete early.

---

## 4. B1 baseline harness (PV testbed, isolation run)

From `variants/b1_baseline/report_b1.md`: 4 scenarios × 30 seeds, `k = 5`, `N = 150`, `MaxIt = 200`.

| Scenario | Success | Mean voltage error | Mean evaluations |
|---|---|---|---|
| uniform | 100% | 0.0000 V | 30,150 |
| simple_partial | 100% | 0.0000 V | 30,150 |
| complex_partial | 100% | 0.0000 V | 30,150 |
| adversarial_narrow_peak | 100% | 0.0000 V | 30,150 |

**Caveat.** This is B1 in isolation. The report explicitly notes that no plain-PSOMbest or P&O baseline had been run through the same harness at that point, so these numbers are not a comparison. They establish that B1 works on the testbed; the comparative evidence is in experiments 01–04.

---

## 5. Regrouping at iterations 300–700 (experiment 04)

Median absolute error over 19 cases (F18 excluded), frozen groups vs regrouping:

| Variant | Frozen median | Regroup median | Regroup wins / losses / ties (per-case median) |
|---|---|---|---|
| B1-fade | 0.0016 | 0.0099 | 1 / 10 / 8 |
| B1-coord-fade | 0.0019 | 0.0032 | 11 / 0 / 8 |
| B1-proj-fade | 0.0004 | 0.0100 | 4 / 7 / 8 |

**Interpretation.** Regrouping is not a universal win. It helps B1-coord-fade (11 wins, 0 losses), where per-coordinate groups can become stale, but hurts B1-fade and is mixed for B1-proj-fade. The default frozen design is the safer choice.

---

## 6. Ablation: which PSOMbest features help B1-proj-fade? (experiment 06)

Median absolute error across 10 settings; ranks are per-case average ranks (1 = best).

| Config | Original median | Original rank | Shifted median | Shifted rank |
|---|---|---|---|---|
| `base` | 2.19 | 2.00 | 10.38 | 2.10 |
| `lhs` | 2.78 | 2.40 | 12.54 | 2.00 |
| `novel` | 1.90 | 2.50 | 2.43 | 3.20 |
| `vmax` | 17.79 | 3.40 | 22.06 | 3.20 |
| `c12` | 6.83 | 4.80 | 12.95 | 4.60 |
| `all4` | 45.91 | 5.90 | 82.82 | 5.90 |

**Interpretation.** The source defaults (`base`) are already near-optimal. Switching to the PSOMbest acceleration coefficients (`c12`) or the literal full-range velocity limit (`vmax`) hurts. Latin-hypercube initialisation is roughly neutral (slightly better on shifted cases). Discarding velocity (`novel`) is competitive on shifted cases but not a clear win on the original set. Combining all four (`all4`) is catastrophic, which is expected: it stacks two harmful changes on top of each other.

---

## 7. Leader isolation (experiment 07)

Median absolute error across 10 settings, original vs shifted optima:

| Leader rule | Original median | Shifted median |
|---|---|---|
| `scalar_mean` | 4.6e-10 | 385.05 |
| `scalar_best1` | 7.2e-05 | 417.43 |
| `scalar_first` (supplied) | 0.4506 | 538.96 |
| `stored_best` (vector) | 15.64 | 16.78 |
| `pbest_vec` (vector) | 16.10 | 17.48 |
| `none` | 79.66 | 104.61 |

**Interpretation.** The scalar leader is extremely effective on the original benchmarks — `scalar_mean` and `scalar_best1` drive the median error to essentially zero. But it collapses when the optimum is shifted so that its coordinates differ: median errors jump to ~385–539. The vector rules, especially `stored_best`, are almost unchanged (~16–17). This is the clearest evidence that the scalar leader's advantage is tied to the equal-coordinate structure of these benchmarks, not to a general optimisation benefit.

---

## 8. Scalar leader applied to B1 (experiment 08)

Median absolute error for the 9 configurations (3 grouping rules × 3 scalar rules):

| Config | Original median | Shifted median |
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

**Interpretation.** The same pattern repeats: scalar leaders are near-perfect on the original functions and degrade by ~27–37 median error on shifted functions. `leadmean` is the least harmful scalar rule. `analyse.py` in that folder produces the full win/loss/rank-sum comparison against PSOMbest and the vector-leader baselines.

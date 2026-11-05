# B1 baseline — static confined clustering and shared evaluation harness

This folder contains the first evaluation of B1 on the synthetic PV testbed, plus the algorithm-agnostic harness used to measure it.

## Files

| File | Purpose |
|---|---|
| `eval_harness.py` | Shared experiment harness: runs any algorithm with the `(power_fn, v_min, v_max, seed) -> result` interface across scenarios and seeds; records success rate, voltage/power error, execution time and evaluation count; writes CSV, Markdown report and plots |
| `run_b1_experiments.py` | Runs B1 across the four testbed scenarios with 30 seeds each and produces the report |
| `report_b1.md` | Historical report from the first completed run |

## What B1 does

B1 static confined clustering splits the voltage range into `k = 5` contiguous regions. Particles are assigned to a region and **confined** to it for the whole run. There is no global-best term; the social pull is the best personal best inside the particle's own region. This is "Mechanism 1" only — no coverage tracking or reassignment.

The intended narrative is:

1. Confinement prevents the swarm from collapsing onto one peak and preserves diversity across regions.
2. Because particles are captive, a region can later be frozen once it has been fully explored (this is the basis of B2, in `../b2_coverage/`).

## How B1 is called

`run_b1_experiments.py` expects:

```python
from scenarios import uniform, simple_partial, complex_partial, adversarial_narrow_peak
from b1_static_confined_pso import run_b1, B1Params

def b1_runner(power_fn, v_min, v_max, seed):
    return run_b1(power_fn, v_min, v_max, B1Params(k=5, N=150, MaxIt=200, seed=seed))
```

The harness interacts with the algorithm only through:

```python
result.gmpp_voltage
result.gmpp_power
result.best_power_history   # optional, for convergence plots
```

A `CountingPowerFn` wrapper counts every objective call non-invasively, so algorithms never need to know they are being measured.

## Running it

> **Missing module.** `b1_static_confined_pso.py` is not in this snapshot. To re-run, place it in this folder together with `scenarios.py` (from `../../pv_testbed/`), then:

```bash
python run_b1_experiments.py
```

The script writes:

- `results_b1.csv`
- `report_b1.md`
- `plot_convergence_b1.png`
- `plot_errors_b1.png`
- `plot_success_rate_b1.png`
- `plot_exec_time_b1.png`
- `plot_n_evals_b1.png`

## Recorded result (historical)

30 seeds × 4 scenarios, `k = 5`, `N = 150`, `MaxIt = 200`, success tolerance `0.15 V`:

| Scenario | Success | Mean voltage error | Mean evaluations |
|---|---|---|---|
| uniform | 100% | 0.0000 V | 30,150 |
| simple_partial | 100% | 0.0000 V | 30,150 |
| complex_partial | 100% | 0.0000 V | 30,150 |
| adversarial_narrow_peak | 100% | 0.0000 V | 30,150 |

**Caveat from the original report:** this is B1 in isolation. No plain-PSOMbest or P&O baseline had been run through the same harness at that point, so these numbers show that B1 works on the testbed, not that it beats a baseline. The comparative evidence is in `experiments/01_toe_to_toe_50runs/` through `04_b1_regrouping/`.

## Metrics defined by the harness

| Metric | Definition |
|---|---|
| `success` | `|V_found − V_GMPP| < 0.15 V` |
| `v_error` | absolute voltage error, V |
| `p_error_pct` | `100 · |P_found − P_GMPP| / P_GMPP` |
| `n_evals` | number of `power_at_voltage` calls (counted) |
| `exec_time_s` | wall-clock run time |
| `n_true_peaks` | ground-truth peak count for the scenario |

The harness is intentionally algorithm-agnostic: B1 today, and any future baseline, go through the same code so results are directly comparable.

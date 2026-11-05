# B2 — coverage-based (tabu-style) region exhaustion

B2 adds a second mechanism on top of B1: track which voltage bins have been visited, and act when a region is fully covered.

## Files

| File | Purpose |
|---|---|
| `tabu_coverage.py` | `RegionCoverage`: per-region bin visitation and exhaustion flags (passive tracking only) |
| `b1_confined_b2.py` | B2 applied to **confined** B1: freezes exhausted regions and stops evaluating their particles |
| `b1_dynamic_b2.py` | B2 applied to **dynamic** B1: drops the leader pull in exhausted regions but never freezes or skips evaluations |
| `plot_b2_budget_sweep.png` | Budget-sweep figure |
| `results_b2_full_suite.csv` | 7 scenarios × 30 seeds for B1_confined, B1_confined_B2, B1_dynamic, B1_dynamic_B2, P&O, StandardPSO |
| `results_b2_budget_sweep.csv` | Budget sweep over six `(N, MaxIt)` settings |

## Coverage rule

Each region is divided into bins of width **0.3 V**. The width comes from the narrowest verified peak in the adversarial scenario (~0.96 V) divided by 3, a safety margin chosen earlier in the project. A bin is visited the first time any particle position falls inside it; a region is **exhausted** when all its bins have been visited.

The tracker is deliberately passive: it never restricts or redirects movement, and there is no aspiration criterion. "Exhausted" means *fully covered*, not *quality-plateaued*. This reduces the risk of declaring a region done while an unexplored peak remains.

## The two variants are not equivalent

### Confined (`b1_confined_b2.py`)

Particles were already trapped in their region, so "nothing left to discover here" genuinely means "stop spending evaluations":

- freeze all particles in an exhausted region;
- stop passing frozen particles to the fitness function;
- only the active subset is evaluated each iteration.

This is where B2's compute-saving story lives. The initial scatter also seeds coverage, and any region already exhausted at iteration 0 is frozen immediately — a bug found during verification, without which those particles would run the full budget with no saving.

### Dynamic (`b1_dynamic_b2.py`)

Particles were never confined, so there is no captive population to freeze. Instead:

- particles currently in an exhausted region lose the cluster-pull term and fall back to inertia + personal best;
- they keep moving and keep being evaluated;
- they regain a real pull if they wander into an active region.

Consequently, `n_evals` is **identical** to plain B1-dynamic. This is expected and intentional, not a shortcoming: any effect is on accuracy, not compute.

## Result summary (full suite, 30 seeds/scenario)

| Algorithm | Mean evaluations | Mean voltage error | Success |
|---|---|---|---|
| B1_confined | 30,150 | ~0 V | 100% |
| **B1_confined_B2** | **150** | 0.007–0.064 V | 100% |
| B1_dynamic | 30,150 | ~0 V | 100% |
| B1_dynamic_B2 | 30,150 | 0.007–0.048 V | 100% |
| P&O | 501 | 0.02–4.31 V | 13–100% |
| Standard PSO | 30,150 | ~0 V | 100% |

The confined variant cuts evaluations by **99.5%** while preserving 100% success on every scenario. Its voltage error is slightly higher because most particles freeze after the initial coverage sweep, but the best personal best is still retained.

The budget sweep shows the advantage depends on having enough budget for coverage to complete early. At `N = 15, MaxIt = 30` the two variants are identical (465 evaluations); at `N = 8, MaxIt = 20` they are also identical (168) and both drop to 80% success.

## Using it

Both modules expect a 1-D power function:

```python
power_fn(V) -> array of power values
```

and return:

```python
B1B2Result(
    gmpp_voltage, gmpp_power, best_power_history,
    n_regions_exhausted, exhaustion_iteration
)
```

Example with the PV testbed:

```python
import sys
sys.path.append("../../pv_testbed")
from scenarios import adversarial_narrow_peak
from b1_confined_b2 import run_b1_b2, B1B2Params

curve, _ = adversarial_narrow_peak()
v_min, v_max = curve.voltage_range
result = run_b1_b2(curve.power_at_voltage, v_min, v_max,
                   B1B2Params(k=5, N=150, MaxIt=200, seed=0))
print(result.gmpp_voltage, result.gmpp_power,
      result.n_regions_exhausted, result.exhaustion_iteration)
```

The CSV result files are the source of the summary table above; no runner script for the sweep is included in this snapshot, but both result files are checked in for auditing.

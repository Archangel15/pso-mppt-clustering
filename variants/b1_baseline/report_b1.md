# B1 (Static Confined Clustering) — Experiment Report

Generated: 2026-09-17 14:21:57  
Total trials: 120  
Algorithms: B1 (k=5, N=150, MaxIt=200)  
Scenarios: uniform, simple_partial, complex_partial, adversarial_narrow_peak  
Seeds per (algorithm, scenario): 30-30

## Notes
First recorded run of B1 (static confined clustering, no coverage tracking or reassignment yet -- Mechanism 1 only). Parameters: k=5 fixed regions, N=150 particles (inherited from the verified PSOMbest baseline), MaxIt=200, tolerance for a 'success' = 0.15 V from the true GMPP voltage. No comparison baseline (plain PSOMbest, P&O) has been run through this harness yet -- these numbers are B1 in isolation, not yet a comparison.

## Summary (per algorithm x scenario)

| algorithm | scenario | n_trials | success_rate_pct | v_error_mean | v_error_std | v_error_max | p_error_pct_mean | p_error_pct_std | exec_time_mean_s | exec_time_std_s | n_evals_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 (k=5, N=150, MaxIt=200) | adversarial_narrow_peak | 30 | 100.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0128 | 0.0003 | 30150.00 |
| B1 (k=5, N=150, MaxIt=200) | complex_partial | 30 | 100.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0133 | 0.0016 | 30150.00 |
| B1 (k=5, N=150, MaxIt=200) | simple_partial | 30 | 100.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0130 | 0.0005 | 30150.00 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 30 | 100.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0127 | 0.0002 | 30150.00 |

## Plots

**Convergence (best power vs. iteration, 3 seeds per scenario)**

![Convergence (best power vs. iteration, 3 seeds per scenario)](plot_convergence_b1.png)

**Voltage error distribution**

![Voltage error distribution](plot_errors_b1.png)

**GMPP success rate**

![GMPP success rate](plot_success_rate_b1.png)

**Execution time**

![Execution time](plot_exec_time_b1.png)

**Function evaluations**

![Function evaluations](plot_n_evals_b1.png)

## Raw per-trial records

Full data: see accompanying CSV. First 10 rows shown here for reference.

| algorithm | scenario | seed | success | v_error | p_error_pct | exec_time_s | n_evals |
|---|---|---|---|---|---|---|---|
| B1 (k=5, N=150, MaxIt=200) | uniform | 0 | True | 0.0000 | 0.0000 | 0.0129 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 1 | True | 0.0000 | 0.0000 | 0.0126 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 2 | True | 0.0000 | 0.0000 | 0.0127 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 3 | True | 0.0000 | 0.0000 | 0.0126 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 4 | True | 0.0000 | 0.0000 | 0.0127 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 5 | True | 0.0000 | 0.0000 | 0.0126 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 6 | True | 0.0000 | 0.0000 | 0.0127 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 7 | True | 0.0000 | 0.0000 | 0.0127 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 8 | True | 0.0000 | 0.0000 | 0.0127 | 30150 |
| B1 (k=5, N=150, MaxIt=200) | uniform | 9 | True | 0.0000 | 0.0000 | 0.0126 | 30150 |
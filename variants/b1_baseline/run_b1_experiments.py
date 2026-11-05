"""
run_b1_experiments.py

Runs B1 across all testbed scenarios with multiple seeds, through the
shared evaluation harness, and produces:
  - results_b1.csv           (every trial, raw)
  - report_b1.md             (summary tables + embedded plots)
  - plot_convergence_b1.png
  - plot_errors_b1.png
  - plot_success_rate_b1.png
  - plot_exec_time_b1.png
  - plot_n_evals_b1.png

Run this after any change to b1_static_confined_pso.py to get an
updated, dated record for the thesis write-up.
"""

from scenarios import uniform, simple_partial, complex_partial, adversarial_narrow_peak
from b1_static_confined_pso import run_b1, B1Params
import eval_harness as eh

OUT_DIR = "."  # overwritten to the real output dir by the caller/copy step

N_SEEDS = 30
SEEDS = list(range(N_SEEDS))

scenarios = {
    "uniform": uniform(),
    "simple_partial": simple_partial(),
    "complex_partial": complex_partial(),
    "adversarial_narrow_peak": adversarial_narrow_peak(),
}


def b1_runner(power_fn, v_min, v_max, seed):
    params = B1Params(k=5, N=150, MaxIt=200, seed=seed)
    return run_b1(power_fn, v_min, v_max, params)


runners = {"B1 (k=5, N=150, MaxIt=200)": b1_runner}

print(f"Running B1 across {len(scenarios)} scenarios x {N_SEEDS} seeds "
      f"({len(scenarios) * N_SEEDS} trials)...")

df, histories = eh.run_experiment(runners, scenarios, SEEDS)

print(f"Done. {len(df)} trial records collected.")
df.to_csv(f"{OUT_DIR}/results_b1.csv", index=False)
print(f"Saved raw results -> results_b1.csv")

summary = eh.summarize(df)
print("\n=== Summary ===")
print(summary.to_string(index=False))

eh.plot_convergence(histories, f"{OUT_DIR}/plot_convergence_b1.png")
eh.plot_error_boxplots(df, f"{OUT_DIR}/plot_errors_b1.png")
eh.plot_bar_metric(summary, "success_rate_pct", "success rate (%)",
                    "GMPP success rate by scenario", f"{OUT_DIR}/plot_success_rate_b1.png")
eh.plot_bar_metric(summary, "exec_time_mean_s", "mean execution time (s)",
                    "Execution time by scenario", f"{OUT_DIR}/plot_exec_time_b1.png")
eh.plot_bar_metric(summary, "n_evals_mean", "mean function evaluations",
                    "Function evaluations by scenario", f"{OUT_DIR}/plot_n_evals_b1.png")

print("Saved plots.")

notes = (
    "First recorded run of B1 (static confined clustering, no coverage tracking "
    "or reassignment yet -- Mechanism 1 only). Parameters: k=5 fixed regions, "
    "N=150 particles (inherited from the verified PSOMbest baseline), MaxIt=200, "
    "tolerance for a 'success' = 0.15 V from the true GMPP voltage. "
    "No comparison baseline (plain PSOMbest, P&O) has been run through this "
    "harness yet -- these numbers are B1 in isolation, not yet a comparison."
)

eh.write_markdown_report(
    summary, df, f"{OUT_DIR}/report_b1.md",
    title="B1 (Static Confined Clustering) — Experiment Report",
    plot_paths={
        "Convergence (best power vs. iteration, 3 seeds per scenario)": "plot_convergence_b1.png",
        "Voltage error distribution": "plot_errors_b1.png",
        "GMPP success rate": "plot_success_rate_b1.png",
        "Execution time": "plot_exec_time_b1.png",
        "Function evaluations": "plot_n_evals_b1.png",
    },
    notes=notes,
)
print(f"\nReport written -> report_b1.md")

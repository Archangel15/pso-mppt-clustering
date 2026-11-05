"""
eval_harness.py

Shared experiment-tracking harness -- runs any algorithm with the
common (power_fn, v_min, v_max, seed) -> result interface across a set
of scenarios and seeds, and records the metrics from the concept
document's evaluation plan (Section 8): success rate, voltage/power
error, execution time, and function-evaluation count. Produces a CSV
of raw per-trial records, a Markdown summary report with aggregated
tables, and diagnostic plots.

This is written to be algorithm-agnostic: B1 today, B2/B3/baselines
(P&O, standard PSO, fixed-k PSOMbest) later, all go through the same
harness so results are directly comparable -- this is the "shared
evaluation/instrumentation harness" item from the concept document's
roadmap (Section 9).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CountingPowerFn:
    """Wraps a scenario's power_at_voltage so every call is counted --
    this is the function-evaluation metric, measured non-invasively
    (the algorithm under test never needs to know it's being counted)."""
    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, V):
        V_arr = np.atleast_1d(V)
        self.count += V_arr.size
        return self.fn(V)


@dataclass
class TrialRecord:
    algorithm: str
    scenario: str
    seed: int
    success: bool
    v_error: float
    p_error_pct: float
    gmpp_true_v: float
    gmpp_true_p: float
    gmpp_found_v: float
    gmpp_found_p: float
    n_evals: int
    exec_time_s: float
    n_true_peaks: int


def run_trial(runner, algorithm_name, curve, scenario_name, seed,
              tolerance_v: float = 0.15):
    """
    runner: callable(power_fn, v_min, v_max, seed) -> object with
            .gmpp_voltage, .gmpp_power, .best_power_history
    Returns (TrialRecord, best_power_history_or_None).
    """
    gt = curve.ground_truth()
    v_min, v_max = curve.voltage_range
    counting_fn = CountingPowerFn(curve.power_at_voltage)

    t0 = time.perf_counter()
    result = runner(counting_fn, v_min, v_max, seed)
    elapsed = time.perf_counter() - t0

    v_err = abs(result.gmpp_voltage - gt["gmpp_voltage"])
    p_err_pct = 100.0 * abs(result.gmpp_power - gt["gmpp_power"]) / gt["gmpp_power"]
    success = v_err < tolerance_v

    record = TrialRecord(
        algorithm=algorithm_name,
        scenario=scenario_name,
        seed=seed,
        success=success,
        v_error=v_err,
        p_error_pct=p_err_pct,
        gmpp_true_v=gt["gmpp_voltage"],
        gmpp_true_p=gt["gmpp_power"],
        gmpp_found_v=result.gmpp_voltage,
        gmpp_found_p=result.gmpp_power,
        n_evals=counting_fn.count,
        exec_time_s=elapsed,
        n_true_peaks=gt["peak_count"],
    )
    history = getattr(result, "best_power_history", None)
    return record, history


def run_experiment(runners: dict, scenarios: dict, seeds: list,
                    tolerance_v: float = 0.15, n_histories_per_cell: int = 3):
    """
    runners: {algorithm_name: callable(power_fn, v_min, v_max, seed) -> result}
    scenarios: {scenario_name: (curve, label)}
    seeds: list of ints, reused across every (algorithm, scenario) pair.

    Returns (results_df, histories) where histories is
    {(algorithm, scenario): [(seed, history_array), ...]} for the first
    n_histories_per_cell seeds only (kept small deliberately -- this is
    for convergence plots, not bulk storage).
    """
    records = []
    histories = {}

    for algo_name, runner in runners.items():
        for scen_name, (curve, label) in scenarios.items():
            key = (algo_name, scen_name)
            histories[key] = []
            for seed in seeds:
                record, history = run_trial(runner, algo_name, curve, scen_name,
                                             seed, tolerance_v)
                records.append(record)
                if history is not None and len(histories[key]) < n_histories_per_cell:
                    histories[key].append((seed, history))

    df = pd.DataFrame([asdict(r) for r in records])
    return df, histories


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (algorithm, scenario) aggregate stats -- mean/std/min/max plus
    success rate, matching the concept document's stated statistical
    reporting practice (Section 8)."""
    grouped = df.groupby(["algorithm", "scenario"])
    summary = grouped.agg(
        n_trials=("seed", "count"),
        success_rate_pct=("success", lambda s: 100.0 * s.mean()),
        v_error_mean=("v_error", "mean"),
        v_error_std=("v_error", "std"),
        v_error_max=("v_error", "max"),
        p_error_pct_mean=("p_error_pct", "mean"),
        p_error_pct_std=("p_error_pct", "std"),
        exec_time_mean_s=("exec_time_s", "mean"),
        exec_time_std_s=("exec_time_s", "std"),
        n_evals_mean=("n_evals", "mean"),
    ).reset_index()
    return summary


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_convergence(histories: dict, out_path: str):
    """One panel per (algorithm, scenario) cell, showing best-power-so-far
    across iterations for a few representative seeds."""
    keys = list(histories.keys())
    n = len(keys)
    ncols = min(3, n) if n > 0 else 1
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        algo, scen = key
        for seed, hist in histories[key]:
            ax.plot(hist, label=f"seed {seed}", lw=1.2)
        ax.set_title(f"{algo} — {scen}", fontsize=9)
        ax.set_xlabel("iteration")
        ax.set_ylabel("best power (W)")
        ax.legend(fontsize=7)

    for idx in range(len(keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_error_boxplots(df: pd.DataFrame, out_path: str):
    """Voltage-error distribution per scenario (one box per algorithm,
    grouped by scenario)."""
    scenarios = df["scenario"].unique()
    algorithms = df["algorithm"].unique()
    fig, ax = plt.subplots(figsize=(1.8 * len(scenarios) * max(1, len(algorithms)) + 2, 5))

    positions = []
    data = []
    labels = []
    pos = 0
    group_gap = 1.5
    for scen in scenarios:
        for algo in algorithms:
            subset = df[(df.scenario == scen) & (df.algorithm == algo)]["v_error"]
            if len(subset) == 0:
                continue
            data.append(subset.values)
            labels.append(f"{algo}\n{scen}")
            positions.append(pos)
            pos += 1
        pos += group_gap

    ax.boxplot(data, positions=positions, widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("voltage error (V)")
    ax.set_title("GMPP voltage error distribution by scenario")
    ax.axhline(0.15, color="red", linestyle="--", linewidth=1, label="success tolerance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_bar_metric(summary: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: str):
    """Generic grouped bar chart: one group per scenario, one bar per algorithm."""
    scenarios = summary["scenario"].unique()
    algorithms = summary["algorithm"].unique()
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(algorithms))

    fig, ax = plt.subplots(figsize=(1.6 * len(scenarios) + 2, 4.5))
    for i, algo in enumerate(algorithms):
        vals = [summary[(summary.scenario == s) & (summary.algorithm == algo)][metric].values[0]
                if len(summary[(summary.scenario == s) & (summary.algorithm == algo)]) else np.nan
                for s in scenarios]
        ax.bar(x + i * width, vals, width, label=algo)

    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(scenarios, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def write_markdown_report(summary: pd.DataFrame, df: pd.DataFrame, out_path: str,
                           title: str, plot_paths: dict, notes: str = ""):
    lines = [f"# {title}", ""]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"Total trials: {len(df)}  ")
    lines.append(f"Algorithms: {', '.join(df['algorithm'].unique())}  ")
    lines.append(f"Scenarios: {', '.join(df['scenario'].unique())}  ")
    lines.append(f"Seeds per (algorithm, scenario): {df.groupby(['algorithm','scenario']).size().min()}"
                  f"-{df.groupby(['algorithm','scenario']).size().max()}")
    lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append(notes)
        lines.append("")

    lines.append("## Summary (per algorithm x scenario)")
    lines.append("")
    cols = ["algorithm", "scenario", "n_trials", "success_rate_pct",
            "v_error_mean", "v_error_std", "v_error_max",
            "p_error_pct_mean", "p_error_pct_std",
            "exec_time_mean_s", "exec_time_std_s", "n_evals_mean"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in summary.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for caption, path in plot_paths.items():
            lines.append(f"**{caption}**")
            lines.append("")
            lines.append(f"![{caption}]({path})")
            lines.append("")

    lines.append("## Raw per-trial records")
    lines.append("")
    lines.append(f"Full data: see accompanying CSV. First 10 rows shown here for reference.")
    lines.append("")
    preview_cols = ["algorithm", "scenario", "seed", "success", "v_error",
                     "p_error_pct", "exec_time_s", "n_evals"]
    lines.append("| " + " | ".join(preview_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(preview_cols)) + "|")
    for _, row in df[preview_cols].head(10).iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in preview_cols]
        lines.append("| " + " | ".join(vals) + " |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

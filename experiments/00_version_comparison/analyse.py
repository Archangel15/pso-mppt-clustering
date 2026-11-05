"""Merge results_*.csv and print mean / SD / best / worst per case and version, plus average rank."""
import glob, pandas as pd
df = pd.concat([pd.read_csv(f) for f in glob.glob("results_[0-9]*.csv")])
df["best"] = df["best"].astype(float)
order = ["fixed", "master_repro", "master_corrected", "master_table3", "pso", "mineA", "mineB"]
for stat in ["mean", "std", "min", "max"]:
    print(f"\n=== {stat} of final best cost over runs ===")
    print(df.groupby(["case", "algo"]).best.agg(stat).unstack()[order].to_string(float_format=lambda v: f"{v:.4g}"))
m = df.groupby(["case", "algo"]).best.mean().unstack()[order]
print("\n=== average rank by mean error (1 = best) ===\n", m.rank(axis=1).mean().round(2).to_string())
print("\n=== mean time per run (s) ===\n", df.groupby("algo").time_s.mean()[order].round(2).to_string())
print("\n=== mean objective evaluations per run ===\n", df.groupby("algo").n_evals.mean()[order].round(0).to_string())
df.to_csv("all_results_merged.csv", index=False)

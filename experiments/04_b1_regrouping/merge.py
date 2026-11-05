"""python merge.py -> merged_regroup.csv (all results_*.csv, duplicates dropped). Send this file back."""
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("results_*.csv")]).drop_duplicates(["case", "algo", "run"])
d.to_csv("merged_regroup.csv", index=False); print(d.groupby("algo").size())

"""python merge.py -> merged_shift.csv (duplicates dropped). Send this file back."""
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("results_*.csv")]).drop_duplicates(["case", "algo", "run"])
d.to_csv("merged_shift.csv", index=False); print(d.groupby("algo").size())

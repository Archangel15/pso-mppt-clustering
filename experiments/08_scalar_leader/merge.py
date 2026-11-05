"""python merge.py -> merged_scalar.csv (duplicates dropped). Send this file back."""
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("results_*.csv")]).drop_duplicates(["case", "version", "kind", "scalar", "run"])
d.to_csv("merged_scalar.csv", index=False); print(d.groupby(["version", "kind", "scalar"]).size())

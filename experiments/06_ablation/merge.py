"""python merge.py -> merged_ablation.csv (duplicates dropped). Send this file back."""
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("results_*.csv")]).drop_duplicates(["case", "version", "config", "run"])
d.to_csv("merged_ablation.csv", index=False); print(d.groupby(["version", "config"]).size())

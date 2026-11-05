"""Merge results_*.csv and print paper-layout tables (paper-reported vs re-run) with 50-run statistics.
usage: python analyze.py [paper_values.json]   -> writes tables.md and merged.csv"""
import glob, json, sys, numpy as np, pandas as pd
from scipy.stats import ranksums
P = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "paper_values.json"))
paper = {s: pd.DataFrame(P[s]) for s in P}
OPT = {l: 0.0 for l in paper["best"].index}; OPT["F7"] = -1.0; OPT["F12"] = -3916.16599
d = pd.concat([pd.read_csv(f) for f in glob.glob("results_*.csv")]).drop_duplicates(["case", "algo", "run"])
d["best"] = d.best.astype(float); d["err"] = (d.best - d.case.map(OPT)).abs(); d.to_csv("merged.csv", index=False)
NAME = {"pso": "PSO", "pso_awdv": "PSO-AWDV", "psoea": "PSOEA", "ga": "GA", "master_repro": "PSOMbest (Master code)",
        "psombest_master_vm": "PSOMbest (Master code + velocity memory)", "master_corrected": "PSOMbest (corrected leader + velocity memory)",
        "b1_fade": "B1-fade", "b1_coord_fade": "B1-coord-fade", "b1_proj_fade": "B1-proj-fade"}
PAPER = {"pso": "PSO", "pso_awdv": "PSO-AWDV", "psoea": "PSOEA", "ga": "GA", "master_repro": "PSOMbest"}
cases = [c for c in paper["best"].index if c not in ("F18_2D", "F18_30D", "F18_100D")]  # F18 duplicates F4's formula
g = d.groupby(["case", "algo"])
S = {"mae": g.err.mean().unstack(), "sd": g.best.apply(lambda x: x.std(ddof=0)).unstack(),
     "best": g.best.min().unstack(), "worst": g.best.max().unstack(), "n": g.size().unstack()}
out = []
for stat, title in [("mae", "MAE"), ("sd", "SD"), ("best", "Best"), ("worst", "Worst")]:
    cols = [a for a in NAME if a in S[stat]]
    t = pd.DataFrame(index=cases)
    for a in cols:
        if a in PAPER: t[f"{PAPER[a]} (paper-reported)"] = paper[stat][PAPER[a]].reindex(cases)
        t[NAME[a] + " (re-run)"] = S[stat][a].reindex(cases)
    out.append(f"## {title}\n\n" + t.to_markdown(floatfmt=".4g") + "\n")
nmin = int(S["n"].reindex(cases).min().min()) if S["n"].reindex(cases).notna().all().all() else 0
out.insert(0, f"**Runs per cell: min {nmin}, max {int(S['n'].max().max())}. Statistics are comparable only when every cell has the same number of runs (see table at the end).**\n")
out.append("## Runs per cell\n\n" + S["n"].reindex(cases).to_markdown() + "\n")
# reproduction: re-run MAE within 10x of paper MAE (or both <1e-8)
rep = []
for a, pn in PAPER.items():
    if a not in S["mae"]: continue
    r, p = S["mae"][a].reindex(cases), paper["mae"][pn].reindex(cases)
    ok = ((r <= 10 * p) & (r >= p / 10)) | ((r < 1e-8) & (p < 1e-8))
    rep.append((NAME[a], int(ok.sum()), len(cases)))
out.append("## Reproduction of the paper's MAE (within 10x either way)\n\n" + pd.DataFrame(rep, columns=["algorithm", "cases reproduced", "of"]).to_markdown(index=False) + "\n")
# average rank of the re-run algorithms (ties <=1e-10 equal) and wins vs paper-reported PSOMbest MAE
M = S["mae"].reindex(cases); R = M.apply(lambda r: r.round(10).rank(method="average"), axis=1)
out.append("## Average rank on re-run MAE (all re-run algorithms together, lower is better)\n\n" + R.mean().rename(index=NAME).sort_values().to_frame("avg rank").to_markdown(floatfmt=".2f") + "\n")
w = []
for a in ["b1_fade", "b1_coord_fade", "b1_proj_fade"]:
    if a not in M: continue
    for ref in ["master_repro", "psombest_master_vm", "pso"]:
        if ref not in M: continue
        x, y = d[d.algo == a], d[d.algo == ref]
        wins = sum(M.loc[c, a] < M.loc[c, ref] * 0.999999 for c in cases); loss = sum(M.loc[c, a] > M.loc[c, ref] * 1.000001 for c in cases)
        sig = sum(ranksums(x[x.case == c].best, y[y.case == c].best).pvalue < 0.05 and M.loc[c, a] < M.loc[c, ref] for c in cases)
        w.append((NAME[a], NAME[ref], wins, loss, len(cases) - wins - loss, sig))
out.append("## B1 variants vs references (cases by MAE; last column = significantly better, rank-sum p<0.05)\n\n" + pd.DataFrame(w, columns=["B1 variant", "versus", "wins", "losses", "ties", "sig. better"]).to_markdown(index=False) + "\n")
t = d.groupby("algo").time_s.mean().rename(index=NAME); out.append("## Mean time per run (s) and ratio to PSO\n\n" + pd.DataFrame({"s": t, "x PSO": t / t["PSO"]}).to_markdown(floatfmt=".2f") + "\n")
open("tables.md", "w").write("\n".join(out)); print("\n".join(out))

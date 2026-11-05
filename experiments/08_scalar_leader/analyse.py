"""python analyse.py [merged_scalar.csv] -> results.md
Compares the 9 scalar configs with: the paper's published PSOMbest, our re-run of the supplied PSOMbest code (master_repro),
plain PSO, and the normal vector-leader B1 versions (earlier Master-loop runs).  error = |best - optimum|, errors < 1e-10 = 0."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import ranksums
import ports
f = sys.argv[1] if len(sys.argv) > 1 else "merged_scalar.csv"
S = pd.read_csv(f); S["algo"] = S.kind.str.replace("_fade", "") + "+" + S.scalar
P = json.load(open("paper_values.json"))["mae"]["PSOMbest"]
def err(df):
    e = (df.best - df.case.map(ports.PAPER_OPT)).abs(); return e.where(e >= 1e-10, 0.0)
BO = pd.read_csv("baseline_orig.csv"); BS = pd.read_csv("baseline_shift.csv")
BO["algo"] = BO.algo.replace({"master_repro": "PSOMbest(ours)", "pso": "PSO(ours)", "b1_fade_ml": "b1 vector", "b1_coord_fade_ml": "coord vector", "b1_proj_fade_ml": "proj vector"})
BS["algo"] = BS.algo.replace({"master_repro": "PSOMbest(ours)", "pso": "PSO(ours)", "b1_fade": "b1 vector", "b1_coord_fade": "coord vector", "b1_proj_fade": "proj vector"})
BS = BS[BS.algo.isin(["PSOMbest(ours)", "PSO(ours)", "b1 vector", "coord vector", "proj vector"])]
def fmt(x): return "0" if x == 0 else f"{x:.3g}"
out = []
for ver, base in (("orig", BO), ("shift", BS)):
    s = S[S.version == ver].copy()
    if s.empty: continue
    s["err"] = err(s); b = base[base.case.isin(s.case.unique())].copy(); b["err"] = err(b)
    allr = pd.concat([s[["case", "algo", "run", "err"]], b[["case", "algo", "run", "err"]]])
    cases = [c for c in ports.CASES if c in s.case.unique()]
    cfg = sorted(s.algo.unique()); ref = ["PSO(ours)", "PSOMbest(ours)", "b1 vector", "coord vector", "proj vector"]
    mean = allr.groupby(["case", "algo"]).err.mean().unstack(); med = allr.groupby(["case", "algo"]).err.median().unstack()
    n = allr.groupby("algo").run.nunique()
    out.append(f"\n## {'ORIGINAL' if ver=='orig' else 'SHIFTED'} functions  (runs per setting: scalar configs {int(s.groupby(['case','algo']).size().min())})\n")
    for title, tab in (("Mean error", mean), ("Median error", med)):
        cols = ref + cfg
        hdr = "| Setting | " + ("Paper PSOMbest | " if ver == "orig" else "") + " | ".join(cols) + " |"
        out.append(f"\n### {title}\n\n{hdr}\n|" + "---|" * (len(cols) + 1 + (ver == "orig")))
        for c in cases:
            row = [c] + ([fmt(P[c.split('_')[0] if c.startswith('F18') else c])] if ver == "orig" else [])
            row += [fmt(tab.loc[c, a]) if a in tab.columns and not np.isnan(tab.loc[c, a]) else "-" for a in cols]
            out.append("| " + " | ".join(row) + " |")
    out.append("\n### Does each scalar config beat PSOMbest?  (settings where it is better / same / worse;  same = both below 1e-10, or rank-sum p >= 0.05)\n")
    out.append("| Config | vs PSOMbest(ours) by mean | vs PSOMbest(ours) by median | vs PSOMbest(ours) rank-sum p<0.05 (better/same/worse) |" + (" vs paper PSOMbest MAE (better/worse; settings where both are zero are skipped) |" if ver == "orig" else "") + "\n|---|---|---|---|" + ("---|" if ver == "orig" else ""))
    for a in cfg + ["b1 vector", "coord vector", "proj vector"]:
        r = {"mean": [0, 0, 0], "med": [0, 0, 0], "p": [0, 0, 0]}; pb = [0, 0]
        for c in cases:
            x = allr[(allr.case == c) & (allr.algo == a)].sort_values("run").err.values
            y = allr[(allr.case == c) & (allr.algo == "PSOMbest(ours)")].sort_values("run").err.values
            if len(x) == 0 or len(y) == 0: continue
            def cmp(u, v): return 1 if (u < v and not (u < 1e-10 and v < 1e-10)) else (2 if (u > v and not (u < 1e-10 and v < 1e-10)) else 0)
            k = cmp(x.mean(), y.mean()); r["mean"][{1: 0, 0: 1, 2: 2}[k]] += 1
            k = cmp(np.median(x), np.median(y)); r["med"][{1: 0, 0: 1, 2: 2}[k]] += 1
            if np.all(x < 1e-10) and np.all(y < 1e-10): r["p"][1] += 1
            else:
                p = ranksums(x, y).pvalue
                r["p"][1 if p >= 0.05 else (0 if x.mean() < y.mean() else 2)] += 1
            if ver == "orig":
                pv = P[c.split('_')[0] if c.startswith('F18') else c]
                if not (x.mean() < 1e-10 and pv < 1e-10): pb[0 if x.mean() < pv else 1] += 1
        out.append(f"| {a} | {r['mean'][0]}/{r['mean'][1]}/{r['mean'][2]} | {r['med'][0]}/{r['med'][1]}/{r['med'][2]} | {r['p'][0]}/{r['p'][1]}/{r['p'][2]} |" + (f" {pb[0]}/{pb[1]} |" if ver == "orig" else ""))
open("results.md", "w").write("\n".join(out)); print("\n".join(out))

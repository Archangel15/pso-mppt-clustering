"""Arch's B1 variants (c3 fade kept) in the Master-file PSO loop, with the groups REBUILT at iterations 300, 350, 400, ..., 700
(9 regroupings, each from the current positions, with the variant's own grouping method); frozen before 300 and after 700.
Same seeds (1..50), cases and settings as the frozen-group runs, so the two can be compared run by run.
usage: python run_all.py WORKER_ID N_WORKERS [RUNS=50]      (one process per worker, ids 0..N-1)"""
import sys, os, time, csv, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ports import CASES, run_b1_masterloop
wid, nw = int(sys.argv[1]), int(sys.argv[2])
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
SCHEDULE = list(range(300, 701, 50))
CASE_LIST = [c for c in CASES if not c.startswith("F18")]          # F18 duplicates F4, excluded
ALGOS = ["b1_fade", "b1_coord_fade", "b1_proj_fade"]
done = set()
for fn in glob.glob("results_*.csv"):
    for r in csv.DictReader(open(fn)): done.add((r["case"], r["algo"], int(r["run"])))
jobs = [(c, a, r) for r in range(1, RUNS + 1) for c in CASE_LIST for a in ALGOS]
out = f"results_w{wid}.csv"; new = not os.path.exists(out)
fh = open(out, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["case", "algo", "run", "best", "n_evals", "time_s"]); fh.flush()
for j, (c, a, r) in enumerate(jobs):
    name = a + "_rg300_700_50"
    if j % nw != wid or (c, name, r) in done: continue
    lb, ub, d = CASES[c]; t = time.time()
    res = run_b1_masterloop(c, lb, ub, d, a, MaxIt=1000, seed=r, regroup_iters=SCHEDULE)
    wr.writerow([c, name, r, repr(res["best"]), res["n_evals"], round(time.time() - t, 3)]); fh.flush()
print("WORKER DONE", wid, flush=True)

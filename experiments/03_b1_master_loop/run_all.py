"""Arch's three B1 variants inside the Master-file PSO loop: 22 cases x 3 algorithms x 50 runs (seeds 1..50).
usage: python run_all.py WORKER_ID N_WORKERS [RUNS=50] [MAXIT=1000]   (run one process per worker, ids 0..N-1)
Job ownership is fixed by position in the full job list; restarting resumes and never duplicates."""
import sys, os, time, csv, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ports import CASES, run_b1_masterloop
wid, nw = int(sys.argv[1]), int(sys.argv[2])
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
MAXIT = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
ALGOS = ["b1_fade", "b1_coord_fade", "b1_proj_fade"]
done = set()
for fn in glob.glob("results_*.csv"):
    for r in csv.DictReader(open(fn)): done.add((r["case"], r["algo"], int(r["run"])))
jobs = [(c, a, r) for r in range(1, RUNS + 1) for c in CASES for a in ALGOS]
out = f"results_w{wid}.csv"; new = not os.path.exists(out)
fh = open(out, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["case", "algo", "run", "best", "n_evals", "time_s"]); fh.flush()
for j, (c, a, r) in enumerate(jobs):
    if j % nw != wid or (c, a, r) in done: continue
    lb, ub, d = CASES[c]; t = time.time()
    res = run_b1_masterloop(c, lb, ub, d, a, MaxIt=MAXIT, seed=r)
    wr.writerow([c, a + "_ml", r, repr(res["best"]), res["n_evals"], round(time.time() - t, 3)]); fh.flush()
print("WORKER DONE", wid, flush=True)

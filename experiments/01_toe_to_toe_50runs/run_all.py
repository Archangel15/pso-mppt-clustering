"""Toe-to-toe run against Twumasi et al. (2024): same 22 cases, 150 particles, 1000 iterations, 50 runs.
usage: python run_all.py WORKER_ID N_WORKERS [RUNS=50] [MAXIT=1000]
Jobs are ordered run-by-run (run 1 for every case and algorithm, then run 2, ...), so you can stop at any time and
analyse what exists; every algorithm then has the same number of runs. Re-running resumes; existing results_*.csv files
(including the earlier 10-run files) are reused."""
import sys, os, time, csv, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ports import *
wid, nw = int(sys.argv[1]), int(sys.argv[2])
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
MAXIT = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
ALGOS = ["pso", "pso_awdv", "psoea", "ga", "master_repro",              # the five algorithms of the paper (Master file code)
         "psombest_master_vm", "master_corrected",                      # Master PSOMbest WITH velocity memory (future experiment)
         "b1_fade", "b1_coord_fade", "b1_proj_fade"]                    # Arch's algorithms
done = set()
for fn in glob.glob("results_*.csv"):
    for r in csv.DictReader(open(fn)): done.add((r["case"], r["algo"], int(r["run"])))
# job ownership is fixed by the position in the FULL list, so restarting (or changing RUNS) never moves a job to another worker
jobs = [(c, a, r) for r in range(1, RUNS + 1) for c in CASES for a in ALGOS]
out = f"results_w{wid}.csv"; new = not os.path.exists(out)
fh = open(out, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["case", "algo", "run", "best", "n_evals", "time_s"]); fh.flush()
for j, (c, a, r) in enumerate(jobs):
    if j % nw != wid or (c, a, r) in done: continue
    lb, ub, d = CASES[c]; t = time.time()
    if a in PROFILES: res = run_family(a, c, lb, ub, d, MaxIt=MAXIT, seed=r)
    elif a == "pso": res = run_master_pso(c, lb, ub, d, MaxIt=MAXIT, seed=r)
    elif a == "pso_awdv": res = run_pso_awd(c, lb, ub, d, MaxIt=MAXIT, seed=r)
    elif a == "psoea": res = run_psoea(c, lb, ub, d, MaxIt=MAXIT, seed=r)
    elif a == "ga": res = run_ga(c, lb, ub, d, MaxIt=MAXIT, seed=r)
    else: res = run_b1(c, lb, ub, d, a, MaxIt=MAXIT, seed=r)
    wr.writerow([c, a, r, repr(res["best"]), res["n_evals"], round(time.time() - t, 3)]); fh.flush()
print("WORKER DONE", wid, flush=True)

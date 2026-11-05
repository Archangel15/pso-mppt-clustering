"""Shifted-optimum test: 10 cases x 6 algorithms x 30 runs (seeds 1..30), 150 particles, 1000 iterations, sigma 0.01.
usage: python run_all.py WORKER_ID N_WORKERS [RUNS=30]     (one process per worker, ids 0..N-1; restart-safe)"""
import sys, os, time, csv, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ports, shifted
shifted.install()
wid, nw = int(sys.argv[1]), int(sys.argv[2])
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
CASE_LIST = ["F2", "F3", "F9", "F10", "F11", "F12", "F14", "F16", "F17_30D", "F17_100D"]
ALGOS = ["pso", "master_repro", "psombest_master_vm", "b1_fade", "b1_coord_fade", "b1_proj_fade"]
done = set()
for fn in glob.glob("results_*.csv"):
    for r in csv.DictReader(open(fn)): done.add((r["case"], r["algo"], int(r["run"])))
jobs = [(c, a, r) for r in range(1, RUNS + 1) for c in CASE_LIST for a in ALGOS]
out = f"results_w{wid}.csv"; new = not os.path.exists(out)
fh = open(out, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["case", "algo", "run", "best", "n_evals", "time_s"]); fh.flush()
for j, (c, a, r) in enumerate(jobs):
    if j % nw != wid or (c, a, r) in done: continue
    lb, ub, d = ports.CASES[c]; t = time.time()
    if a == "pso": res = ports.run_master_pso(c, lb, ub, d, seed=r)
    elif a in ports.PROFILES: res = ports.run_family(a, c, lb, ub, d, seed=r)
    else: res = ports.run_b1_masterloop(c, lb, ub, d, a, seed=r)
    wr.writerow([c, a, r, repr(res["best"]), res["n_evals"], round(time.time() - t, 3)]); fh.flush()
print("WORKER DONE", wid, flush=True)

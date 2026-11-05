"""Leader isolation: 6 leader rules x 10 settings x {original, shifted} x 30 runs (seeds 1..30) = 3600 runs.
usage: python run_all.py WORKER_ID N_WORKERS [RUNS=30]     (one process per worker; restart-safe)"""
import sys, os, time, csv, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ports, shifted, leader_test
wid, nw = int(sys.argv[1]), int(sys.argv[2])
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
CASE_LIST = ["F2", "F3", "F9", "F10", "F11", "F12", "F14", "F16", "F17_30D", "F17_100D"]
VERSIONS = ["orig", "shift"]
done = set()
for fn in glob.glob("results_*.csv"):
    for r in csv.DictReader(open(fn)): done.add((r["case"], r["version"], r["config"], int(r["run"])))
jobs = [(c, v, g, r) for r in range(1, RUNS + 1) for c in CASE_LIST for v in VERSIONS for g in leader_test.LEADERS]
out = f"results_w{wid}.csv"; new = not os.path.exists(out)
fh = open(out, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["case", "version", "config", "run", "best", "n_evals", "time_s"]); fh.flush()
for j, (c, v, g, r) in enumerate(jobs):
    if j % nw != wid or (c, v, g, r) in done: continue
    ports.objective = shifted.shifted if v == "shift" else shifted._orig
    lb, ub, d = ports.CASES[c]; t = time.time()
    res = leader_test.run_leader(g, c, lb, ub, d, seed=r)
    wr.writerow([c, v, g, r, repr(res["best"]), res["n_evals"], round(time.time() - t, 3)]); fh.flush()
print("WORKER DONE", wid, flush=True)

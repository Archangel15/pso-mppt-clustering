"""Scalar-leader experiment.  9 configs (3 grouping rules x 3 scalar rules) x
   * 19 original settings (F18 excluded: duplicate of F4) x RUNS runs (seeds 1..RUNS, default 50)  = 8550 runs
   * 10 shifted settings x SHIFT_RUNS runs (seeds 1..SHIFT_RUNS, default 30)                       = 2700 runs
usage: python run_all.py WORKER_ID N_WORKERS [RUNS=50] [SHIFT_RUNS=30]   (one process per worker; restart-safe)
Use SHIFT_RUNS=0 to skip the shifted part.  Jobs go run-number by run-number, so you can stop at any time and analyse."""
import sys, os, time, csv, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ports, shifted, scalar_test
wid, nw = int(sys.argv[1]), int(sys.argv[2])
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
SHIFT_RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
ORIG = [c for c in ports.CASES if not c.startswith("F18")]
SHIFT = ["F2", "F3", "F9", "F10", "F11", "F12", "F14", "F16", "F17_30D", "F17_100D"]
CONFIGS = [(k, s) for k in scalar_test.KINDS for s in scalar_test.SCALARS]
done = set()
for fn in glob.glob("results_*.csv"):
    for r in csv.DictReader(open(fn)): done.add((r["case"], r["version"], r["kind"], r["scalar"], int(r["run"])))
jobs = []
for r in range(1, max(RUNS, SHIFT_RUNS) + 1):
    if r <= RUNS:       jobs += [(c, "orig", k, s, r) for c in ORIG for k, s in CONFIGS]
    if r <= SHIFT_RUNS: jobs += [(c, "shift", k, s, r) for c in SHIFT for k, s in CONFIGS]
out = f"results_w{wid}.csv"; new = not os.path.exists(out)
fh = open(out, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["case", "version", "kind", "scalar", "run", "best", "n_evals", "time_s"]); fh.flush()
orig_obj = ports.objective
for j, (c, v, k, s, r) in enumerate(jobs):
    if j % nw != wid or (c, v, k, s, r) in done: continue
    ports.objective = shifted.shifted if v == "shift" else orig_obj
    lb, ub, d = ports.CASES[c]; t = time.time()
    res = scalar_test.run_scalar(k, s, c, lb, ub, d, seed=r)
    wr.writerow([c, v, k, s, r, repr(res["best"]), res["n_evals"], round(time.time() - t, 3)]); fh.flush()
print("WORKER DONE", wid, flush=True)

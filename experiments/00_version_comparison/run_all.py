import sys, os, time, csv
from ports import *
wid, nw, RUNS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ALGOS = ["fixed", "master_repro", "master_corrected", "master_table3", "pso", "mineA", "mineB"]
if len(sys.argv) > 4 and sys.argv[4] == "orig_check":
    ALGOS = ["orig"]; CASE_LIST = ["F8", "F9", "F13", "F3"]; out = f"results_orig_check_{wid}.csv"
else:
    CASE_LIST = list(CASES); out = f"results_{wid}.csv"
done = set()
if os.path.exists(out):
    for r in csv.DictReader(open(out)): done.add((r["case"], r["algo"], int(r["run"])))
jobs = [(c, a, r) for c in CASE_LIST for a in ALGOS for r in range(1, RUNS + 1)]
new = not os.path.exists(out)
fh = open(out, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["case", "algo", "run", "best", "n_evals", "time_s"]); fh.flush()
for j, (c, a, r) in enumerate(jobs):
    if j % nw != wid or (c, a, r) in done: continue
    lb, ub, d = CASES[c]; t = time.time()
    if a in PROFILES: res = run_family(a, c, lb, ub, d, seed=r)
    elif a == "pso": res = run_master_pso(c, lb, ub, d, seed=r)
    else: res = run_mine(c, lb, ub, d, seed=r, variant=a[-1])
    wr.writerow([c, a, r, repr(res["best"]), res["n_evals"], round(time.time() - t, 3)]); fh.flush()
print("WORKER DONE", wid, flush=True)

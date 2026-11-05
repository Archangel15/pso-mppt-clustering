"""Timing / resource test: PSO vs PSOMbest (Master-code port) vs the three B1 fade variants.
All algorithms run in ONE process, one at a time, round-robin (order rotated every repeat) so that slow drifts
(throttling, background load) hit every algorithm equally.  Nothing here changes any algorithm: the functions are
the ones used for the 50-run tables.

usage:  python timing_test.py [REPS] [MAXIT]      (default 3 repeats, 1000 iterations = the paper protocol)
output: timing_runs.csv  memory.csv  grouping_micro.csv  and a summary printed at the end (also summary.txt)
"""
import sys, time, os, csv, platform
import numpy as np
import ports

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
N = 150
CASES = ["F1", "F5", "F9", "F12", "F14", "F17_100D"]          # 30-D, 2-D, 100-D, 100-D, 30-D noisy, 100-D
ALGOS = ["pso", "master_repro", "psombest_master_vm", "b1_fade", "b1_coord_fade", "b1_proj_fade"]
LABEL = {"pso": "PSO", "master_repro": "PSOMbest (Master code)", "psombest_master_vm": "PSOMbest + velocity memory",
         "b1_fade": "B1-fade", "b1_coord_fade": "B1-coord-fade", "b1_proj_fade": "B1-proj-fade"}

# ---------------------------------------------------------------- keep Windows from throttling this process
def no_throttle():
    try:
        import psutil
        p = psutil.Process()
        if platform.system() == "Windows": p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception: pass
    if platform.system() == "Windows":
        try:
            import ctypes
            class ST(ctypes.Structure): _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
            st = ST(1, 1, 0)       # EXECUTION_SPEED control, state 0 = throttling OFF
            ok = ctypes.windll.kernel32.SetProcessInformation(ctypes.windll.kernel32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))
            print("power throttling switched off:", bool(ok))
        except Exception as e: print("could not switch power throttling off:", e)
no_throttle()

# ---------------------------------------------------------------- k-means call / iteration tally (copy of ports.local_kmeans + counters)
TALLY = {"calls": 0, "iters": 0}
def kmeans_counted(data, k, max_iter, tol, rng):
    n, d = data.shape; k = min(max(1, round(k)), n); TALLY["calls"] += 1
    centers = np.zeros((k, d)); centers[0] = data[rng.integers(n)]
    closest = np.sum((data - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        tot = closest.sum()
        if not np.isfinite(tot) or tot <= np.finfo(float).eps: sel = rng.integers(n)
        else:
            thr = rng.random() * tot; cs = np.cumsum(closest); idx = np.nonzero(cs >= thr)[0]
            sel = idx[0] if len(idx) else n - 1
        centers[c] = data[sel]; closest = np.minimum(closest, np.sum((data - centers[c]) ** 2, axis=1))
    lab = np.full(n, -1)
    for _ in range(max_iter):
        TALLY["iters"] += 1
        sqd = np.stack([np.sum((data - centers[c]) ** 2, axis=1) for c in range(k)], axis=1)
        newlab = sqd.argmin(axis=1); dist = sqd[np.arange(n), newlab]
        for c in range(k):
            if not np.any(newlab == c):
                counts = np.bincount(newlab, minlength=k); elig = counts[newlab] > 1
                ed = np.where(elig, dist, -np.inf); f = int(ed.argmax())
                if not np.isfinite(ed[f]): f = int(rng.integers(n))
                newlab[f] = c; dist[f] = 0
        newc = centers.copy()
        for c in range(k):
            m = data[newlab == c]
            if len(m): newc[c] = m.mean(axis=0)
        ch = np.max(np.sqrt(np.sum((newc - centers) ** 2, axis=1))); same = np.array_equal(newlab, lab)
        centers, lab = newc, newlab
        if same or ch <= tol: break
    return lab, centers
ports.local_kmeans = kmeans_counted          # run_family looks this name up in the ports module

def run_algo(algo, case, seed, maxit):
    lb, ub, d = ports.CASES[case]
    if algo == "pso": return ports.run_master_pso(case, lb, ub, d, N=N, MaxIt=maxit, seed=seed)
    if algo in ports.PROFILES: return ports.run_family(algo, case, lb, ub, d, N=N, MaxIt=maxit, seed=seed)
    return ports.run_b1_masterloop(case, lb, ub, d, algo, N=N, MaxIt=maxit, seed=seed)

def timed(algo, case, seed, maxit):
    TALLY["calls"] = TALLY["iters"] = 0
    t0, c0 = time.perf_counter(), time.process_time()
    r = run_algo(algo, case, seed, maxit)
    return time.perf_counter() - t0, time.process_time() - c0, r["n_evals"], TALLY["calls"], TALLY["iters"], r["best"]

# ---------------------------------------------------------------- 1. full runs
print(f"python {platform.python_version()}, numpy {np.__version__}, {platform.processor() or platform.machine()}")
print(f"warm-up ...", flush=True)
for a in ALGOS: timed(a, "F1", 0, 20)
rows = []
for case in CASES:
    for rep in range(REPS):
        order = ALGOS[rep % len(ALGOS):] + ALGOS[:rep % len(ALGOS)]
        for a in order:
            wall, cpu, ne, kc, ki, best = timed(a, case, 1000 + rep, MAXIT)
            rows.append(dict(case=case, algo=a, rep=rep, wall_s=wall, cpu_s=cpu, n_evals=ne, kmeans_calls=kc, kmeans_iters=ki, best=best))
            print(f"{case:9s} rep{rep} {a:20s} {wall:8.2f} s  evals={ne}  kmeans calls={kc} iters={ki}", flush=True)
with open("timing_runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---------------------------------------------------------------- 2. peak memory (100 iterations, 100-D case)
import tracemalloc
try:
    import psutil; proc = psutil.Process()
except Exception: proc = None
mem = []
for a in ALGOS:
    tracemalloc.start(); rss0 = proc.memory_info().rss if proc else float("nan")
    timed(a, "F9", 7, 100)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    rss1 = proc.memory_info().rss if proc else float("nan")
    mem.append(dict(algo=a, tracemalloc_peak_MB=peak / 2**20, rss_change_MB=(rss1 - rss0) / 2**20))
    print(f"memory {a:20s} peak {peak/2**20:7.2f} MB", flush=True)
with open("memory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(mem[0])); w.writeheader(); w.writerows(mem)

# ---------------------------------------------------------------- 3. the grouping step alone (no particle updates, no objective)
def micro(d, iters=MAXIT, k=5, n=N):
    rng = np.random.default_rng(3); pos = rng.uniform(-5, 5, (n, d)); pb_x = pos.copy(); pb_c = rng.random(n); bx = pos[0]
    out = {}
    t = time.perf_counter()
    for _ in range(iters): ports.local_kmeans(pos, k, 100, 1e-10, rng)       # what PSOMbest does EVERY iteration
    out["PSOMbest: k-means every iteration"] = time.perf_counter() - t
    cen = pos[rng.choice(n, size=k, replace=False)].copy(); qs = np.linspace(0, 1, k + 1)[1:-1]
    t = time.perf_counter()
    for _ in range(iters):                                                   # B1-fade, frozen centres
        lab = np.argmin(((pos[:, None, :] - cen[None]) ** 2).sum(2), axis=1); tab = np.empty((k, d))
        for g in range(k):
            m = lab == g; tab[g] = pb_x[np.where(m)[0][np.argmin(pb_c[m])]] if m.any() else bx
    out["B1-fade: label + leader table"] = time.perf_counter() - t
    cuts = np.quantile(pos, qs, axis=0); ar = np.arange(d)
    t = time.perf_counter()
    for _ in range(iters):                                                   # B1-coord-fade
        lab = (pos[:, None, :] > cuts[None]).sum(1); tab = np.empty((k, d))
        for g in range(k):
            mask = lab == g; idx = np.argmin(np.where(mask, pb_c[:, None], np.inf), axis=0)
            tab[g] = np.where(mask.any(0), pb_x[idx, ar], bx)
    out["B1-coord-fade: label + leader table"] = time.perf_counter() - t
    mu = pos.mean(0); u = np.linalg.svd(pos - mu, full_matrices=False)[2][0]; pc = np.quantile((pos - mu) @ u, qs)
    t = time.perf_counter()
    for _ in range(iters):                                                   # B1-proj-fade
        lab = np.searchsorted(pc, (pos - mu) @ u); tab = np.empty((k, d))
        for g in range(k):
            m = lab == g; tab[g] = pb_x[np.where(m)[0][np.argmin(pb_c[m])]] if m.any() else bx
    out["B1-proj-fade: label + leader table"] = time.perf_counter() - t
    return out
micro_rows = []
for d in (2, 30, 100):
    for name, s in micro(d).items():
        micro_rows.append(dict(dim=d, step=name, seconds_for_all_iterations=s)); print(f"grouping only d={d:3d} {name:40s} {s:7.3f} s", flush=True)
with open("grouping_micro.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(micro_rows[0])); w.writeheader(); w.writerows(micro_rows)

# ---------------------------------------------------------------- summary
import pandas as pd
df = pd.DataFrame(rows); out = []
med = df.groupby(["algo", "case"]).wall_s.median().unstack()
medc = df.groupby(["algo", "case"]).cpu_s.median().unstack()
tot = med.sum(axis=1); totc = medc.sum(axis=1)
out.append(f"Full runs: median over {REPS} repeats, {MAXIT} iterations, N={N}, cases {', '.join(CASES)}\n")
out.append("| Algorithm | Total wall s | Ratio to PSO | CPU s | CPU ratio | k-means calls/run | Lloyd iterations/run |\n|---|---|---|---|---|---|---|")
for a in ALGOS:
    g = df[df.algo == a]
    out.append(f"| {LABEL[a]} | {tot[a]:.1f} | {tot[a]/tot['pso']:.2f} | {totc[a]:.1f} | {totc[a]/totc['pso']:.2f} | {g.kmeans_calls.mean():.0f} | {g.kmeans_iters.mean():.0f} |")
out.append("\nRatio to PSO per case (median wall time)\n")
out.append("| Algorithm | " + " | ".join(CASES) + " |\n|" + "---|" * (len(CASES) + 1))
for a in ALGOS: out.append(f"| {LABEL[a]} | " + " | ".join(f"{med.loc[a, c] / med.loc['pso', c]:.2f}" for c in CASES) + " |")
out.append("\nPeak memory (tracemalloc, F9 100-D, 100 iterations)\n")
out.append("| Algorithm | Peak MB |\n|---|---|")
for m in mem: out.append(f"| {LABEL[m['algo']]} | {m['tracemalloc_peak_MB']:.2f} |")
out.append(f"\nGrouping step alone, {MAXIT} iterations (seconds)\n")
mt = pd.DataFrame(micro_rows).pivot(index="step", columns="dim", values="seconds_for_all_iterations")
out.append(mt.round(3).to_markdown())
out.append("\nPaper (i7-7500U, MATLAB): PSO 84.52 s, PSOMbest 150.29 s -> PSOMbest = 1.78 x PSO.")
txt = "\n".join(out); open("summary.txt", "w", encoding="utf-8").write(txt); print("\n" + txt)
print("\nDone. Send me summary.txt, timing_runs.csv, memory.csv and grouping_micro.csv.")

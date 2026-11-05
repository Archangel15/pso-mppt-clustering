"""Python ports of every PSOMbest version supplied by Arch, plus the earlier in-house version.
Source files: psokmeans.m ("orig"), psokmeans_fixed.m ("fixed"), PSOMbest_Main_Master.m
("master_repro", "master_corrected", "master_table3", "master_pso").  No MATLAB/Octave here, so these are
line-by-line ports, not executed originals: random streams differ from MATLAB, so compare statistically.
"""
import numpy as np

# ---------------------------------------------------------------- Master's objective functions (rows)
CASES = {  # name: (lb, ub, dim)
    "F1": (-65.536, 65.536, 30), "F2": (-5, 10, 30), "F3": (-10, 10, 30), "F4": (-1, 1, 30),
    "F5": (-100, 100, 2), "F6": (-10, 10, 2), "F7": (-5.12, 5.12, 2), "F8": (-5.12, 5.12, 2),
    "F9": (-32.768, 32.768, 100), "F10": (-10, 10, 100), "F11": (-600, 600, 100), "F12": (-5, 5, 100),
    "F13": (-100, 400, 30), "F14": (-5.12, 5.12, 30), "F15": (-32, 32, 30), "F16": (-600, 600, 30),
    "F17_2D": (-5, 10, 2), "F17_30D": (-5, 10, 30), "F17_100D": (-5, 10, 100),
    "F18_2D": (-1, 1, 2), "F18_30D": (-1, 1, 30), "F18_100D": (-1, 1, 100)}
PAPER_OPT = {k: 0.0 for k in CASES}; PAPER_OPT["F7"] = -1.0; PAPER_OPT["F12"] = -3916.16599


def objective(x, name, sigma, rng):
    x = np.atleast_2d(x); d = x.shape[1]; b = name.split("_")[0]
    if b == "F1": y = np.sum(x ** 2 * np.arange(d, 0, -1), axis=1)
    elif b == "F2":
        s = np.sum(0.5 * np.arange(1, d + 1) * x, axis=1); y = np.sum(x ** 2, axis=1) + s ** 2 + s ** 4
    elif b == "F3": y = (x[:, 0] - 1) ** 2 + np.sum(np.arange(2, d + 1) * (2 * x[:, 1:] ** 2 - x[:, :-1]) ** 2, axis=1)
    elif b in ("F4", "F18"): y = np.sum(np.abs(x) ** np.arange(2, d + 2), axis=1)
    elif b == "F5": y = x[:, 0] ** 2 + 2 * x[:, 1] ** 2 - 0.3 * np.cos(3 * np.pi * x[:, 0]) - 0.4 * np.cos(4 * np.pi * x[:, 1]) + 0.7
    elif b == "F6": y = 0.26 * (x[:, 0] ** 2 + x[:, 1] ** 2) - 0.48 * x[:, 0] * x[:, 1]
    elif b == "F7":
        r2 = x[:, 0] ** 2 + x[:, 1] ** 2; y = -(1 + np.cos(12 * np.sqrt(r2))) / (0.5 * r2 + 2)
    elif b == "F8": y = 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)
    elif b == "F9": y = -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2, axis=1))) - np.exp(np.mean(np.cos(2 * np.pi * x), axis=1)) + 20 + np.e
    elif b == "F10":
        w = 1 + (x - 1) / 4
        y = (np.sin(np.pi * w[:, 0]) ** 2 + np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)
             + (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2))
    elif b == "F11": y = 1 + np.sum(x ** 2, axis=1) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=1)
    elif b == "F12": y = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1)
    elif b in ("F13", "F14", "F15", "F16"):
        xn = x + sigma * rng.standard_normal(x.shape)
        if b == "F13": y = np.sum(xn ** 2, axis=1)
        elif b == "F14": y = 10 * d + np.sum(xn ** 2 - 10 * np.cos(2 * np.pi * xn), axis=1)
        elif b == "F15": y = -20 * np.exp(-0.2 * np.sqrt(np.mean(xn ** 2, axis=1))) - np.exp(np.mean(np.cos(2 * np.pi * xn), axis=1)) + 20 + np.e
        else: y = 1 + np.sum(xn ** 2, axis=1) / 4000 - np.prod(np.cos(xn / np.sqrt(np.arange(1, d + 1))), axis=1)
    elif b == "F17": y = np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)
    else: raise ValueError(name)
    y = np.real(y); y = np.where(np.isfinite(y), y, np.finfo(float).max); return y


# ---------------------------------------------------------------- helpers copied from the Master file
def local_lhs(n, d, trials, rng):
    best, bd = None, -np.inf
    for _ in range(trials):
        c = np.empty((n, d))
        for j in range(d):
            c[:, j] = (rng.permutation(n) + 1 - rng.random(n)) / n
        sq = np.sum(c ** 2, axis=1); D = sq[:, None] + sq[None, :] - 2 * c @ c.T
        np.fill_diagonal(D, np.inf); md = np.sqrt(max(0.0, D.min()))
        if md > bd: bd, best = md, c
    return best


def local_kmeans(data, k, max_iter, tol, rng):
    n, d = data.shape; k = min(max(1, round(k)), n)
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


class Counter:
    def __init__(self, name, sigma, rng): self.name, self.sigma, self.rng, self.n = name, sigma, rng, 0
    def __call__(self, x):
        x = np.atleast_2d(x); self.n += x.shape[0]; return objective(x, self.name, self.sigma, self.rng)


# ---------------------------------------------------------------- the asynchronous PSOMbest family
PROFILES = {
    # name: leader rule, store velocity, c1, c2, c3, vmax_factor, assignment
    "orig":             dict(leader="scalar_first", store_vel=False, c=(0.5, 1.5, 1.0), vmax=1.0, assign="nearest_center", lhs=5),
    "fixed":            dict(leader="reeval_best",  store_vel=False, c=(0.5, 1.5, 1.0), vmax=1.0, assign="label", lhs=5),
    "master_repro":     dict(leader="scalar_first", store_vel=False, c=(0.5, 1.5, 1.0), vmax=1.0, assign="label", lhs=5),
    "psombest_master_vm": dict(leader="scalar_first", store_vel=True,  c=(0.5, 1.5, 1.0), vmax=1.0, assign="label", lhs=5),
    "master_corrected": dict(leader="stored_best",  store_vel=True,  c=(0.5, 1.5, 1.0), vmax=1.0, assign="label", lhs=5),
    "master_table3":    dict(leader="stored_best",  store_vel=True,  c=(1.5, 2.0, 1.0), vmax=1.0, assign="label", lhs=5),
}


def run_family(profile, name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0, k=5, w0=1.0, wdamp=0.99):
    P = PROFILES[profile]; rng = np.random.default_rng(seed); f = Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    pos = lo + local_lhs(N, d, P["lhs"], rng) * span
    vel = np.zeros((N, d)); cost = np.empty(N)
    best_s, best_x = np.inf, np.zeros(d)
    for i in range(N):
        cost[i] = f(pos[i])[0]
        if cost[i] < best_s: best_s, best_x = cost[i], pos[i].copy()
    pb_x, pb_c = pos.copy(), cost.copy()
    vmax = P["vmax"] * span; vmin = -vmax; w = w0; c1, c2, c3 = P["c"]
    for it in range(MaxIt):
        lab, centers = local_kmeans(pos, k, 100, 1e-10, rng)
        leaders = np.zeros((k, d))
        for c in range(k):
            mem = np.nonzero(lab == c)[0]
            if len(mem) == 0: leaders[c] = best_x
            elif P["leader"] == "scalar_first": leaders[c] = pos[mem[0], 0]
            elif P["leader"] == "stored_best": leaders[c] = pos[mem[np.argmin(cost[mem])]]
            else:  # reeval_best: psokmeans_fixed.m re-evaluates every member's current position
                cc = np.array([f(pos[m])[0] for m in mem]); leaders[c] = pos[mem[np.argmin(cc)]]
        for i in range(N):
            if P["assign"] == "nearest_center":
                ci = int(np.argmin(np.linalg.norm(centers - pos[i], axis=1)))
            else: ci = lab[i]
            v = (w * vel[i] + c1 * rng.random(d) * (pb_x[i] - pos[i]) + c2 * rng.random(d) * (best_x - pos[i])
                 + c3 * rng.random(d) * (leaders[ci] - pos[i]))
            v = np.minimum(np.maximum(v, vmin), vmax)
            p = np.minimum(np.maximum(pos[i] + v, lo), hi)
            pos[i] = p; vel[i] = v if P["store_vel"] else 0.0
            cost[i] = f(p)[0]
            if cost[i] < pb_c[i]:
                pb_c[i] = cost[i]; pb_x[i] = p
                if cost[i] < best_s: best_s, best_x = cost[i], p.copy()
        w *= wdamp
    return dict(best=float(best_s), n_evals=f.n)


def run_master_pso(name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0, w0=1.0, wdamp=0.99, c1=1.5, c2=2.0, vf=0.1):
    """run_classical_pso from the Master file (uniform init, mirror boundary velocity flip, velocity stored)."""
    rng = np.random.default_rng(seed); f = Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    pos = lo + rng.random((N, d)) * span; vel = np.zeros((N, d)); cost = np.empty(N)
    best_s, best_x = np.inf, np.zeros(d)
    for i in range(N):
        cost[i] = f(pos[i])[0]
        if cost[i] < best_s: best_s, best_x = cost[i], pos[i].copy()
    pb_x, pb_c = pos.copy(), cost.copy(); vmax = vf * span; vmin = -vmax; w = w0
    for it in range(MaxIt):
        for i in range(N):
            v = w * vel[i] + c1 * rng.random(d) * (pb_x[i] - pos[i]) + c2 * rng.random(d) * (best_x - pos[i])
            v = np.minimum(np.maximum(v, vmin), vmax); p = pos[i] + v
            out = (p < lo) | (p > hi); v = np.where(out, -v, v); p = np.minimum(np.maximum(p, lo), hi)
            pos[i] = p; vel[i] = v; cost[i] = f(p)[0]
            if cost[i] < pb_c[i]:
                pb_c[i] = cost[i]; pb_x[i] = p
                if cost[i] < best_s: best_s, best_x = cost[i], p.copy()
        w *= wdamp
    return dict(best=float(best_s), n_evals=f.n)


def run_mine(name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0, variant="A"):
    """Earlier in-house version (synchronous, vectorised, warm-started k-means, velocity kept).
    A = settings used in all earlier benchmark tables (c1=1.5, c2=2.0, velocity clamp 0.1*range).
    B = same code with the supplied source's settings (c1=0.5, c2=1.5, clamp = full range)."""
    import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from b1_pkg.optimizers_variants import run
    rng = np.random.default_rng(seed + 777); f = Counter(name, sigma, rng)
    kw = dict(c1=1.5, c2=2.0, vel_scale=0.1) if variant == "A" else dict(c1=0.5, c2=1.5, vel_scale=1.0)
    r = run("psombest", f, d, (lb, ub), N=N, k=5, MaxIt=MaxIt, seed=seed, **kw)
    return dict(best=r["best_cost"], n_evals=f.n)


# ---------------------------------------------------------------- the other paper algorithms (Master file ports)
def _clip_mirror(p, v, lo, hi, mirror):
    out = (p < lo) | (p > hi)
    if mirror: v = np.where(out, -v, v)
    return np.minimum(np.maximum(p, lo), hi), v


def _pso_loop(f, rng, pos, cost, lo, hi, span, MaxIt, w0=1.0, wdamp=0.99, c1=1.5, c2=2.0, vf=0.1, best=None):
    """The classical-PSO loop used by PSOEA after its evolutionary start (velocity stored, mirror boundary)."""
    N, d = pos.shape; vel = np.zeros((N, d)); pb_x, pb_c = pos.copy(), cost.copy()
    best_s = np.inf if best is None else best[0]; best_x = np.zeros(d) if best is None else best[1].copy()
    vmax = vf * span; vmin = -vmax; w = w0
    for it in range(MaxIt):
        for i in range(N):
            v = w * vel[i] + c1 * rng.random(d) * (pb_x[i] - pos[i]) + c2 * rng.random(d) * (best_x - pos[i])
            v = np.minimum(np.maximum(v, vmin), vmax); p, v = _clip_mirror(pos[i] + v, v, lo, hi, True)
            pos[i] = p; vel[i] = v; c = f(p)[0]; cost[i] = c
            if c < pb_c[i]:
                pb_c[i] = c; pb_x[i] = p
                if c < best_s: best_s, best_x = c, p.copy()
        w *= wdamp
    return best_s


def run_pso_awd(name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0):
    """run_pso_awd from the Master file (adaptive weighted delay velocity, Xu et al. 2021 settings)."""
    rng = np.random.default_rng(seed); f = Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    pos = lo + rng.random((N, d)) * span; vel = np.zeros((N, d)); cost = np.empty(N)
    best_s, best_x = np.inf, np.zeros(d)
    for i in range(N):
        cost[i] = f(pos[i])[0]
        if cost[i] < best_s: best_s, best_x = cost[i], pos[i].copy()
    init_worst = max(0.0, cost.max()); pb_x, pb_c = pos.copy(), cost.copy()
    vmax = 0.1 * span; vmin = -vmax; w = 1.0; c1, c2 = 1.5, 2.0
    for it in range(1, MaxIt + 1):
        if init_worst == 0: es = np.inf if best_s < 0 else 0.0
        else: es = (init_worst - best_s) / init_worst
        prev = np.zeros(d)
        for i in range(N):
            delay = np.zeros(d) if i == 0 else (1 - w) * prev
            v = w * vel[i] + delay + c1 * rng.random(d) * (pb_x[i] - pos[i]) + c2 * rng.random(d) * (best_x - pos[i])
            v = np.minimum(np.maximum(v, vmin), vmax); p, v = _clip_mirror(pos[i] + v, v, lo, hi, True)
            pos[i] = p; vel[i] = v; prev = v; c = f(p)[0]; cost[i] = c
            if c < pb_c[i]:
                pb_c[i] = c; pb_x[i] = p
                if c < best_s: best_s, best_x = c, p.copy()
        ratio = (MaxIt - it) / MaxIt
        c1 = (2.5 - 0.5) * ratio + 0.5; c2 = (0.5 - 2.5) * ratio + 2.5
        with np.errstate(over="ignore"): w = 1 - 0.9 / (1 + np.exp(0.5 * es))
    return dict(best=float(best_s), n_evals=f.n)


def _rank_prob(costs, pressure):
    n = len(costs); order = np.argsort(costs, kind="stable"); ranks = np.empty(n); ranks[order] = np.arange(n)
    wts = np.exp(-pressure * ranks / max(1, n - 1)); return wts / wts.sum()


def _roulette(rng, cum):
    k = int(np.searchsorted(cum, rng.random(), side="left")); return min(k, len(cum) - 1)


def run_psoea(name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0):
    """run_psoea from the Master file: rank-selected crossover start, then classical PSO."""
    rng = np.random.default_rng(seed); f = Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    orig = lo + rng.random((N, d)) * span; oc = np.empty(N); best_s, best_x = np.inf, np.zeros(d)
    for i in range(N):
        oc[i] = f(orig[i])[0]
        if oc[i] < best_s: best_s, best_x = oc[i], orig[i].copy()
    cum = np.cumsum(_rank_prob(oc, 1.0)); pos = np.empty((N, d)); cost = np.empty(N)
    for i in range(N):
        a = _roulette(rng, cum); b = _roulette(rng, cum)
        while b == a and N > 1: b = _roulette(rng, cum)
        mask = rng.random(d) < 0.5; child = orig[a].copy(); child[~mask] = orig[b][~mask]
        pos[i] = child; cost[i] = f(child)[0]
        if cost[i] < best_s: best_s, best_x = cost[i], child.copy()
    best = _pso_loop(f, rng, pos, cost, lo, hi, span, MaxIt, best=(best_s, best_x))
    return dict(best=float(best), n_evals=f.n)


def run_ga(name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0):
    """run_real_coded_ga from the Master file (rank selection, arithmetic crossover, cooled Gaussian mutation, 5% elites)."""
    rng = np.random.default_rng(seed); f = Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    pos = lo + rng.random((N, d)) * span; cost = np.array([f(pos[i])[0] for i in range(N)])
    best_s = cost.min(); elite_n = min(max(2, int(np.floor(0.05 * N + 0.5))), N - 1)
    for it in range(1, MaxIt + 1):
        order = np.argsort(cost, kind="stable"); elites = pos[order[:elite_n]]; ec = cost[order[:elite_n]]
        cum = np.cumsum(_rank_prob(cost, 1.0)); nc = N - elite_n
        kids = np.empty((nc, d)); kc = np.empty(nc)
        cool = 1 - 0.9 * (it - 1) / max(1, MaxIt - 1)
        for j in range(nc):
            a = _roulette(rng, cum); b = _roulette(rng, cum)
            while b == a and N > 1: b = _roulette(rng, cum)
            if rng.random() <= 1.0:
                al = rng.random(d); child = al * pos[a] + (1 - al) * pos[b]
            else: child = pos[a].copy()
            m = rng.random(d) < 0.02
            if m.any(): child[m] += 0.10 * cool * span[m] * rng.standard_normal(int(m.sum()))
            child = np.minimum(np.maximum(child, lo), hi); kids[j] = child; kc[j] = f(child)[0]
        pos = np.vstack([elites, kids]); cost = np.concatenate([ec, kc]); best_s = min(best_s, cost.min())
    return dict(best=float(best_s), n_evals=f.n)


def run_b1(name, lb, ub, d, kind, N=150, MaxIt=1000, sigma=0.01, seed=0):
    """Arch's B1 variants with c3 fading 1 -> 0, velocity kept, c1=1.5, c2=2.0, clamp 0.1*range (the settings used in all B1 benchmark runs)."""
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from b1_pkg.optimizers_variants import run
    rng = np.random.default_rng(seed + 777); f = Counter(name, sigma, rng)
    r = run({"b1_fade": "b1", "b1_coord_fade": "b1_coord", "b1_proj_fade": "b1_proj"}[kind], f, d, (lb, ub),
            N=N, k=5, MaxIt=MaxIt, seed=seed, c1=1.5, c2=2.0, vel_scale=0.1, store_vel=True, c3_fade=True)
    return dict(best=r["best_cost"], n_evals=f.n)

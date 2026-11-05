"""B1-proj-fade with four switches borrowed from the supplied PSOMbest code, one at a time.
Everything else is identical to ports.run_b1_masterloop(kind='b1_proj_fade'): groups made once along the first principal
direction, 5 equal-count cuts, leader = best personal best of the group, c3 fades 1 -> 0, particle-by-particle update,
immediate global-best update, mirror boundary, w starts at 1 and is multiplied by 0.99 per iteration.

switches (config name -> what changes):
  base : c1,c2 = 1.5,2.0 | speed limit 10% of range | speed kept     | uniform random start
  c12  : c1,c2 = 0.5,1.5 (weaker pull to own best and global best)
  vmax : speed limit = 100% of the range
  novel: speed discarded after every move (no inertia)
  lhs  : Latin-hypercube start (same routine as the PSOMbest port)
  all4 : all four switches on
"""
import numpy as np, ports

CONFIGS = {
    "base":  dict(c12=False, vmax=False, novel=False, lhs=False),
    "c12":   dict(c12=True,  vmax=False, novel=False, lhs=False),
    "vmax":  dict(c12=False, vmax=True,  novel=False, lhs=False),
    "novel": dict(c12=False, vmax=False, novel=True,  lhs=False),
    "lhs":   dict(c12=False, vmax=False, novel=False, lhs=True),
    "all4":  dict(c12=True,  vmax=True,  novel=True,  lhs=True),
}


def run_proj(cfg, name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0, k=5, w0=1.0, wdamp=0.99):
    S = CONFIGS[cfg]
    c1, c2 = (0.5, 1.5) if S["c12"] else (1.5, 2.0)
    vf = 1.0 if S["vmax"] else 0.1
    rng = np.random.default_rng(seed); f = ports.Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    if S["lhs"]: pos = lo + ports.local_lhs(N, d, 5, rng) * span
    else:        pos = lo + rng.random((N, d)) * span
    vel = np.zeros((N, d)); cost = np.empty(N)
    best_s, best_x = np.inf, np.zeros(d)
    for i in range(N):
        cost[i] = f(pos[i])[0]
        if cost[i] < best_s: best_s, best_x = cost[i], pos[i].copy()
    pb_x, pb_c = pos.copy(), cost.copy(); vmax = vf * span; vmin = -vmax; w = w0
    ke = min(k, N); qs = np.linspace(0, 1, ke + 1)[1:-1]
    mu = pos.mean(0); u = np.linalg.svd(pos - mu, full_matrices=False)[2][0]
    pcuts = np.quantile((pos - mu) @ u, qs)
    for it in range(MaxIt):
        c3 = 1.0 - it / MaxIt
        lab = np.searchsorted(pcuts, (pos - mu) @ u)
        tab = np.empty((ke, d))
        for g in range(ke):
            m = lab == g
            tab[g] = pb_x[np.where(m)[0][np.argmin(pb_c[m])]] if m.any() else best_x
        for i in range(N):
            lead = tab[lab[i]]
            v = (w * vel[i] + c1 * rng.random(d) * (pb_x[i] - pos[i]) + c2 * rng.random(d) * (best_x - pos[i])
                 + c3 * rng.random(d) * (lead - pos[i]))
            v = np.minimum(np.maximum(v, vmin), vmax); p = pos[i] + v
            out = (p < lo) | (p > hi); v = np.where(out, -v, v); p = np.minimum(np.maximum(p, lo), hi)
            pos[i] = p; vel[i] = 0.0 if S["novel"] else v; cost[i] = f(p)[0]
            if cost[i] < pb_c[i]:
                pb_c[i] = cost[i]; pb_x[i] = p
                if cost[i] < best_s: best_s, best_x = cost[i], p.copy()
        w *= wdamp
    return dict(best=float(best_s), n_evals=f.n)

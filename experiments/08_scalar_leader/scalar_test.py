"""B1-fade (3 grouping rules) with the PSOMbest-style ONE-NUMBER leader added.
Everything is identical to ports.run_b1_masterloop (the Master-loop B1 runs already done): groups made once and frozen,
c1=1.5, c2=2.0, speed limit 10% of range, speed kept, uniform start, mirror boundary, w starts at 1 and x0.99,
c3 fades 1 -> 0, leader table rebuilt from the personal bests every iteration.  The ONLY change is what the group pull
points at.  kind = b1_fade (k-means centres) | b1_coord_fade (per-coordinate cuts) | b1_proj_fade (cuts along first principal direction)

scalar rule (what the pull points at; the number is copied to EVERY coordinate):
  none     : the normal B1 vector leader (must reproduce the earlier Master-loop runs exactly: self-test checks this)
  lead0    : the first coordinate of the normal leader vector
  leadmean : the mean of all coordinates of the normal leader vector
  first    : the supplied PSOMbest rule: the FIRST member of the group (lowest particle number), its FIRST coordinate,
             using current positions.  (coord kind: group of coordinate 0.)  Empty group -> first coordinate of global best.
"""
import numpy as np, ports

KINDS = ["b1_fade", "b1_coord_fade", "b1_proj_fade"]
SCALARS = ["lead0", "leadmean", "first"]


def run_scalar(kind, scalar, name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0, k=5, w0=1.0, wdamp=0.99,
               c1=1.5, c2=2.0, vf=0.1):
    rng = np.random.default_rng(seed); f = ports.Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    pos = lo + rng.random((N, d)) * span; vel = np.zeros((N, d)); cost = np.empty(N)
    best_s, best_x = np.inf, np.zeros(d)
    for i in range(N):
        cost[i] = f(pos[i])[0]
        if cost[i] < best_s: best_s, best_x = cost[i], pos[i].copy()
    pb_x, pb_c = pos.copy(), cost.copy(); vmax = vf * span; vmin = -vmax; w = w0
    ke = min(k, N); qs = np.linspace(0, 1, ke + 1)[1:-1]; ar = np.arange(d)
    if kind == "b1_fade":
        cen = pos[rng.choice(N, size=ke, replace=False)].copy()
        for _ in range(20):
            lab = np.argmin(((pos[:, None, :] - cen[None]) ** 2).sum(2), axis=1); new = cen.copy()
            for g in range(ke):
                m = lab == g
                if m.any(): new[g] = pos[m].mean(0)
            if np.allclose(new, cen): break
            cen = new
    elif kind == "b1_coord_fade":
        cuts = np.quantile(pos, qs, axis=0)
    elif kind == "b1_proj_fade":
        mu = pos.mean(0); u = np.linalg.svd(pos - mu, full_matrices=False)[2][0]
        pcuts = np.quantile((pos - mu) @ u, qs)
    else: raise ValueError(kind)
    for it in range(MaxIt):
        c3 = 1.0 - it / MaxIt
        if kind == "b1_fade":
            lab = np.argmin(((pos[:, None, :] - cen[None]) ** 2).sum(2), axis=1)
        elif kind == "b1_proj_fade":
            lab = np.searchsorted(pcuts, (pos - mu) @ u)
        else:
            lab = (pos[:, None, :] > cuts[None]).sum(1)          # (N, d)
        if kind == "b1_coord_fade":
            tab = np.empty((ke, d))
            for g in range(ke):
                mask = lab == g
                idx = np.argmin(np.where(mask, pb_c[:, None], np.inf), axis=0)
                tab[g] = np.where(mask.any(0), pb_x[idx, ar], best_x)
        else:
            tab = np.empty((ke, d))
            for g in range(ke):
                m = lab == g
                tab[g] = pb_x[np.where(m)[0][np.argmin(pb_c[m])]] if m.any() else best_x
        if scalar == "first":
            lab0 = lab[:, 0] if kind == "b1_coord_fade" else lab
            fv = np.empty(ke)
            for g in range(ke):
                mem = np.nonzero(lab0 == g)[0]
                fv[g] = pos[mem[0], 0] if len(mem) else best_x[0]
        for i in range(N):
            lead = tab[lab[i], ar] if kind == "b1_coord_fade" else tab[lab[i]]
            if scalar == "lead0":      lead = np.full(d, lead[0])
            elif scalar == "leadmean": lead = np.full(d, lead.mean())
            elif scalar == "first":    lead = np.full(d, fv[lab[i, 0] if kind == "b1_coord_fade" else lab[i]])
            elif scalar not in (None, "none"): raise ValueError(scalar)
            v = (w * vel[i] + c1 * rng.random(d) * (pb_x[i] - pos[i]) + c2 * rng.random(d) * (best_x - pos[i])
                 + c3 * rng.random(d) * (lead - pos[i]))
            v = np.minimum(np.maximum(v, vmin), vmax); p = pos[i] + v
            out = (p < lo) | (p > hi); v = np.where(out, -v, v); p = np.minimum(np.maximum(p, lo), hi)
            pos[i] = p; vel[i] = v; cost[i] = f(p)[0]
            if cost[i] < pb_c[i]:
                pb_c[i] = cost[i]; pb_x[i] = p
                if cost[i] < best_s: best_s, best_x = cost[i], p.copy()
        w *= wdamp
    return dict(best=float(best_s), n_evals=f.n)

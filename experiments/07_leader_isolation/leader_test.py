"""PSOMbest (supplied-code profile 'master_repro') with ONE thing changed: how the group leader is built.
Everything else is identical to ports.run_family('master_repro'): k-means (k=5) on the current positions EVERY iteration,
c1,c2,c3 = 0.5,1.5,1.0 (c3 constant), speed limit = whole range, speed discarded after each move, Latin-hypercube start,
particle-by-particle update, position clipped at the edges, w starts at 1 and is multiplied by 0.99.

leader rule (config name):
  scalar_first : the supplied code. First cluster member's FIRST coordinate, copied to every coordinate.
  scalar_mean  : the cluster's mean over all members and all coordinates (one number), copied to every coordinate.
  scalar_best1 : the best member's (lowest current cost) FIRST coordinate, copied to every coordinate.
  stored_best  : a real vector - the cluster member with the lowest current cost.
  pbest_vec    : a real vector - the best personal best among the cluster members.
  none         : no group pull at all (c3 = 0): PSOMbest's settings with plain PSO behaviour.
Scalar rules push every coordinate towards ONE common value; vector rules cannot do that.
"""
import numpy as np, ports

LEADERS = ["scalar_first", "scalar_mean", "scalar_best1", "stored_best", "pbest_vec", "none"]


def run_leader(rule, name, lb, ub, d, N=150, MaxIt=1000, sigma=0.01, seed=0, k=5, w0=1.0, wdamp=0.99):
    c1, c2, c3 = 0.5, 1.5, (0.0 if rule == "none" else 1.0)
    rng = np.random.default_rng(seed); f = ports.Counter(name, sigma, rng)
    lo = np.full(d, float(lb)); hi = np.full(d, float(ub)); span = hi - lo
    pos = lo + ports.local_lhs(N, d, 5, rng) * span
    vel = np.zeros((N, d)); cost = np.empty(N)
    best_s, best_x = np.inf, np.zeros(d)
    for i in range(N):
        cost[i] = f(pos[i])[0]
        if cost[i] < best_s: best_s, best_x = cost[i], pos[i].copy()
    pb_x, pb_c = pos.copy(), cost.copy()
    vmax = 1.0 * span; vmin = -vmax; w = w0
    leaders = np.zeros((k, d))
    for it in range(MaxIt):
        if rule != "none":
            lab, centers = ports.local_kmeans(pos, k, 100, 1e-10, rng)
            for c in range(k):
                mem = np.nonzero(lab == c)[0]
                if len(mem) == 0: leaders[c] = best_x
                elif rule == "scalar_first": leaders[c] = pos[mem[0], 0]
                elif rule == "scalar_mean":  leaders[c] = pos[mem].mean()
                elif rule == "scalar_best1": leaders[c] = pos[mem[np.argmin(cost[mem])], 0]
                elif rule == "stored_best":  leaders[c] = pos[mem[np.argmin(cost[mem])]]
                elif rule == "pbest_vec":    leaders[c] = pb_x[mem[np.argmin(pb_c[mem])]]
                else: raise ValueError(rule)
        for i in range(N):
            v = w * vel[i] + c1 * rng.random(d) * (pb_x[i] - pos[i]) + c2 * rng.random(d) * (best_x - pos[i])
            if rule != "none": v = v + c3 * rng.random(d) * (leaders[lab[i]] - pos[i])
            v = np.minimum(np.maximum(v, vmin), vmax)
            p = np.minimum(np.maximum(pos[i] + v, lo), hi)
            pos[i] = p; vel[i] = 0.0; cost[i] = f(p)[0]
            if cost[i] < pb_c[i]:
                pb_c[i] = cost[i]; pb_x[i] = p
                if cost[i] < best_s: best_s, best_x = cost[i], p.copy()
        w *= wdamp
    return dict(best=float(best_s), n_evals=f.n)

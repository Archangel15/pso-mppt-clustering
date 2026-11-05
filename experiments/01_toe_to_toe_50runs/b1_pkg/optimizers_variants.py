"""EXPERIMENTAL: B1 variants with one-dimensional grouping, plus the three
existing algorithms re-implemented in one runner that also records extra
metrics. Nothing here replaces B1_dynamic_gbest.

Algorithms (algo argument):
  std        Standard PSO
  psombest   PSOMbest (live k-means every iteration, warm start)
  b1         B1_dynamic_gbest (one-time k-means, frozen centroids)
  b1_coord   EXPERIMENT: for every coordinate j, k groups cut once on that
             single line (equal-count cuts of the initial swarm, frozen).
             Each coordinate has its own group leader (best personal best in
             the group); the cluster term pulls coordinate j toward the
             leader's j-th value.
  b1_proj    EXPERIMENT: one frozen line (first principal axis of the
             initial swarm) cut into k intervals; each particle's group is
             its interval; leader is the best personal best in the group.

The logic for std / psombest / b1 is copied from optimizers_fast.py,
including the order in which random numbers are drawn, so results for the
same seed match the earlier runs (checked in verify_variants.py).

Extra metrics returned: history (best-so-far every `hist_every`
iterations), last_improve_iter, final_diversity (mean per-coordinate std of
the swarm divided by the search range).
"""
import numpy as np
from b1_pkg.optimizers_fast import normalize_bounds, _lhs_init, _kmeans_nd

VEL_MAX_SCALE = 0.1
ALGOS = ["std", "psombest", "b1", "b1_coord", "b1_proj"]


def _mbest_vector(labels, k_eff, pbest_pos, pbest_cost, gbest_pos, dim):
    mbest = np.empty((k_eff, dim))
    for c in range(k_eff):
        mask = labels == c
        if np.any(mask):
            idx = np.where(mask)[0]
            mbest[c] = pbest_pos[idx[np.argmin(pbest_cost[idx])]]
        else:
            mbest[c] = gbest_pos
    return mbest


def run(algo, cost_fn, dim, bounds, N=150, k=5, MaxIt=1000, seed=None,
        w=1.0, wdamp=0.99, c1=1.5, c2=2.0, c3=1.0,
        vel_scale=VEL_MAX_SCALE, hist_every=10, c3_fade=False, w_linear=None, store_vel=True):
    rng = np.random.default_rng(seed)
    lo, hi = normalize_bounds(bounds, dim)
    rangev = hi - lo
    vel_max = vel_scale * rangev
    vel_min = -vel_max

    position = _lhs_init(N, dim, lo, hi, rng)
    velocity = np.zeros((N, dim))
    cost = cost_fn(position)
    n_evals = N

    pbest_pos = position.copy()
    pbest_cost = cost.copy()
    gi = int(np.argmin(pbest_cost))
    gbest_pos = pbest_pos[gi].copy()
    gbest_cost = pbest_cost[gi]

    k_eff = min(k, N)
    prev_centroids = None
    frozen_centroids = None
    cuts_coord = None
    proj_u = proj_mu = proj_cuts = None

    if algo == "b1":
        idx0 = rng.choice(N, size=k_eff, replace=False)
        frozen_centroids = position[idx0].copy()
        for _ in range(20):
            dist = np.linalg.norm(position[:, None, :] - frozen_centroids[None, :, :], axis=2)
            lab0 = np.argmin(dist, axis=1)
            newc = frozen_centroids.copy()
            for c in range(k_eff):
                m = lab0 == c
                if np.any(m):
                    newc[c] = position[m].mean(axis=0)
            if np.allclose(newc, frozen_centroids):
                break
            frozen_centroids = newc
    elif algo in ("b1_coord", "b1_hybrid"):
        qs = np.linspace(0, 1, k_eff + 1)[1:-1]
        cuts_coord = np.quantile(position, qs, axis=0)          # (k-1, d)
    if algo in ("b1_proj", "b1_hybrid"):
        proj_mu = position.mean(axis=0)
        _, _, vt = np.linalg.svd(position - proj_mu, full_matrices=False)
        proj_u = vt[0]
        t0 = (position - proj_mu) @ proj_u
        proj_cuts = np.quantile(t0, np.linspace(0, 1, k_eff + 1)[1:-1])

    history = [gbest_cost]
    last_improve = 0
    wc = w
    ar = np.arange(dim)

    for it in range(MaxIt):
        if w_linear is not None:
            wc = w_linear[0] + (w_linear[1] - w_linear[0]) * it / MaxIt
        if algo == "std":
            social_c = 0.0
            mb_term = 0.0
        else:
            if algo == "psombest":
                labels, centroids = _kmeans_nd(position, k_eff, rng, init_centroids=prev_centroids)
                prev_centroids = centroids
                kk = len(centroids)
                mbest = np.empty((kk, dim))
                for c in range(kk):
                    m = labels == c
                    if np.any(m):
                        idx = np.where(m)[0]
                        mbest[c] = position[idx[np.argmin(cost[idx])]]
                    else:
                        mbest[c] = gbest_pos
                mb = mbest[labels]
            elif algo == "b1":
                dist = np.linalg.norm(position[:, None, :] - frozen_centroids[None, :, :], axis=2)
                labels = np.argmin(dist, axis=1)
                mbest = _mbest_vector(labels, k_eff, pbest_pos, pbest_cost, gbest_pos, dim)
                mb = mbest[labels]
            elif algo == "b1_proj":
                t = (position - proj_mu) @ proj_u
                labels = np.searchsorted(proj_cuts, t)
                mbest = _mbest_vector(labels, k_eff, pbest_pos, pbest_cost, gbest_pos, dim)
                mb = mbest[labels]
            elif algo == "b1_coord":
                labels = (position[:, None, :] > cuts_coord[None, :, :]).sum(axis=1)  # (N, d)
                mbest_tab = np.empty((k_eff, dim))
                for g in range(k_eff):
                    mask = labels == g
                    masked = np.where(mask, pbest_cost[:, None], np.inf)
                    idx = np.argmin(masked, axis=0)
                    vals = pbest_pos[idx, ar]
                    empty = ~np.any(mask, axis=0)
                    vals = np.where(empty, gbest_pos, vals)
                    mbest_tab[g] = vals
                mb = mbest_tab[labels, ar[None, :]]
            elif algo == "b1_hybrid":
                # EXPERIMENT: equal blend of the per-coordinate leader and the projection-group leader
                t = (position - proj_mu) @ proj_u
                lab_p = np.searchsorted(proj_cuts, t)
                mbest_p = _mbest_vector(lab_p, k_eff, pbest_pos, pbest_cost, gbest_pos, dim)
                mb_p = mbest_p[lab_p]
                lab_c = (position[:, None, :] > cuts_coord[None, :, :]).sum(axis=1)
                mbest_tab = np.empty((k_eff, dim))
                for g in range(k_eff):
                    mask = lab_c == g
                    masked = np.where(mask, pbest_cost[:, None], np.inf)
                    idx = np.argmin(masked, axis=0)
                    vals = np.where(~np.any(mask, axis=0), gbest_pos, pbest_pos[idx, ar])
                    mbest_tab[g] = vals
                mb = 0.5 * (mbest_tab[lab_c, ar[None, :]] + mb_p)
            else:
                raise ValueError(algo)

        r1 = rng.random((N, dim))
        r2 = rng.random((N, dim))
        if algo == "std":
            velocity = (wc * velocity
                        + c1 * r1 * (pbest_pos - position)
                        + c2 * r2 * (gbest_pos - position))
        else:
            r3 = rng.random((N, dim))
            c3_t = c3 * (1.0 - it / MaxIt) if c3_fade else c3
            velocity = (wc * velocity
                        + c1 * r1 * (pbest_pos - position)
                        + c2 * r2 * (gbest_pos - position)
                        + c3_t * r3 * (mb - position))
        velocity = np.clip(velocity, vel_min, vel_max)
        position = np.clip(position + velocity, lo, hi)
        if not store_vel:
            velocity = np.zeros_like(velocity)   # supplied MATLAB source never stores velocity

        cost = cost_fn(position)
        n_evals += N

        improved = cost < pbest_cost
        pbest_pos[improved] = position[improved]
        pbest_cost[improved] = cost[improved]
        ib = int(np.argmin(pbest_cost))
        if pbest_cost[ib] < gbest_cost:
            gbest_cost = pbest_cost[ib]
            gbest_pos = pbest_pos[ib].copy()
            last_improve = it + 1
        wc *= wdamp
        if (it + 1) % hist_every == 0:
            history.append(gbest_cost)

    diversity = float(np.mean(position.std(axis=0) / rangev))
    return dict(best_cost=float(gbest_cost), n_evals=n_evals,
                gbest_pos=gbest_pos.copy(),
                history=np.array(history, dtype=float),
                last_improve_iter=last_improve, final_diversity=diversity)

"""Vectorized versions of the three optimizers for the heavier paper-scale
run (N=150, MaxIt=1000, d up to 100). cost_fn is called ONCE per iteration
on the whole (N,d) population matrix and must return a length-N array.
Otherwise identical logic/math to benchmarks/optimizers.py.

VEL_MAX_SCALE: the paper's own reference code (psokmeans.m) sets
VelMax = 1 * (VarMax - VarMin), i.e. the full search-space range. A
diagnostic run (see report) showed this literal setting causes severe
overshoot/divergence in this implementation across every tested function
(unimodal and multimodal alike) -- e.g. mean error on 30-D Zakharov drops
from 52.5 at scale=1.0 to 9.4e-8 at scale=0.1. 0.1 is the standard PSO
literature convention (Eberhart & Shi recommend Vmax = 0.1-0.2 x range) and
is used here as the corrected, documented setting instead of the literal
psokmeans.m value. This is a deliberate, flagged deviation -- see the
reproducibility section of the report.
"""
from dataclasses import dataclass
import numpy as np

VEL_MAX_SCALE = 0.1


def normalize_bounds(bounds, dim):
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and np.isscalar(bounds[0]):
        lo, hi = bounds
        return np.full(dim, float(lo)), np.full(dim, float(hi))
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo, hi


def _lhs_init(N, dim, lo, hi, rng):
    cut = np.linspace(0, 1, N + 1)
    u = rng.uniform(size=(N, dim))
    a = cut[:N][:, None]
    b = cut[1:N + 1][:, None]
    pts = a + u * (b - a)
    for j in range(dim):
        rng.shuffle(pts[:, j])
    return lo + pts * (hi - lo)


@dataclass
class PSOParams:
    N: int = 150
    MaxIt: int = 1000
    w: float = 1.0
    wdamp: float = 0.99
    c1: float = 1.5
    c2: float = 2.0
    seed: int = None


def run_standard_pso(cost_fn, dim, bounds, params: PSOParams):
    rng = np.random.default_rng(params.seed)
    N = params.N
    lo, hi = normalize_bounds(bounds, dim)
    vel_max = VEL_MAX_SCALE * (hi - lo)
    vel_min = -vel_max

    position = _lhs_init(N, dim, lo, hi, rng)
    velocity = np.zeros((N, dim))
    cost = cost_fn(position)
    n_evals = N

    pbest_pos = position.copy()
    pbest_cost = cost.copy()
    gbest_idx = int(np.argmin(pbest_cost))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    w = params.w
    for it in range(params.MaxIt):
        r1 = rng.random((N, dim))
        r2 = rng.random((N, dim))
        velocity = (w * velocity
                    + params.c1 * r1 * (pbest_pos - position)
                    + params.c2 * r2 * (gbest_pos - position))
        velocity = np.clip(velocity, vel_min, vel_max)
        position = np.clip(position + velocity, lo, hi)

        cost = cost_fn(position)
        n_evals += N

        improved = cost < pbest_cost
        pbest_pos[improved] = position[improved]
        pbest_cost[improved] = cost[improved]

        it_best = int(np.argmin(pbest_cost))
        if pbest_cost[it_best] < gbest_cost:
            gbest_cost = pbest_cost[it_best]
            gbest_pos = pbest_pos[it_best].copy()

        w *= params.wdamp

    return dict(best_cost=float(gbest_cost), n_evals=n_evals)


@dataclass
class PSOMbestParams:
    N: int = 150
    k: int = 5
    MaxIt: int = 1000
    w: float = 1.0
    wdamp: float = 0.99
    c1: float = 1.5
    c2: float = 2.0
    c3: float = 1.0
    seed: int = None


def _kmeans_nd(X, k, rng, max_iter=20, init_centroids=None):
    n = X.shape[0]
    k_eff = min(k, n)
    if init_centroids is not None and len(init_centroids) == k_eff:
        centroids = init_centroids.copy()
    else:
        idx = rng.choice(n, size=k_eff, replace=False)
        centroids = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k_eff):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)
    return labels, centroids


def run_psombest(cost_fn, dim, bounds, params: PSOMbestParams):
    rng = np.random.default_rng(params.seed)
    N, k = params.N, params.k
    lo, hi = normalize_bounds(bounds, dim)
    vel_max = VEL_MAX_SCALE * (hi - lo)
    vel_min = -vel_max

    position = _lhs_init(N, dim, lo, hi, rng)
    velocity = np.zeros((N, dim))
    cost = cost_fn(position)
    n_evals = N

    pbest_pos = position.copy()
    pbest_cost = cost.copy()
    gbest_idx = int(np.argmin(pbest_cost))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    w = params.w
    prev_centroids = None

    for it in range(params.MaxIt):
        labels, centroids = _kmeans_nd(position, k, rng, init_centroids=prev_centroids)
        prev_centroids = centroids
        k_eff = len(centroids)

        mbest_pos = np.empty((k_eff, dim))
        for c in range(k_eff):
            mask = labels == c
            if np.any(mask):
                idx_in_cluster = np.where(mask)[0]
                best_local = idx_in_cluster[np.argmin(cost[idx_in_cluster])]
                mbest_pos[c] = position[best_local]
            else:
                mbest_pos[c] = gbest_pos

        r1 = rng.random((N, dim))
        r2 = rng.random((N, dim))
        r3 = rng.random((N, dim))
        velocity = (w * velocity
                    + params.c1 * r1 * (pbest_pos - position)
                    + params.c2 * r2 * (gbest_pos - position)
                    + params.c3 * r3 * (mbest_pos[labels] - position))
        velocity = np.clip(velocity, vel_min, vel_max)
        position = np.clip(position + velocity, lo, hi)

        cost = cost_fn(position)
        n_evals += N

        improved = cost < pbest_cost
        pbest_pos[improved] = position[improved]
        pbest_cost[improved] = cost[improved]

        it_best = int(np.argmin(pbest_cost))
        if pbest_cost[it_best] < gbest_cost:
            gbest_cost = pbest_cost[it_best]
            gbest_pos = pbest_pos[it_best].copy()

        w *= params.wdamp

    return dict(best_cost=float(gbest_cost), n_evals=n_evals)


@dataclass
class B1GbestParams:
    N: int = 150
    k: int = 5
    MaxIt: int = 1000
    w: float = 1.0
    wdamp: float = 0.99
    c1: float = 1.5
    c2: float = 2.0
    c3: float = 1.0
    seed: int = None


def run_b1_dynamic_gbest(cost_fn, dim, bounds, params: B1GbestParams):
    rng = np.random.default_rng(params.seed)
    N, k = params.N, params.k
    lo, hi = normalize_bounds(bounds, dim)
    vel_max = VEL_MAX_SCALE * (hi - lo)
    vel_min = -vel_max

    position = _lhs_init(N, dim, lo, hi, rng)
    velocity = np.zeros((N, dim))
    cost = cost_fn(position)
    n_evals = N

    pbest_pos = position.copy()
    pbest_cost = cost.copy()
    gbest_idx = int(np.argmin(pbest_cost))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    k_eff = min(k, N)
    idx0 = rng.choice(N, size=k_eff, replace=False)
    frozen_centroids = position[idx0].copy()
    for _ in range(20):
        dist = np.linalg.norm(position[:, None, :] - frozen_centroids[None, :, :], axis=2)
        labels0 = np.argmin(dist, axis=1)
        new_centroids = frozen_centroids.copy()
        for c in range(k_eff):
            mask = labels0 == c
            if np.any(mask):
                new_centroids[c] = position[mask].mean(axis=0)
        if np.allclose(new_centroids, frozen_centroids):
            break
        frozen_centroids = new_centroids

    w = params.w

    for it in range(params.MaxIt):
        dist = np.linalg.norm(position[:, None, :] - frozen_centroids[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)

        mbest_pos = np.empty((k_eff, dim))
        for c in range(k_eff):
            mask = labels == c
            if np.any(mask):
                idx_in_cluster = np.where(mask)[0]
                best_local = idx_in_cluster[np.argmin(pbest_cost[idx_in_cluster])]
                mbest_pos[c] = pbest_pos[best_local]
            else:
                mbest_pos[c] = gbest_pos

        r1 = rng.random((N, dim))
        r2 = rng.random((N, dim))
        r3 = rng.random((N, dim))
        velocity = (w * velocity
                    + params.c1 * r1 * (pbest_pos - position)
                    + params.c2 * r2 * (gbest_pos - position)
                    + params.c3 * r3 * (mbest_pos[labels] - position))
        velocity = np.clip(velocity, vel_min, vel_max)
        position = np.clip(position + velocity, lo, hi)

        cost = cost_fn(position)
        n_evals += N

        improved = cost < pbest_cost
        pbest_pos[improved] = position[improved]
        pbest_cost[improved] = cost[improved]

        it_best = int(np.argmin(pbest_cost))
        if pbest_cost[it_best] < gbest_cost:
            gbest_cost = pbest_cost[it_best]
            gbest_pos = pbest_pos[it_best].copy()

        w *= params.wdamp

    return dict(best_cost=float(gbest_cost), n_evals=n_evals)

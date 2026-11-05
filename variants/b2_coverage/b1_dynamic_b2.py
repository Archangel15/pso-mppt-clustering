"""
b1_dynamic_b2.py

B1-dynamic (b1_dynamic_membership.py) + Mechanism 2 (tabu-style
coverage tracking) -- everything about free movement, per-iteration
region relabelling, global LHS init, and dropped gbest is unchanged;
the only addition is coverage tracking per region.

WHAT B2 DOES HERE, AND WHY IT'S DIFFERENT FROM b1_confined_b2.py:
particles here were never confined -- there is no captive population
to freeze once a region is "done," so freezing (as in B1_confined)
doesn't apply. Instead: once a region's bins are all visited, that
region's Mbest term is simply DROPPED from the velocity update for
whichever particles currently happen to be in it -- there is no more
useful local signal to offer, so those particles fall back to
inertia + personal best alone (identical to how standard PSO's
particles already behave, minus a gbest term this variant never had).
Crucially, those particles are NOT frozen and NOT excluded from
fitness evaluation -- they keep moving freely and can wander into a
still-active region, where they immediately regain a real Mbest pull.

HONEST CONSEQUENCE: because nothing is frozen and nothing is skipped,
this does NOT reduce function evaluations the way b1_confined_b2.py
does -- n_evals here should come out identical to plain B1-dynamic.
Any effect this variant has will show up in ACCURACY (does dropping a
stale local pull change where the swarm ends up), not in compute cost
savings. That asymmetry with the confined variant is expected and
should be reported as such, not treated as a shortcoming to fix.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tabu_coverage import RegionCoverage


@dataclass
class B1DynB2Params:
    N: int = 150
    k: int = 5
    MaxIt: int = 1000
    w: float = 1.0
    wdamp: float = 0.99
    c1: float = 0.5
    c3: float = 1.0
    bin_width: float = 0.3
    seed: int | None = None


def build_regions(v_min: float, v_max: float, k: int):
    edges = np.linspace(v_min, v_max, k + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(k)]


def current_region_ids(position: np.ndarray, region_edges: np.ndarray, k: int) -> np.ndarray:
    ids = np.searchsorted(region_edges[1:-1], position, side="right")
    return np.clip(ids, 0, k - 1)


@dataclass
class B1DynB2Result:
    gmpp_voltage: float
    gmpp_power: float
    best_power_history: np.ndarray
    n_regions_exhausted: int
    exhaustion_iteration: list


def run_b1_dyn_b2(power_fn, v_min: float, v_max: float,
                   params: B1DynB2Params = B1DynB2Params()) -> B1DynB2Result:
    rng = np.random.default_rng(params.seed)
    region_bounds = build_regions(v_min, v_max, params.k)
    region_edges = np.linspace(v_min, v_max, params.k + 1)
    coverage = RegionCoverage(region_bounds, bin_width=params.bin_width)
    exhaustion_iteration = [None] * params.k

    edges = np.linspace(v_min, v_max, params.N + 1)
    offsets = rng.uniform(0.0, 1.0, size=params.N)
    position = edges[:-1] + offsets * (edges[1:] - edges[:-1])
    rng.shuffle(position)

    velocity = np.zeros(params.N)
    cost = np.asarray(power_fn(position), dtype=float)
    pbest_position = position.copy()
    pbest_cost = cost.copy()

    vel_limit = 1.0 * (v_max - v_min)
    w = params.w
    best_power_history = np.zeros(params.MaxIt)

    # Seed coverage with initial positions.
    init_region_ids = current_region_ids(position, region_edges, params.k)
    for r in range(params.k):
        mask = init_region_ids == r
        coverage.record(r, position[mask])
        if coverage.exhausted[r]:
            exhaustion_iteration[r] = 0

    for it in range(params.MaxIt):
        region_ids = current_region_ids(position, region_edges, params.k)

        region_mbest_position = np.zeros(params.N)
        has_region_pull = np.zeros(params.N, dtype=bool)  # False where region is exhausted
        for r in range(params.k):
            mask = region_ids == r
            if not mask.any():
                continue
            if not coverage.exhausted[r]:
                best_idx_in_region = np.argmax(pbest_cost[mask])
                region_mbest_position[mask] = pbest_position[mask][best_idx_in_region]
                has_region_pull[mask] = True
            # else: leave has_region_pull False for these -- Mbest term
            # drops out entirely for particles currently in an exhausted region.

        r1 = rng.uniform(0, 1, size=params.N)
        r3 = rng.uniform(0, 1, size=params.N)

        region_term = np.where(has_region_pull, params.c3 * r3 * (region_mbest_position - position), 0.0)
        velocity = (
            w * velocity
            + params.c1 * r1 * (pbest_position - position)
            + region_term
        )
        velocity = np.clip(velocity, -vel_limit, vel_limit)

        position = np.clip(position + velocity, v_min, v_max)
        cost = np.asarray(power_fn(position), dtype=float)  # full population, every iteration -- no freezing here

        improved = cost > pbest_cost
        pbest_position[improved] = position[improved]
        pbest_cost[improved] = cost[improved]

        # Update coverage using CURRENT (post-move) positions/region membership.
        region_ids_after = current_region_ids(position, region_edges, params.k)
        for r in range(params.k):
            if coverage.exhausted[r]:
                continue
            mask = region_ids_after == r
            if mask.any():
                coverage.record(r, position[mask])
                if coverage.exhausted[r]:
                    exhaustion_iteration[r] = it + 1

        best_power_history[it] = pbest_cost.max()
        w *= params.wdamp

    overall_best_idx = int(np.argmax(pbest_cost))

    return B1DynB2Result(
        gmpp_voltage=float(pbest_position[overall_best_idx]),
        gmpp_power=float(pbest_cost[overall_best_idx]),
        best_power_history=best_power_history,
        n_regions_exhausted=int(coverage.exhausted.sum()),
        exhaustion_iteration=exhaustion_iteration,
    )

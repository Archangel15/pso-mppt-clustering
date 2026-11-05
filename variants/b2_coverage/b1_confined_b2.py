"""
b1_confined_b2.py

B1 (static confined clustering) + Mechanism 2 (tabu-style coverage
tracking), incrementally built on top of the verified B1_confined
implementation (b1_static_confined_pso.py) -- everything about
confinement, dropped gbest, and monotonic region-Mbest is unchanged;
the only addition is coverage-based early termination per region.

WHAT B2 DOES HERE, SPECIFICALLY: each region tracks which of its
0.3V-wide bins have been visited by its (confined) particles. Once a
region's bins are ALL visited, that region is marked exhausted, and
its particles are FROZEN -- they stop moving, and critically, they
stop being passed to the fitness function for the rest of the run.
This is where B2's actual compute-saving story lives: since particles
were already trapped in their region (B1_confined's whole premise),
"nothing left to discover here" genuinely means "stop spending
evaluations on this region" -- there's no cost to freezing a captive
population, unlike B1_dynamic (see b1_dynamic_b2.py for why that
variant's B2 does something different).

Function evaluations are only spent on the currently-active (non-
frozen) subset of particles each iteration -- this is what lets
eval_harness's n_evals metric actually show a reduction relative to
plain B1_confined, rather than just adding bookkeeping with no real
saving.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tabu_coverage import RegionCoverage


@dataclass
class B1B2Params:
    N: int = 150
    k: int = 5
    MaxIt: int = 1000
    w: float = 1.0
    wdamp: float = 0.99
    c1: float = 0.5
    c3: float = 1.0
    bin_width: float = 0.3
    min_particles_per_region: int = 5
    seed: int | None = None


def build_regions(v_min: float, v_max: float, k: int):
    edges = np.linspace(v_min, v_max, k + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(k)]


def assign_particles_to_regions(N: int, k: int):
    base = N // k
    extra = N % k
    region_ids = []
    for r in range(k):
        count = base + (1 if r < extra else 0)
        region_ids.extend([r] * count)
    return np.array(region_ids)


def lhs_1d(n_samples: int, lo: float, hi: float, rng: np.random.Generator):
    edges = np.linspace(lo, hi, n_samples + 1)
    offsets = rng.uniform(0.0, 1.0, size=n_samples)
    points = edges[:-1] + offsets * (edges[1:] - edges[:-1])
    rng.shuffle(points)
    return points


@dataclass
class B1B2Result:
    gmpp_voltage: float
    gmpp_power: float
    best_power_history: np.ndarray
    n_regions_exhausted: int
    exhaustion_iteration: list   # iteration each region froze, or None if never


def run_b1_b2(power_fn, v_min: float, v_max: float, params: B1B2Params = B1B2Params()) -> B1B2Result:
    if params.k > params.N // params.min_particles_per_region:
        raise ValueError(
            f"k={params.k} violates the particles-per-region floor "
            f"(N={params.N}, floor={params.min_particles_per_region})."
        )

    rng = np.random.default_rng(params.seed)
    regions = build_regions(v_min, v_max, params.k)
    region_ids = assign_particles_to_regions(params.N, params.k)
    coverage = RegionCoverage(regions, bin_width=params.bin_width)
    exhaustion_iteration = [None] * params.k

    position = np.zeros(params.N)
    for r, (lo, hi) in enumerate(regions):
        mask = region_ids == r
        position[mask] = lhs_1d(mask.sum(), lo, hi, rng)

    velocity = np.zeros(params.N)
    cost = np.asarray(power_fn(position), dtype=float)
    pbest_position = position.copy()
    pbest_cost = cost.copy()

    region_width = np.array([regions[r][1] - regions[r][0] for r in region_ids])
    vel_limit = 1.0 * region_width

    # Seed coverage with the initial positions.
    for r in range(params.k):
        mask = region_ids == r
        coverage.record(r, position[mask])
        if coverage.exhausted[r]:
            exhaustion_iteration[r] = 0

    w = params.w
    best_power_history = np.zeros(params.MaxIt)
    frozen = np.zeros(params.N, dtype=bool)  # per-particle, mirrors its region's exhaustion
    # Freeze any region that was ALREADY exhausted by the initial scatter --
    # without this, those particles would keep running for the whole
    # budget with no savings, silently defeating the point of B2 (this
    # was a real bug caught during verification, not a hypothetical).
    for r in range(params.k):
        if coverage.exhausted[r]:
            frozen[region_ids == r] = True

    for it in range(params.MaxIt):
        region_mbest_position = np.zeros(params.N)
        for r in range(params.k):
            mask = region_ids == r
            if not mask.any():
                continue
            best_idx_in_region = np.argmax(pbest_cost[mask])
            region_mbest_position[mask] = pbest_position[mask][best_idx_in_region]

        active = ~frozen
        if active.any():
            r1 = rng.uniform(0, 1, size=params.N)
            r3 = rng.uniform(0, 1, size=params.N)

            new_velocity = velocity.copy()
            new_velocity[active] = (
                w * velocity[active]
                + params.c1 * r1[active] * (pbest_position[active] - position[active])
                + params.c3 * r3[active] * (region_mbest_position[active] - position[active])
            )
            new_velocity[active] = np.clip(new_velocity[active], -vel_limit[active], vel_limit[active])
            velocity = new_velocity

            new_position = position.copy()
            new_position[active] = position[active] + velocity[active]
            region_lo = np.array([regions[r][0] for r in region_ids])
            region_hi = np.array([regions[r][1] for r in region_ids])
            new_position[active] = np.clip(new_position[active], region_lo[active], region_hi[active])
            position = new_position

            # Only evaluate the fitness function for ACTIVE particles --
            # this is the actual compute saving from freezing.
            new_cost = cost.copy()
            new_cost[active] = np.asarray(power_fn(position[active]), dtype=float)
            cost = new_cost

            improved = active & (cost > pbest_cost)
            pbest_position[improved] = position[improved]
            pbest_cost[improved] = cost[improved]

            # Update coverage and check for newly-exhausted regions.
            for r in range(params.k):
                if coverage.exhausted[r]:
                    continue
                mask = active & (region_ids == r)
                if mask.any():
                    coverage.record(r, position[mask])
                    if coverage.exhausted[r]:
                        exhaustion_iteration[r] = it + 1
                        frozen[region_ids == r] = True

        best_power_history[it] = pbest_cost.max()
        w *= params.wdamp

    overall_best_idx = int(np.argmax(pbest_cost))

    return B1B2Result(
        gmpp_voltage=float(pbest_position[overall_best_idx]),
        gmpp_power=float(pbest_cost[overall_best_idx]),
        best_power_history=best_power_history,
        n_regions_exhausted=int(coverage.exhausted.sum()),
        exhaustion_iteration=exhaustion_iteration,
    )

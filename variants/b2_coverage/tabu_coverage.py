"""
tabu_coverage.py

Shared coverage-tracking mechanism (Mechanism 2 / "B2") for the B1
family. A region's extent is discretized into bins of fixed width
(default 0.3V, derived earlier in this project from the narrowest
verified peak width -- 0.96V -- divided by 3, per the margin-of-safety
reasoning in the concept document). A bin is marked "visited" the
first time any particle's current position falls inside it. A region
is "exhausted" once every one of its bins has been visited at least
once.

Deliberately simplified relative to canonical Tabu Search: exhaustion
is COVERAGE-based (has every bin been sampled), not stagnation-based
(has quality stopped improving) -- this reduces the risk of declaring
a region "done" while it still contains an unexplored peak. There is
no aspiration criterion (revisiting a marked bin because it looks
promising), since "exhausted" here means fully covered, not quality-
plateaued -- a stated simplification, not an oversight.

Tracking itself is purely passive: it never restricts or redirects
particle movement (the "option 3" design settled on earlier). It only
ever records what has already happened and reports whether a region
is done. What EACH variant does in response to "this region is
exhausted" is left to that variant, because B1_confined and B1_dynamic
have genuinely different things they can do about it -- see the
module docstrings in b1_confined_b2.py and b1_dynamic_b2.py.
"""

from __future__ import annotations
import numpy as np


class RegionCoverage:
    def __init__(self, region_bounds: list, bin_width: float = 0.3):
        self.n_regions = len(region_bounds)
        self.bin_width = bin_width
        self.bin_edges = []       # one edge array per region
        self.visited = []         # one boolean array per region
        self.exhausted = np.zeros(self.n_regions, dtype=bool)

        for lo, hi in region_bounds:
            n_bins = max(1, int(np.ceil((hi - lo) / bin_width)))
            edges = np.linspace(lo, hi, n_bins + 1)
            self.bin_edges.append(edges)
            self.visited.append(np.zeros(n_bins, dtype=bool))

    def record(self, region_idx: int, positions: np.ndarray):
        """Mark bins visited by these positions within the given region.
        No-op if the region is already exhausted (nothing left to gain)."""
        if self.exhausted[region_idx] or len(positions) == 0:
            return
        edges = self.bin_edges[region_idx]
        idx = np.clip(np.searchsorted(edges[1:-1], positions, side="right"),
                       0, len(edges) - 2)
        self.visited[region_idx][idx] = True
        if self.visited[region_idx].all():
            self.exhausted[region_idx] = True

    def coverage_fraction(self, region_idx: int) -> float:
        return float(self.visited[region_idx].mean())

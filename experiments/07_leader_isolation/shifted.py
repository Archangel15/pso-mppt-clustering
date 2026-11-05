"""Shifted-optimum versions of the benchmarks: f_shift(x) = f(x - s), with a DIFFERENT offset s_j in every coordinate
(drawn once per case, up to +-25% of the search range).  The optimum value is unchanged, but the best point is no longer
the same number in every coordinate, so a leader that pushes all coordinates to one common value cannot help."""
import numpy as np, ports
_orig = ports.objective
OPTX = {"F2": 0.0, "F3": 1.0, "F9": 0.0, "F10": 1.0, "F11": 0.0, "F12": -2.903534, "F14": 0.0, "F16": 0.0,
        "F17": 1.0}            # (approximate) common coordinate value of each optimum; used only to keep the shifted optimum inside the bounds
SH = {}
for i, c in enumerate(sorted(ports.CASES)):
    lb, ub, d = ports.CASES[c]; m = 0.03 * (ub - lb); o = OPTX.get(c.split("_")[0], 0.0)
    SH[c] = np.clip(np.random.default_rng(1000 + i).uniform(-0.25, 0.25, d) * (ub - lb), lb + m - o, ub - m - o)
def shifted(x, name, sigma, rng):
    return _orig(np.atleast_2d(x) - SH[name], name, sigma, rng)
def install(): ports.objective = shifted

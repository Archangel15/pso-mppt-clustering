"""
scenarios.py

Preset and randomised partial-shading scenario generators built on top
of pv_model.PVStringCurve. Each function returns a built PVStringCurve
plus a short human-readable label describing the shading pattern.

Scenario categories, matching the evaluation plan in the concept
document (Section 8):
  - uniform            : sanity check, single peak
  - simple_partial      : a few well-separated, comparably-sized peaks
  - complex_partial      : many modules, several close/uneven peaks
  - adversarial_narrow_peak : the true GMPP is carried by a NARROW group
                              (few modules) but at HIGH irradiance, sitting
                              alongside a WIDE but LOWER-power group -- the
                              stress case for resolution/noise sensitivity
  - random_scenario     : parametrised random generator for bulk seeded runs
"""

from __future__ import annotations
import numpy as np
from pv_model import PVStringCurve, DEFAULT_PARAMS


def _make(irradiances, temperature_C=20.0, label=""):
    curve = PVStringCurve(irradiances=irradiances, temperature_C=temperature_C).build()
    return curve, label


def uniform(n_modules: int = 6, G: float = 1000.0, temperature_C: float = 20.0):
    """No shading: every module at the same irradiance. Expect 1 peak."""
    return _make([G] * n_modules, temperature_C, f"uniform G={G}")


def simple_partial(temperature_C: float = 20.0):
    """
    Two well-separated groups (matches the source paper's own Case 1:
    uniform-per-panel shading). Expect 2 clearly distinct peaks.
    """
    irradiances = [1000, 1000, 1000, 500, 500, 500]
    return _make(irradiances, temperature_C, "simple: 3@1000, 3@500")


def complex_partial(temperature_C: float = 20.0):
    """
    Several distinct groups of uneven size (matches the source paper's
    Case 2: non-uniform shading down to individual-module level).
    Expect 3-4 peaks of varying height, some fairly close together.
    """
    irradiances = [1000, 800, 400, 700, 1000, 200, 900, 550]
    return _make(irradiances, temperature_C, "complex: 8 modules, 6 distinct levels")


def adversarial_narrow_peak(temperature_C: float = 20.0):
    """
    Deliberately constructed stress case: the true GMPP sits on a
    NARROW peak (1 module at high irradiance, active alone at the top
    of the current sweep after the other groups have bypassed out),
    while a WIDE but lower-power peak exists earlier in the sweep
    (4 modules at low irradiance, all active together over a much
    wider voltage span). Verified numerically (see pv_testbed/verify.py):
    2 distinct peaks, with the global one roughly 5x narrower (~1.0 V)
    than the local one (~4.9 V) at the 90%-of-peak-height width measure.
    This directly tests whether a coarse sweep resolution can miss a
    narrow-but-global peak, per the resolution-sensitivity discussion
    in the concept document (Section 5.3).
    """
    irradiances = [300, 300, 300, 300, 600, 600, 600, 1000]
    return _make(irradiances, temperature_C,
                 "adversarial: 4@300 (wide, local) + 3@600 + 1@1000 (narrow, global)")


def random_scenario(n_modules: int, n_groups: int, seed: int,
                     G_min: float = 200.0, G_max: float = 1000.0,
                     temperature_C: float = 20.0):
    """
    Randomised scenario for bulk multi-seed testing. Assigns each of
    n_modules to one of n_groups irradiance levels (levels themselves
    drawn randomly in [G_min, G_max]), independent of any assumption
    about how many peaks will result -- ground_truth() establishes
    that from the actual curve, not from n_groups.
    """
    rng = np.random.default_rng(seed)
    group_levels = rng.uniform(G_min, G_max, size=n_groups)
    assignment = rng.integers(0, n_groups, size=n_modules)
    irradiances = [float(group_levels[a]) for a in assignment]
    label = f"random(seed={seed}, n_modules={n_modules}, n_groups={n_groups})"
    return _make(irradiances, temperature_C, label)


PRESETS = {
    "uniform": uniform,
    "simple_partial": simple_partial,
    "complex_partial": complex_partial,
    "adversarial_narrow_peak": adversarial_narrow_peak,
}

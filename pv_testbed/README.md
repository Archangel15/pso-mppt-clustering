# Synthetic PV String Testbed

Generates P-V curves for a series-connected string of PV modules under
partial shading, for validating MPPT algorithms (B1/B2/B3, PSOMbest,
P&O, INC, etc.) against known ground truth.

## Files
- `pv_model.py` — Bishop diode model + series-string bypass-diode logic
  (`PVStringCurve`). Parameters and equations sourced from Mao et al.
  2018 (TSPSOEM paper), with one flagged, documented deviation (see
  the module docstring: the source Appendix's `k_i` value was almost
  certainly a transcription error).
- `scenarios.py` — preset shading patterns (`uniform`, `simple_partial`,
  `complex_partial`, `adversarial_narrow_peak`) plus `random_scenario`
  for seeded bulk testing.
- `verify.py` — diagnostic script confirming the model behaves
  physically (run this first if you modify pv_model.py).

## Quick use

```python
from scenarios import adversarial_narrow_peak
import numpy as np

curve, label = adversarial_narrow_peak()
gt = curve.ground_truth()            # {"peak_count", "peak_voltages",
                                      #  "peak_powers", "gmpp_voltage", "gmpp_power"}

V = np.array([2.0, 3.5, 4.45])
P = curve.power_at_voltage(V)        # query power at candidate voltage(s) —
                                      # this is the fitness function your
                                      # PSOMbest/B1/B2/B3 particles should call

# With measurement noise, for the noise-vs-resolution tests:
rng = np.random.default_rng(0)
P_noisy = curve.power_at_voltage_noisy(V, noise_std_W=0.05, rng=rng)
```

## What's verified vs. what's still open

**Verified (see verify.py output / verify_scenarios.png):**
- `uniform` → exactly 1 peak.
- `simple_partial`, `complex_partial` → multiple distinct, correctly
  located peaks; curve shape matches Figure 1 of the source paper.
- `adversarial_narrow_peak` → 2 peaks, GMPP correctly on the narrower
  one (~0.96 V wide) vs. the local peak (~4.9 V wide) — a ~5x contrast.

**Still open / not yet built (see the roadmap in the concept document):**
- This models one string only. Multi-string / DMPPT (separate
  algorithm instances per panel, as TSPSOEM itself uses) is not
  implemented — add if your evaluation plan needs it.
- Temperature is a free parameter but no temperature-driven Isc drift
  is modelled (`k_i = 0`, documented simplification).
- The `prominence_W` ground-truth threshold is itself a tunable design
  parameter, not a fixed constant — see the note in
  `PVStringCurve.ground_truth()`. The adversarial scenario's local
  peak is deliberately low-prominence (~0.45 W) so this sensitivity is
  testable, not hidden.
- No physical minimum-peak-width bound has been derived yet from a
  specific reference system (Section 5.3 of the concept document) —
  scenario irradiance/module counts here were tuned empirically to
  produce the target stress case, not derived from that bound. Once
  the bound is derived, revisit whether `adversarial_narrow_peak`'s
  ~0.96 V width is representative of a realistic worst case or needs
  adjusting.

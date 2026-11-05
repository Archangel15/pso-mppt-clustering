"""
pv_model.py

Synthetic multi-peak P-V curve generator for a series-connected string of
PV modules under partial shading, built on the Bishop diode model and
parameter set given in:

    Mao, Zhang, Duan, Duan, Yang (2018), "Grid-Connected Modular
    PV-Converter System with Shuffled Frog Leaping Algorithm Based
    DMPPT Controller", Energy 143, pp. 181-190. (Eqs. 16-17, Appendix)

Each module's forward-bias current-voltage relationship follows the
Bishop model. Modules are chained in series; a module that cannot
support the string's forced current in forward bias is treated as
bypass-diode-protected (a standard, widely used idealisation in the
partial-shading PV literature): instead of following the model's own
avalanche-breakdown branch, its terminal voltage is clamped to a small
fixed negative bypass voltage, and it stops contributing current
limitation to the rest of the string.

KNOWN DATA CAVEAT (flagged deliberately, not silently patched):
The source paper's Appendix lists the "temperature coefficient of the
short-circuit current (k_i)" as 1.380658e-23 A. That numeric value is,
to high precision, Boltzmann's constant in J/K -- not a plausible
short-circuit current temperature coefficient (which is typically of
order 1e-3 A/K for silicon cells). This looks like a transcription
error in the source paper's appendix (k_i and Boltzmann's constant k
were likely swapped or conflated). This module therefore:
  - uses the standard value of Boltzmann's constant for the k that
    appears in the diode exponential term (q*Vj / (A*k*Tc)), and
  - sets k_i (the short-circuit current temperature coefficient) to 0,
    i.e. temperature-driven drift of Isc is not modelled, only the
    direct irradiance dependence (Isc = Iscr * G) is.
This is a deliberate, documented simplification, not a silent guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Bishop model parameters (Mao et al. 2018, Appendix), with the k_i caveat
# documented above.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BishopParams:
    A: float = 1.72                 # diode ideality factor
    q: float = 1.609e-19            # electron charge, C (as given in source paper)
    k_boltzmann: float = 1.380649e-23   # standard Boltzmann constant, J/K
    Rs: float = 5e-5                # series resistance, ohm
    Rp: float = 5e5                 # parallel (shunt) resistance, ohm
    ki: float = 0.0                 # short-circuit current temp. coefficient, A/K
                                     # (see module docstring: source value was implausible;
                                     # set to 0 as a documented simplification)
    Tr: float = 301.18              # reference temperature, K
    Iscr: float = 3.3               # rated short-circuit current at reference, A
    Io: float = 19.9693e-6          # reverse saturation current, A
    Vbr: float = -4.0               # junction breakdown voltage, V (unused: see bypass model)
    a_shunt: float = 0.1            # fraction of ohmic (shunt) current
    m_avalanche: float = 3.7        # avalanche breakdown exponent (unused: see bypass model)

    # --- Rated single-module electrical characteristics (Table 1) ---
    Pmpp_rated: float = 15.0        # W, at 1000 W/m^2
    Voc_rated: float = 6.6          # V
    Isc_rated: float = 3.28         # A


DEFAULT_PARAMS = BishopParams()


def cell_temperature_K(T_ambient_C: float, G: float) -> float:
    """T_c = T + 273 + 0.2*G, per source paper Appendix note."""
    return T_ambient_C + 273.0 + 0.2 * G


def module_Isc(G: float, T_ambient_C: float, p: BishopParams = DEFAULT_PARAMS) -> float:
    """Short-circuit current of a single module at irradiance G (W/m^2)."""
    Tc = cell_temperature_K(T_ambient_C, G)
    return (p.Iscr + p.ki * (Tc - p.Tr)) * (G / 1000.0)


def module_forward_current(Vj: float, G: float, T_ambient_C: float,
                            p: BishopParams = DEFAULT_PARAMS) -> float:
    """
    Bishop model current at junction voltage Vj, in the forward-bias
    region (Vj well away from Vbr, so the avalanche term is negligible
    and is dropped -- valid here because bypassed modules never reach
    that region under the idealised-bypass-diode treatment used below).
    """
    Tc = cell_temperature_K(T_ambient_C, G)
    Isc = module_Isc(G, T_ambient_C, p)
    diode_term = p.Io * (np.exp(p.q * Vj / (p.A * p.k_boltzmann * Tc)) - 1.0)
    shunt_term = (Vj / p.Rp) * (1.0 + p.a_shunt)  # avalanche factor ~1 far from Vbr
    return Isc - diode_term - shunt_term


def module_forward_voltage(I: float, G: float, T_ambient_C: float,
                            p: BishopParams = DEFAULT_PARAMS,
                            v_search_hi: float = 10.0) -> float:
    """
    Invert the Bishop model to find the forward-bias junction voltage
    Vj that produces current I, for a module at irradiance G. Solved
    numerically (Brent's method) since the shunt term is retained.
    Returns terminal voltage V = Vj + Rs*I.
    """
    def residual(Vj):
        return module_forward_current(Vj, G, T_ambient_C, p) - I

    lo, hi = -0.5, v_search_hi
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0:
        # I exceeds what this module can supply anywhere in-range;
        # caller should have already routed this through the bypass path.
        return 0.0
    Vj = brentq(residual, lo, hi, xtol=1e-10)
    return Vj + p.Rs * I


# ---------------------------------------------------------------------------
# String-level model: series modules with idealised bypass-diode protection
# ---------------------------------------------------------------------------

BYPASS_VOLTAGE = -0.6  # V; standard forward-conduction drop of a protective bypass diode


@dataclass
class PVStringCurve:
    """
    A simulated P-V curve for a string of series-connected PV modules,
    each possibly at a different irradiance (partial shading), with
    ground truth established from an independent, very fine reference
    sweep -- not from the shading pattern's group count, since merged
    or sub-threshold peaks mean those are not always equal.
    """
    irradiances: list          # W/m^2, one per module
    temperature_C: float
    params: BishopParams = field(default_factory=lambda: DEFAULT_PARAMS)

    # populated by build()
    _V: np.ndarray = field(default=None, repr=False)
    _P: np.ndarray = field(default=None, repr=False)

    n_reference_points: int = 4000

    def build(self):
        n = len(self.irradiances)
        Iscs = [module_Isc(G, self.temperature_C, self.params) for G in self.irradiances]
        I_max = max(Iscs)
        I_grid = np.linspace(1e-6, I_max, self.n_reference_points)

        V_string = np.zeros_like(I_grid)
        for idx, I in enumerate(I_grid):
            v_sum = 0.0
            for G, Isc_i in zip(self.irradiances, Iscs):
                if I >= Isc_i:
                    v_sum += BYPASS_VOLTAGE
                else:
                    v_sum += module_forward_voltage(I, G, self.temperature_C, self.params)
            V_string[idx] = v_sum

        P_string = V_string * I_grid

        # Keep only the physically meaningful (non-decreasing-then-any V) span
        # where string voltage is positive, sorted by voltage for interpolation.
        mask = V_string > 0
        V, P = V_string[mask], P_string[mask]
        order = np.argsort(V)
        self._V, self._P = V[order], P[order]
        return self

    # -- query interface: matches how PSOMbest/TSPSO(EM) parametrise MPPT --
    def power_at_voltage(self, V) -> np.ndarray:
        """Interpolated power at arbitrary query voltage(s)."""
        if self._V is None:
            self.build()
        V = np.atleast_1d(np.asarray(V, dtype=float))
        P = np.interp(V, self._V, self._P, left=0.0, right=0.0)
        return P if P.shape != (1,) else P[0]

    def power_at_voltage_noisy(self, V, noise_std_W: float, rng: np.random.Generator):
        """Same as power_at_voltage but with additive Gaussian measurement noise."""
        P = self.power_at_voltage(V)
        return P + rng.normal(0.0, noise_std_W, size=np.shape(P))

    @property
    def voltage_range(self):
        if self._V is None:
            self.build()
        return float(self._V.min()), float(self._V.max())

    # -- ground truth, from the fine reference sweep, not the shading pattern --
    def ground_truth(self, prominence_W: float = 0.02):
        """
        Peak count/location/global-best established from the dense
        reference sweep via scipy's prominence-based peak detector.
        `prominence_W` is the minimum power drop on both sides required
        for a local maximum to count as a real peak (filters out
        numerical/solver noise, not a stand-in for measurement noise).

        NOTE: this threshold is not a free implementation detail --
        it is the same design parameter discussed at length in the
        concept document (Section 5.3/5.4) as an open, empirically-
        tuned quantity. The bundled adversarial_narrow_peak() scenario
        (see scenarios.py) deliberately has a LOW-prominence local
        peak (~0.45 W rise on a curve peaking near 4.2 W) alongside
        its narrow global peak, specifically so this sensitivity is
        visible and testable here rather than hidden: at prominence_W
        below ~0.03 both peaks are detected; above that, the local
        peak silently disappears from the ground truth. Whatever
        threshold a candidate algorithm's own peak-counting mechanism
        uses should be tested against this same sensitivity, not
        assumed safe.
        """
        if self._V is None:
            self.build()
        peak_idx, _ = find_peaks(self._P, prominence=prominence_W)
        # Also check the boundary in case the true maximum sits at the
        # edge of the swept range (rare, but not structurally impossible).
        if len(peak_idx) == 0:
            peak_idx = np.array([int(np.argmax(self._P))])

        peak_voltages = self._V[peak_idx]
        peak_powers = self._P[peak_idx]
        gmpp_i = int(np.argmax(peak_powers))

        return {
            "peak_count": len(peak_idx),
            "peak_voltages": peak_voltages.tolist(),
            "peak_powers": peak_powers.tolist(),
            "gmpp_voltage": float(peak_voltages[gmpp_i]),
            "gmpp_power": float(peak_powers[gmpp_i]),
        }

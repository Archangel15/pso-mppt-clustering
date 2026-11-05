import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scenarios import uniform, simple_partial, complex_partial, adversarial_narrow_peak, random_scenario

scenarios = [
    ("uniform", *uniform()),
    ("simple_partial", *simple_partial()),
    ("complex_partial", *complex_partial()),
    ("adversarial_narrow_peak", *adversarial_narrow_peak()),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

print(f"{'scenario':28s} {'V_range':18s} {'peaks':6s} {'GMPP_V':8s} {'GMPP_P':8s}")
print("-" * 80)

for ax, (name, curve, label) in zip(axes.flat, scenarios):
    gt = curve.ground_truth(prominence_W=0.02)
    vmin, vmax = curve.voltage_range
    V = np.linspace(vmin, vmax, 2000)
    P = curve.power_at_voltage(V)

    ax.plot(V, P, lw=1.5)
    ax.scatter(gt["peak_voltages"], gt["peak_powers"], color="red", zorder=5, s=40)
    ax.scatter([gt["gmpp_voltage"]], [gt["gmpp_power"]], color="gold",
               edgecolor="black", zorder=6, s=90, marker="*", label="GMPP")
    ax.set_title(f"{name}\n{label}", fontsize=9)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Power (W)")
    ax.legend(fontsize=8)

    print(f"{name:28s} [{vmin:5.2f},{vmax:5.2f}]     "
          f"{gt['peak_count']:<6d} {gt['gmpp_voltage']:<8.2f} {gt['gmpp_power']:<8.2f}")

plt.tight_layout()
plt.savefig("verify_scenarios.png", dpi=110)
print("\nSaved verify_scenarios.png")

# --- Specific check for the adversarial case: is the GMPP peak actually narrow? ---
print("\n--- Adversarial narrow-peak diagnostic ---")
_, curve, _ = ("adversarial_narrow_peak",) + adversarial_narrow_peak()
gt = curve.ground_truth(prominence_W=0.02)
print("All detected peaks (V, P):", list(zip(
    [round(v, 2) for v in gt["peak_voltages"]],
    [round(p, 2) for p in gt["peak_powers"]])))
print("GMPP at V=%.2f, P=%.2f" % (gt["gmpp_voltage"], gt["gmpp_power"]))

# Estimate width of the GMPP peak at half-prominence, to confirm "narrow"
vmin, vmax = curve.voltage_range
Vfine = np.linspace(vmin, vmax, 20000)
Pfine = curve.power_at_voltage(Vfine)
gmpp_idx = np.argmin(np.abs(Vfine - gt["gmpp_voltage"]))
half_level = gt["gmpp_power"] * 0.9
# walk outward from the peak until power drops below half_level
lo = gmpp_idx
while lo > 0 and Pfine[lo] > half_level:
    lo -= 1
hi = gmpp_idx
while hi < len(Pfine) - 1 and Pfine[hi] > half_level:
    hi += 1
width_90pct = Vfine[hi] - Vfine[lo]
print(f"Approx width of GMPP peak at 90%% power level: {width_90pct:.3f} V "
      f"(full sweep range: {vmax - vmin:.2f} V)")

# --- Quick bulk random-scenario sanity check ---
print("\n--- Random scenario batch (seeds 0-4) ---")
for seed in range(5):
    curve, label = random_scenario(n_modules=8, n_groups=4, seed=seed)
    gt = curve.ground_truth(prominence_W=0.02)
    print(f"seed={seed}: peaks={gt['peak_count']}, GMPP=({gt['gmpp_voltage']:.2f} V, "
          f"{gt['gmpp_power']:.2f} W)  [{label}]")

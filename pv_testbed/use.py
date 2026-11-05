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
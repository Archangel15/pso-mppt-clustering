import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ports, scalar_test, time
for kind in scalar_test.KINDS:
    a = ports.run_b1_masterloop("F2", *ports.CASES["F2"], kind, MaxIt=100, seed=3)["best"]
    b = scalar_test.run_scalar(kind, None, "F2", *ports.CASES["F2"], MaxIt=100, seed=3)["best"]
    print(kind, "scalar off reproduces the earlier code:", a == b)
t = time.time(); r = scalar_test.run_scalar("b1_proj_fade", "first", "F1", *ports.CASES["F1"], seed=1)
print("b1_proj_fade + first, F1 seed 1: best =", repr(r["best"]), " evals", r["n_evals"], " time %.1fs" % (time.time() - t))
print("expected on the sandbox: best = 1.7971464851013818e-26 (tiny differences on another PC are possible), evals 150150")

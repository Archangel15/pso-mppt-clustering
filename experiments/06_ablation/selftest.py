import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ports, ablation, time
t = time.time(); r = ablation.run_proj("base", "F2", *ports.CASES["F2"], seed=1)
print("base F2 seed 1 best =", repr(r["best"]), " time %.1fs" % (time.time() - t), " evals", r["n_evals"])

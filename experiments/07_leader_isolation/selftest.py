import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ports, leader_test, time
t = time.time(); r = leader_test.run_leader("scalar_first", "F2", *ports.CASES["F2"], seed=1)
print("scalar_first F2 seed 1 best =", repr(r["best"]), " time %.1fs" % (time.time() - t), " evals", r["n_evals"])

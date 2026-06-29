"""
signal_benchmark.py
Computes a SIGNAL-AWARE but NON-LEARNING benchmark (signal-proportional TWAP)
on the exact H1 evaluation protocol, so its IS is directly comparable to
Table 4.1. Purpose: separate the value of *conditioning on the order-flow
signal* (capturable by a trivial rule) from the value of the *RL policy* itself.

  m(s) = max(0, 1 + beta * signal);   fraction of residual = min(1, m/(N-k))

beta is chosen on a SEPARATE seed (not the test seed) to avoid tuning on the
evaluation set, mirroring the agents' train/test discipline.
"""
import numpy as np
from execution_study import Params
from experiments import eval_policies, newey_west_tstat, summary, REGIMES, N_TEST, TEST_SEED
from strategies import twap_policy


def signal_twap_policy(p, beta):
    def pol(s):
        m = max(0.0, 1.0 + beta * s["sig"])
        return min(1.0, m / (p.N - s["k"]))
    return pol


def main():
    p = Params()
    grid = [0.0, 0.25, 0.5, 0.8, 1.2, 1.6, 2.0]
    # ---- pick beta on a separate (non-test) seed ----
    sel = {f"b{b}": signal_twap_policy(p, b) for b in grid}
    res_sel, _ = eval_policies(p, sel, n=4000, seed=99)
    means = {b: res_sel[f"b{b}"].mean() for b in grid}
    beta = min(means, key=means.get)
    print("beta grid mean IS:", {b: round(float(v), 3) for b, v in means.items()})
    print("selected beta:", beta)

    # ---- evaluate on the held-out H1 protocol (same seed/n as Table 4.1) ----
    pols = {"TWAP": twap_policy(p), "Signal-TWAP": signal_twap_policy(p, beta)}
    res, regs = eval_policies(p, pols, n=N_TEST, seed=TEST_SEED)
    tw, sg = res["TWAP"], res["Signal-TWAP"]
    s = summary(sg)
    pct_vs_twap = 100 * (1 - sg.mean() / tw.mean())
    t = newey_west_tstat(sg - tw)
    print("\n--- Signal-TWAP on H1 protocol (n=%d, seed=%d) ---" % (N_TEST, TEST_SEED))
    print("TWAP mean        : %.2f" % tw.mean())
    print("Signal-TWAP mean : %.2f  median %.2f  VaR95 %.1f" % (s["mean"], s["median"], s["var95"]))
    print("Signal-TWAP vs TWAP: %+.1f%%  (t=%.1f)" % (pct_vs_twap, t))
    # PPO is 11.676 from results.json -> RL increment over the heuristic
    ppo = 11.676711018067113
    print("PPO mean         : %.2f" % ppo)
    print("PPO vs Signal-TWAP: %+.1f%%" % (100 * (1 - ppo / sg.mean())))


if __name__ == "__main__":
    main()
